"""Warm pool of pre-spawned, single-use Claude CLI clients.

Why single-use: the CLI ignores session_id on stdin user messages — one process is
one conversation, so reusing a live client would bleed context between requests.
The latency win comes from moving the ~1.5-2.5s process spawn OFF the request path:
after a request checks out the warm client for its options-key, a replacement is
spawned in the background while the request runs.

One warm client per options-key; at most POOL_MAX_KEYS keys (LRU eviction); idle
entries are retired after POOL_TTL_S by the sweeper.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from . import config
from .log import logger

# factory: spawns + connects a client; returns (client, holder)
Factory = Callable[[], Awaitable[tuple[Any, Any]]]


class WarmPool:
    def __init__(self) -> None:
        self._slots: dict[str, tuple[Any, Any, float]] = {}  # key -> (client, holder, ts)
        self._lock = asyncio.Lock()
        self._refilling: set[str] = set()

    async def checkout(self, key: str, factory: Factory) -> tuple[Any, Any, bool]:
        """Return (client, holder, warm). Warm hit pops the slot; miss connects cold."""
        if config.WARM_POOL:
            async with self._lock:
                entry = self._slots.pop(key, None)
            if entry is not None:
                client, holder, _ = entry
                return client, holder, True
        client, holder = await factory()
        return client, holder, False

    def schedule_refill(self, key: str, factory: Factory) -> None:
        """Spawn a replacement client in the background (skipped if slot is filled)."""
        if not config.WARM_POOL or key in self._refilling:
            return
        self._refilling.add(key)
        asyncio.get_running_loop().create_task(self._refill(key, factory))

    async def _refill(self, key: str, factory: Factory) -> None:
        try:
            async with self._lock:
                if key in self._slots:
                    return
            client, holder = await factory()
            async with self._lock:
                if key in self._slots:  # raced with another refill
                    self.schedule_retire(client)
                    return
                self._slots[key] = (client, holder, time.monotonic())
                # LRU-evict beyond POOL_MAX_KEYS (oldest ts first)
                while len(self._slots) > config.POOL_MAX_KEYS:
                    oldest = min(self._slots, key=lambda k: self._slots[k][2])
                    old_client, _, _ = self._slots.pop(oldest)
                    self.schedule_retire(old_client)
            logger.debug("pool: warmed key=%s (keys=%d)", key, len(self._slots))
        except Exception as e:
            logger.warning("pool: refill failed for key=%s: %s", key, e)
        finally:
            self._refilling.discard(key)

    def schedule_retire(self, client: Any) -> None:
        """Fire-and-forget disconnect; single-use cleanup for every served client."""
        async def _retire():
            try:
                async with asyncio.timeout(10):
                    await client.disconnect()
            except Exception as e:
                logger.debug("pool: retire error: %s", e)
        try:
            asyncio.get_running_loop().create_task(_retire())
        except RuntimeError:  # no running loop (shutdown edge)
            pass

    async def sweep_loop(self) -> None:
        """Retire idle warm clients past POOL_TTL_S."""
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            async with self._lock:
                stale = [k for k, (_, _, ts) in self._slots.items()
                         if now - ts > config.POOL_TTL_S]
                for k in stale:
                    client, _, _ = self._slots.pop(k)
                    self.schedule_retire(client)
            if stale:
                logger.debug("pool: swept %d idle clients", len(stale))

    async def shutdown(self) -> None:
        async with self._lock:
            entries = list(self._slots.values())
            self._slots.clear()
        for client, _, _ in entries:
            try:
                async with asyncio.timeout(10):
                    await client.disconnect()
            except Exception:
                pass


pool = WarmPool()
