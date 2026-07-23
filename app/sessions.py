"""Session-resume map: reuse a Claude session across stateless OpenAI requests.

The OpenAI protocol carries no session ID — clients resend the full history each
call. But a well-behaved client only ever APPENDS: request N's history is request
N-1's history plus the last exchange. That makes the conversation prefix itself a
usable key.

At response time we predict the key the client's NEXT request will hash to
(current history + the reply we just returned) and store session_id under it.
On the next request we hash trailing prefixes of the incoming history; a hit
means "this conversation continues Claude session X" — the backend resumes it
(forked, so branches never contaminate the original) and sends only the delta
messages instead of replaying everything.

A miss is always safe: fall back to the stateless full-replay path.
"""
from __future__ import annotations

import hashlib
import json
import time

from . import config
from .log import logger
from .models import Message
from .translate import _content_to_text


def canon_message(role: str, content_text: str,
                  tool_calls: list[tuple[str, str]] | None = None) -> dict:
    """Canonical form of one message for hashing.

    Deliberately lenient: tool-call IDs, `name` fields, and multimodal structure
    are excluded so cosmetic client-side differences don't break the match.
    """
    c: dict = {"r": role, "c": content_text}
    if tool_calls:
        c["t"] = tool_calls  # [(function_name, arguments_json_string), ...]
    return c


def canon_from_message(m: Message) -> dict:
    tcs = [(tc.function.name, tc.function.arguments) for tc in (m.tool_calls or [])] or None
    return canon_message(m.role, _content_to_text(m.content), tcs)


def history_key(canon_list: list[dict]) -> str:
    blob = json.dumps(canon_list, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


# How many trailing messages may separate two consecutive requests: one assistant
# reply, or an assistant tool_calls turn plus a handful of tool results.
MAX_TAIL = 6


class SessionMap:
    """In-memory {history_key -> (session_id, ts)} with TTL + size cap."""

    def __init__(self) -> None:
        self._map: dict[str, tuple[str, float]] = {}

    def match(self, messages: list[Message]) -> tuple[str, int] | None:
        """Longest stored prefix of `messages`. Returns (session_id, split_index):
        messages[:split_index] is already inside the session, messages[split_index:]
        is the unsent tail. Longest prefix first = smallest tail."""
        if not config.SESSION_RESUME or len(self._map) == 0:
            return None
        self._prune()
        canon = [canon_from_message(m) for m in messages]
        n = len(canon)
        # tail must be non-empty (something new to send) and small (append-only client)
        for split in range(n - 1, max(n - 1 - MAX_TAIL, 0) - 1, -1):
            entry = self._map.get(history_key(canon[:split]))
            if entry is not None:
                return entry[0], split
        return None

    def store(self, messages: list[Message], reply_canon: dict, session_id: str) -> None:
        """Predict the next request's prefix key (history + our reply) and remember it."""
        if not config.SESSION_RESUME or not session_id:
            return
        canon = [canon_from_message(m) for m in messages] + [reply_canon]
        key = history_key(canon)
        self._map[key] = (session_id, time.monotonic())
        self._prune()
        logger.debug("sessions: stored key=%s session=%s (entries=%d)",
                     key, session_id, len(self._map))

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._map.items()
                   if now - ts > config.SESSION_TTL_S]
        for k in expired:
            del self._map[k]
        while len(self._map) > config.SESSION_MAX:
            oldest = min(self._map, key=lambda k: self._map[k][1])
            del self._map[oldest]


sessions = SessionMap()
