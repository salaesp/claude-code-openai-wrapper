"""Logging setup + a request-id context var shared across the request lifecycle."""
import contextvars
import logging
import os
import sys

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RidFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = request_id.get()
        return True


def configure() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s [%(rid)s] %(message)s", datefmt="%H:%M:%S"
    ))
    handler.addFilter(_RidFilter())
    root = logging.getLogger("wrapper")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


logger = logging.getLogger("wrapper")
