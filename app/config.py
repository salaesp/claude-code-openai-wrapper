"""Runtime config. Layered: process env > persisted config file > ./.env > defaults.

The /setup page writes secrets to CONFIG_FILE (a mounted volume in Docker) so they
survive restarts without editing env vars. config.reload() re-reads it live.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # dotenv optional
    def load_dotenv(*a, **k):  # type: ignore
        return False

# Where /setup persists WRAPPER_API_KEY + CLAUDE_CODE_OAUTH_TOKEN.
CONFIG_FILE = os.environ.get("CONFIG_FILE", "")


def _apply() -> None:
    """Recompute module-level settings from os.environ."""
    global API_KEY, DEFAULT_MODEL, EXPOSED_MODELS, MAX_TURNS, REQUEST_TIMEOUT
    global SINGLE_TURN, DISABLE_THINKING, TOOL_MAX_TURNS
    API_KEY = os.environ.get("WRAPPER_API_KEY", "changeme")
    DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")
    EXPOSED_MODELS = os.environ.get(
        "EXPOSED_MODELS", "claude-opus-4-8,claude-sonnet-5,claude-haiku-4-5"
    ).split(",")
    MAX_TURNS = int(os.environ.get("CLAUDE_MAX_TURNS", "8"))
    # Ceiling for tool/structured requests: enough for the ToolSearch round-trip
    # (~4 turns) plus a little slack. Lower = less room for agentic wandering.
    TOOL_MAX_TURNS = int(os.environ.get("TOOL_MAX_TURNS", "5"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "180"))
    # LLM mode: behave like a plain model (one prompt -> one response), not an agent.
    SINGLE_TURN = os.environ.get("SINGLE_TURN", "1") not in ("0", "false", "False")
    DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "1") not in ("0", "false", "False")


def reload() -> None:
    """Reload ./.env then the persisted config file (persisted wins), then re-apply."""
    load_dotenv()  # ./.env, does not override existing process env
    if CONFIG_FILE and Path(CONFIG_FILE).exists():
        load_dotenv(CONFIG_FILE, override=True)
    _apply()


def is_configured() -> bool:
    """True once a real gateway key + subscription token are present."""
    return API_KEY not in ("", "changeme") and bool(
        os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    )


reload()  # initial load at import
