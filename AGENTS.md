# Repository Guide

## Run and verify

- Create the local environment with `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run locally with `.venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000`, or use `docker compose up --build`. Both paths require an authenticated Claude CLI subscription token and a `WRAPPER_API_KEY` before `/v1` requests succeed.
- There is no automated test, lint, formatter, typecheck, or CI configuration. Use `.venv/bin/python -m compileall app` for a dependency-free syntax check.
- For changes to native structured output, run `python scripts/probe_structured.py [--model MODEL] [--turns N]`; it directly calls the Claude subscription and requires `CLAUDE_CODE_OAUTH_TOKEN`.

## Architecture

- `app/server.py` owns the FastAPI application and OpenAI-compatible routes; `app/models.py` is the supported request/response subset.
- `app/claude_backend.py` selects the `chat`, `tools`, or native `structured` path and translates SDK events; `app/translate.py` renders OpenAI history into Claude prompts.
- Function tools are deliberately captured and returned as OpenAI `tool_calls`; the wrapper must never execute client tools. Built-in Claude Code tools are disabled.
- The API remains stateless at the protocol boundary, but `app/sessions.py` keeps an in-memory, TTL-bound prefix-to-session map to fork/resume append-only conversations. Do not treat it as durable storage.
- `app/pool.py` warm-spawns clients, but every client is single-use to prevent cross-request context bleed.

## Configuration and secrets

- Never commit or expose `.env`, `data/`, `CLAUDE_CODE_OAUTH_TOKEN`, or `WRAPPER_API_KEY`; `data/config.env` is the Docker-mounted secret store.
- Treat `app/config.py` as configuration truth. On `config.reload()`, `.env` respects existing environment values, then `CONFIG_FILE` is loaded with `override=True`; persisted values therefore override process values. This differs from the README and Compose comments.
- Defaults for structured-output turns/retries are also defined in `app/config.py`; `.env.example` currently advertises different values.
- The Docker image needs Node because `claude-agent-sdk` launches the Claude Code CLI; do not remove the CLI installation or make the runtime image Python-only.
