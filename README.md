# Claude → OpenAI-compatible wrapper

Expose your Claude subscription (Pro/Max) as an OpenAI-compatible HTTP API with
**passthrough tool calling** and **structured outputs**. Point any OpenAI SDK / n8n
OpenAI node at it.

> ⚠️ **Terms of service.** This routes traffic through the Claude Code CLI using your
> personal subscription OAuth token. Fine for personal/dev use. Using a consumer
> subscription as a general API proxy — especially reselling or serving third parties —
> may violate Anthropic's terms. You accept that risk. The subscription token grants
> full account access: keep it server-side, never return it to clients.

## How it works

```
OpenAI client ──(Bearer key)──▶ FastAPI wrapper ──▶ claude-agent-sdk ──▶ Claude Code (your session) ──▶ Anthropic
```

- The endpoint is **stateless** — your client resends the full message history each
  call (standard OpenAI behavior). No Claude session is kept between requests.
- **Tools:** each OpenAI tool is registered as an in-process MCP tool so Claude knows
  its schema. A `can_use_tool` hook **captures** the arguments Claude wants to pass and
  **halts** before executing anything — the captured args are returned to your client
  as `tool_calls`. Your client executes them and sends results back on the next request.
- **Structured output:** `response_format: json_schema` registers one forced schema
  tool; its captured args are returned as JSON `content`.
- Claude Code's built-in tools (Bash/Read/Edit/…) are disabled — this is a pure model
  endpoint, not an autonomous agent.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Authenticate the CLI with your SUBSCRIPTION (not an API key).
#    Clear any API key so the CLI uses your account session:
unset ANTHROPIC_API_KEY
claude setup-token          # opens browser; creates a long-lived OAuth token in ~/.claude

# 2) Set the wrapper's own gateway key (what your OpenAI clients present):
export WRAPPER_API_KEY="sk-local-$(openssl rand -hex 16)"

# 3) Run
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### Config (env vars)

| var | default | meaning |
|-----|---------|---------|
| `WRAPPER_API_KEY` | `changeme` | key clients send as `Authorization: Bearer …` |
| `CLAUDE_MODEL` | `claude-opus-4-8` | default model id |
| `EXPOSED_MODELS` | opus/sonnet/haiku | ids listed at `GET /v1/models` |
| `CLAUDE_MAX_TURNS` | `8` | agent turn ceiling per request |
| `REQUEST_TIMEOUT` | `180` | seconds per request |

## Use it

Plain chat:
```bash
curl localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $WRAPPER_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}'
```

OpenAI Python SDK:
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_WRAPPER_API_KEY")
c.chat.completions.create(model="claude-opus-4-8",
    messages=[{"role":"user","content":"2+2?"}])
```

Tool call (passthrough) + structured output examples are in `examples/`.

## Docker

The image bundles Python + the Claude Code CLI (Node). Subscription auth is passed as
a token env var — the CLI writes nothing to Keychain inside the container.

### Easiest: the setup page (no env editing)

```bash
docker compose up --build          # env vars optional; leave .env empty
open http://localhost:8000/setup   # browser opens the setup form
```

On the page you set your **gateway key** (invent one) and paste a **subscription
token**. Get the token on your host (needs a browser):

```bash
unset ANTHROPIC_API_KEY
claude setup-token                 # prints a long-lived token — paste it into the page
```

Both values are written to `./data/config.env` (a mounted volume) and take effect
immediately — no restart. The page is open until configured, then locked behind your
gateway key. Visit `http://localhost:8000/` and it redirects to setup until done.

### Or preseed via env

```bash
cp .env.example .env
# fill WRAPPER_API_KEY, then:
claude setup-token                 # paste output into CLAUDE_CODE_OAUTH_TOKEN in .env
docker compose up --build
```

Env vars, when set, win over the persisted `data/config.env`.

Plain `docker`:
```bash
docker build -t claude-openai .
docker run -p 8000:8000 \
  -e WRAPPER_API_KEY="$WRAPPER_API_KEY" \
  -e CLAUDE_CODE_OAUTH_TOKEN="$(cat token.txt)" \
  claude-openai
```

`CLAUDE_CODE_OAUTH_TOKEN` grants full account access — inject at runtime only, never
bake into the image or commit `.env`.

## Behavior notes

Three request modes, each tuned differently (all configurable in `/setup`):

- **Plain chat** — single turn, one prompt → one response. No tools, no ToolSearch.
- **Structured output** (`response_format`) — uses the CLI's **native** `--json-schema`
  output. No MCP tool, no ToolSearch; the JSON comes back in `structured_output` and is
  returned as the message content. Clean and reliable.
- **Passthrough function tools** (OpenAI `tools`) — the only mode that pays an internal
  `ToolSearch` round-trip: this CLI registers MCP tools as *deferred*, so the model calls
  `ToolSearch` to load the tool, then calls it (~4 turns). Inherent to function-calling
  passthrough; bounded by the **Tool/structured turn limit** (`TOOL_MAX_TURNS`, default 5).

Other notes:

- **`SINGLE_TURN` is chat-only.** Tool/structured requests use `TOOL_MAX_TURNS` so their
  internal steps can complete; forcing 1 turn there returns empty responses.
- **`error=True` in `result:` logs is expected for passthrough tool calls** — we capture
  the model's tool call by denying+interrupting it, which the SDK records as an error. The
  response is still correct. (Structured mode logs `error=False`.)
- **Set log level to `DEBUG`** (in `/setup` or `LOG_LEVEL`) to see the prompt, per-block
  tool_use, and the SDK result text.

## Known limitations

- **No token usage counts** — subscription auth doesn't surface them; `usage` is zeros.
- **Streaming is text-incremental**; tool_call deltas arrive as one chunk each, at end.
- **`temperature`/`max_tokens`** are accepted but not all forwarded (SDK-dependent).
- **Rate limits** are your subscription's; concurrency is bounded by the CLI.
- **Tool ordering:** prior tool results are replayed as text context, not native
  `tool_result` blocks — near-identical behavior for typical single-tool turns.
