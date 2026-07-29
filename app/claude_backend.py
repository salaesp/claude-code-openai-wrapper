"""Drive the Claude Agent SDK and translate its stream into OpenAI-shaped events.

Passthrough tool calling works like this:
  1. Each OpenAI tool is registered as an in-process MCP tool so Claude knows its
     name + JSON schema and can decide to call it.
  2. The tools are deliberately NOT in `allowed_tools`, so every attempt routes
     through `can_use_tool`.
  3. `can_use_tool` captures the arguments Claude wanted to pass, then denies with
     `interrupt=True` — halting the agent instead of executing anything. The captured
     args become the OpenAI `tool_calls` we return to the client, whose job it is to
     execute them and send results back on the next (stateless) request.

Built-in Claude Code tools (Bash/Read/Edit/...) are denied unconditionally so the
wrapper behaves as a pure model endpoint, not an autonomous agent.

Events yielded by `run()`:
  ("text", str)        incremental assistant text
  ("tool_calls", list) client must execute these; turn is over
  ("done", None)       end of turn (text path)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, AsyncIterator

from .log import logger

# The SDK spawns a second node process (`claude -v`) on every connect just to
# check the version. Skip it: real startup-latency win, zero behavior change.
os.environ.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    create_sdk_mcp_server,
    tool,
)

from . import config
from .models import ChatCompletionRequest
from .translate import build_prompt, build_tail_prompt

MCP_SERVER = "openai"
TOOL_PREFIX = f"mcp__{MCP_SERVER}__"


def _extract_usage(usage) -> dict:
    """Map Claude usage -> OpenAI usage. Accepts a dict or an object; tolerates missing."""
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    g = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    inp = (g("input_tokens", 0) or 0)
    cache_read = (g("cache_read_input_tokens", 0) or 0)
    cache_create = (g("cache_creation_input_tokens", 0) or 0)
    out = (g("output_tokens", 0) or 0)
    prompt = inp + cache_read + cache_create
    return {
        "prompt_tokens": prompt,
        "completion_tokens": out,
        "total_tokens": prompt + out,
        # breakdown (underscored: dropped before returning OpenAI usage to client)
        "_input": inp,
        "_cache_read": cache_read,
        "_cache_create": cache_create,
    }


def _gen_id(prefix: str, n: int) -> str:
    # deterministic-ish id without Math.random/Date; good enough for a call id
    return f"{prefix}_{abs(hash((prefix, n))) % (10**12):012d}"


class CaptureHolder:
    """Mutable indirection for per-request capture state.

    A warm client's closures are bound at connect time, possibly by an earlier
    request with the same options key. They read `holder.capture`, which the
    consuming request swaps to its own dict at checkout.
    """
    __slots__ = ("capture",)

    def __init__(self) -> None:
        self.capture: dict = {}


def _request_key(req: ChatCompletionRequest, mode: str) -> str:
    """Hash of everything that influences the spawned CLI process.

    Two requests with the same key produce byte-identical CLI invocations, so a
    client pre-spawned for one can safely serve the other.
    """
    system_prompt, _ = build_prompt(req.messages)
    uses_tools = bool(req.tools) and req.tool_choice != "none"
    sig = {
        "mode": mode,
        "model": req.model,
        "system": system_prompt,
        "schema": (req.response_format.json_schema or {}) if req.response_format else None,
        "rf_type": req.response_format.type if req.response_format else None,
        "tools": [t.model_dump() for t in req.tools] if uses_tools else None,
        "tool_choice": req.tool_choice if uses_tools and isinstance(req.tool_choice, str) else
                       (json.dumps(req.tool_choice, sort_keys=True) if uses_tools and req.tool_choice else None),
        "effort": req.reasoning_effort,
        # config knobs that alter options at build time
        "single_turn": config.SINGLE_TURN,
        "concise": config.CONCISE,
        "thinking_off": config.DISABLE_THINKING,
        "tool_max_turns": config.TOOL_MAX_TURNS,
        "max_turns": config.MAX_TURNS,
    }
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


BUILTIN_TOOLS = ["Bash", "Read", "Edit", "Write", "WebFetch", "WebSearch",
                 "Glob", "Grep", "Task", "NotebookEdit"]

# Reframes the model as a plain LLM. Claude Code otherwise behaves like a coding
# agent ("let me diagnose / verify / search the files") and references tools that
# don't exist in this context. Appended to the user's own system prompt.
LLM_GUARDRAIL = (
    "You are a helpful AI assistant accessed through a plain text API. "
    "You have NO tools, NO file system, NO terminal, and NO ability to run commands, "
    "read files, browse, or inspect any environment. Never say you will diagnose, "
    "investigate, verify behavior, search files, or use any tool. Answer directly and "
    "completely using only your own knowledge and this conversation. If something is "
    "outside your knowledge, say so plainly instead of pretending to look it up."
)


CONCISE_DIRECTIVE = (
    "Respond with the final answer directly and concisely. Do not narrate your reasoning, "
    "show step-by-step work, or add verification unless the user explicitly asks for it."
)


def _sanitize_schema(schema: dict) -> dict:
    """Make a client-supplied JSON Schema safe for the CLI's draft-07 validator.

    OpenAI clients often emit `$schema: .../2020-12`, which the SDK rejects
    outright (it validates as draft-07 and refuses newer declarations). The
    keyword is only a version annotation — dropping it keeps semantics.
    """
    if "$schema" in schema:
        schema = {k: v for k, v in schema.items() if k != "$schema"}
    return schema


# OpenAI reasoning_effort -> CLI --effort. "minimal" has no CLI equivalent; low
# is the floor. Unknown values are ignored (OpenAI-style: tolerate, don't reject).
_EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium",
               "high": "high", "xhigh": "xhigh"}


def _with_guardrail(system_prompt: str) -> str:
    parts = [system_prompt.strip()] if system_prompt else []
    parts.append(LLM_GUARDRAIL)
    if config.CONCISE:
        parts.append(CONCISE_DIRECTIVE)
    return "\n\n".join(parts)


def _make_options(req: ChatCompletionRequest, holder: CaptureHolder,
                  resume: str | None = None) -> ClaudeAgentOptions:
    """Build SDK options. Three modes:

    - structured: native --json-schema output (no MCP, no ToolSearch, single exchange)
    - tools:      passthrough via MCP + can_use_tool capture (needs ToolSearch turns)
    - chat:       plain model call, single turn
    """
    system_prompt, _ = build_prompt(req.messages)
    is_structured = bool(req.response_format and req.response_format.type in ("json_schema", "json_object"))

    base: dict[str, Any] = dict(
        setting_sources=[],
        permission_mode="default",
        tools=[],                    # --tools "" : strip ALL built-in tools + their defs
        disallowed_tools=BUILTIN_TOOLS,   # belt-and-suspenders
        strict_mcp_config=True,      # don't scan user/project MCP configs
        env={"DISABLE_AUTOUPDATER": "1",
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},  # cut CLI startup work
    )
    if config.DISABLE_THINKING:
        base["thinking"] = {"type": "disabled"}
    if req.model:
        base["model"] = req.model
    if req.reasoning_effort:
        effort = _EFFORT_MAP.get(req.reasoning_effort)
        if effort:
            base["effort"] = effort
        else:
            logger.debug("ignoring unknown reasoning_effort=%r", req.reasoning_effort)
    if resume:
        # Fork on resume: each request gets its own copy of the warmed session,
        # so a client branching from the same prefix never contaminates siblings.
        base["resume"] = resume
        base["fork_session"] = True

    def build(opts: dict) -> ClaudeAgentOptions:
        merged = {**base, **opts}
        return ClaudeAgentOptions(**{k: v for k, v in merged.items() if v is not None})

    # === structured output: native, no tools, one answer ===
    if is_structured:
        schema = {"type": "object"}
        if req.response_format.type == "json_schema" and req.response_format.json_schema:
            schema = req.response_format.json_schema.get("schema", schema)
        schema = _sanitize_schema(schema)
        logger.debug("structured native output_format schema=%s", json.dumps(schema)[:300])
        return build(dict(
            output_format={"type": "json_schema", "schema": schema},
            allowed_tools=[],   # no tools -> model answers directly from the prompt
            max_turns=config.STRUCTURED_MAX_TURNS,
            system_prompt=_with_guardrail(system_prompt),
        ))

    # === passthrough function tools ===
    if req.tools and req.tool_choice != "none":
        sdk_tools, tool_names = [], []
        for t in req.tools:
            name = t.function.name
            tschema = t.function.parameters or {"type": "object"}

            @tool(name, t.function.description or name, tschema)
            async def _passthrough(args):  # intercepted before execution
                return {"content": [{"type": "text", "text": "ok"}]}

            sdk_tools.append(_passthrough)
            tool_names.append(TOOL_PREFIX + name)

        forced = req.tool_choice in ("required", "any") or isinstance(req.tool_choice, dict)
        if forced:
            system_prompt += "\n\nYou MUST call one of the provided tools to respond."
        server = create_sdk_mcp_server(name=MCP_SERVER, tools=sdk_tools)
        logger.debug("registered passthrough tools: %s", tool_names)

        async def can_use_tool(tool_name: str, tool_input: dict, ctx) -> Any:
            capture = holder.capture   # indirection: rebound per request at checkout
            if tool_name.startswith(TOOL_PREFIX):
                short = tool_name[len(TOOL_PREFIX):]
                capture.setdefault("calls", []).append({"name": short, "args": tool_input})
                logger.info("captured tool_call: %s", short)
                logger.debug("tool_call args: %s", json.dumps(tool_input)[:400])
                return PermissionResultDeny(behavior="deny", message="captured", interrupt=True)
            capture.setdefault("blocked", []).append(tool_name)
            logger.warning("blocked non-passthrough tool: %s", tool_name)
            return PermissionResultDeny(behavior="deny", message="tool disabled", interrupt=True)

        return build(dict(
            system_prompt=system_prompt or None,
            max_turns=config.TOOL_MAX_TURNS,   # ToolSearch round-trip needs a few turns
            can_use_tool=can_use_tool,
            mcp_servers={MCP_SERVER: server},
        ))

    # === plain chat: single turn ===
    async def deny_all(tool_name, tool_input, ctx):
        holder.capture.setdefault("blocked", []).append(tool_name)
        return PermissionResultDeny(behavior="deny", message="tool disabled", interrupt=True)

    return build(dict(
        system_prompt=_with_guardrail(system_prompt),
        max_turns=1 if config.SINGLE_TURN else config.MAX_TURNS,
        can_use_tool=deny_all,
        include_partial_messages=True,   # emit StreamEvent text deltas for real streaming
    ))


async def run(req: ChatCompletionRequest) -> AsyncIterator[tuple[str, Any]]:
    """Yield ("text", str) / ("tool_calls", list) / ("done", None)."""
    from .pool import pool
    from .sessions import canon_message, sessions

    run_started = time.perf_counter()
    capture: dict = {}

    n_tools = len(req.tools or []) if req.tool_choice != "none" else 0
    mode = "structured" if req.response_format and req.response_format.type != "text" else \
           ("tools" if n_tools else "chat")
    forced = req.tool_choice in ("required", "any") or isinstance(req.tool_choice, dict) \
        or mode == "structured"
    # Report the turn budget each mode actually runs with (see _make_options):
    # structured -> STRUCTURED_MAX_TURNS, tools -> TOOL_MAX_TURNS, chat -> single/MAX.
    eff_turns = config.STRUCTURED_MAX_TURNS if mode == "structured" else \
        config.TOOL_MAX_TURNS if mode == "tools" else \
        (1 if config.SINGLE_TURN else config.MAX_TURNS)

    # Session resume: if the incoming history extends a conversation we've served,
    # resume (a fork of) that Claude session and send only the new tail. Native
    # structured output can safely fall back to a full replay if resume fails.
    resume_session: str | None = None
    if mode in ("chat", "tools", "structured"):
        hit = sessions.match(req.messages)
        if hit:
            resume_session, split = hit
            user_prompt = build_tail_prompt(
                req.messages[split:], tools_available=mode == "tools"
            )
    if resume_session is None:
        _, user_prompt = build_prompt(req.messages, tools_available=mode == "tools")

    key = _request_key(req, mode)

    async def factory(resume: str | None = None):
        holder = CaptureHolder()
        options = _make_options(req, holder, resume=resume)
        client = ClaudeSDKClient(options=options)
        await client.connect()
        return client, holder

    client_setup_started = time.perf_counter()
    if resume_session:
        # Session-specific spawn: the warm pool can't pre-build these.
        try:
            client, holder = await factory(resume_session)
            warm = False
        except Exception as e:
            logger.warning("session resume connect failed (%s: %s); falling back to full replay",
                           type(e).__name__, e)
            resume_session = None
            _, user_prompt = build_prompt(req.messages, tools_available=mode == "tools")
    if resume_session is None:
        client, holder, warm = await pool.checkout(key, factory)
        pool.schedule_refill(key, factory)   # replacement spawns while this request runs
    client_setup_ms = (time.perf_counter() - client_setup_started) * 1000

    logger.info(
        "request: model=%s mode=%s messages=%d tools=%d forced=%s max_turns=%d warm=%s "
        "resume=%s prompt_chars=%d client_setup=%.0fms",
        req.model, mode, len(req.messages), n_tools, forced, eff_turns, warm,
        resume_session or "-", len(user_prompt), client_setup_ms,
    )
    logger.debug("tool_choice=%s response_format=%s prompt=%r",
                 req.tool_choice,
                 req.response_format.type if req.response_format else None,
                 user_prompt[:800])

    from claude_agent_sdk import (
        AssistantMessage, TextBlock, ThinkingBlock, ToolUseBlock,
        SystemMessage, ResultMessage, StreamEvent,
    )

    # For tool/structured requests, buffer text instead of streaming it: any text
    # before a captured tool call is just ToolSearch preamble and must be dropped.
    buffering = mode != "chat"
    # chat streams token deltas live via StreamEvent; buf is a fallback if none arrive
    stream_text = mode == "chat"
    buf: list[str] = []

    t0 = time.perf_counter()
    text_chars = 0
    emitted_text = False
    first_assistant_ms: float | None = None

    async def drain(cli, hld, wrm):
        """Query + consume one full response. Yields text events; fills `capture`.
        Handles a stale-warm client by reconnecting cold once. Retires its client."""
        nonlocal text_chars, emitted_text, first_assistant_ms
        hld.capture = capture
        try:
            try:
                await cli.query(user_prompt)
            except Exception as e:
                if not wrm:
                    raise
                logger.warning("warm client stale (%s); retrying cold", e)
                pool.schedule_retire(cli)
                cli, hld = await factory()
                hld.capture = capture
                await cli.query(user_prompt)

            async for msg in cli.receive_response():
                if isinstance(msg, StreamEvent):
                    if stream_text:
                        ev = msg.event or {}
                        if ev.get("type") == "content_block_delta":
                            delta = ev.get("delta") or {}
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                if first_assistant_ms is None:
                                    first_assistant_ms = (time.perf_counter() - run_started) * 1000
                                emitted_text = True
                                text_chars += len(delta["text"])
                                yield ("text", delta["text"])
                elif isinstance(msg, AssistantMessage):
                    if first_assistant_ms is None:
                        first_assistant_ms = (time.perf_counter() - run_started) * 1000
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text:
                            if stream_text:
                                buf.append(block.text)   # fallback only; deltas already sent
                            elif buffering:
                                text_chars += len(block.text)
                                buf.append(block.text)
                        elif isinstance(block, ThinkingBlock):
                            logger.debug("thinking: %d chars", len(getattr(block, "thinking", "") or ""))
                        elif isinstance(block, ToolUseBlock):
                            logger.debug("assistant tool_use block: name=%s input=%s",
                                         getattr(block, "name", "?"),
                                         json.dumps(getattr(block, "input", {}))[:300])
                elif isinstance(msg, SystemMessage):
                    logger.debug("system message: %s", getattr(msg, "subtype", ""))
                elif isinstance(msg, ResultMessage):
                    capture["usage"] = _extract_usage(getattr(msg, "usage", None))
                    capture["stop_reason"] = getattr(msg, "stop_reason", None)
                    capture["subtype"] = getattr(msg, "subtype", None)
                    capture["session_id"] = getattr(msg, "session_id", None)
                    capture["structured_output"] = getattr(msg, "structured_output", None)
                    denials = getattr(msg, "permission_denials", None) or []
                    result_text = getattr(msg, "result", None)
                    u = capture["usage"]
                    logger.info(
                        "result: turns=%s stop=%s error=%s denials=%d cost=$%s "
                        "tokens=out:%s in:%s cache_read:%s cache_create:%s",
                        getattr(msg, "num_turns", "?"), capture["stop_reason"],
                        getattr(msg, "is_error", "?"), len(denials),
                        getattr(msg, "total_cost_usd", "?"),
                        u.get("completion_tokens"), u.get("_input"),
                        u.get("_cache_read"), u.get("_cache_create"),
                    )
                    if capture["structured_output"] is not None:
                        logger.debug("native structured_output: %s",
                                     json.dumps(capture["structured_output"])[:300])
                    if result_text:
                        logger.debug("result text: %r", str(result_text)[:400])
                    break
        finally:
            pool.schedule_retire(cli)

    # Resume path runs once, with fallback to full replay if the session is gone
    # (evicted JSONL, restarted container) — safe only before any text streamed.
    resume_done = False
    if resume_session:
        try:
            async for ev in drain(client, holder, False):
                yield ev
            resume_done = True
        except Exception as e:
            if emitted_text:
                raise   # already streamed to the client; can't restart cleanly
            logger.warning("session resume failed (%s: %s); falling back to full replay",
                           type(e).__name__, e)
            capture.clear()
            buf.clear()
            resume_session = None
            _, user_prompt = build_prompt(req.messages, tools_available=mode == "tools")
            client, holder, warm = await pool.checkout(key, factory)
            pool.schedule_refill(key, factory)

    # Structured output may need whole-request retries: if the model's StructuredOutput
    # call fails schema validation and it runs out of turns, structured_output is empty.
    # Chat/tools stream during drain, so they run exactly once.
    attempts = config.STRUCTURED_RETRIES if mode == "structured" else 1
    for attempt in range(max(1, attempts)) if not resume_done else []:
        if attempt == 0:
            cli, hld, wrm = client, holder, warm
        else:
            logger.warning("structured retry %d/%d (prev stop=%s, empty output)",
                           attempt, attempts - 1, capture.get("stop_reason"))
            capture.clear()
            buf.clear()
            cli, hld = await factory()
            wrm = False
        async for ev in drain(cli, hld, wrm):
            yield ev
        if mode != "structured" or capture.get("structured_output") is not None:
            break

    calls = capture.get("calls") or []
    blocked = capture.get("blocked") or []
    structured = capture.get("structured_output")
    elapsed = (time.perf_counter() - t0) * 1000
    total_elapsed = (time.perf_counter() - run_started) * 1000
    first_assistant = f"{first_assistant_ms:.0f}ms" if first_assistant_ms is not None else "-"
    outcome = "structured" if mode == "structured" else ("tool_calls" if calls else "text")
    logger.info(
        "done: outcome=%s text_chars=%d captured=%d blocked=%d first_assistant=%s "
        "elapsed=%.0fms total=%.0fms",
        outcome, text_chars, len(calls), len(blocked), first_assistant, elapsed, total_elapsed,
    )

    # Remember this conversation's session so the client's next request (history +
    # our reply) resumes it instead of replaying everything.
    if mode in ("chat", "tools", "structured") and capture.get("session_id"):
        reply = None
        if calls:
            reply = canon_message(
                "assistant", "",
                [(c["name"], json.dumps(c["args"])) for c in calls])
        elif mode == "structured" and structured is not None:
            reply = canon_message("assistant", json.dumps(structured))
        elif mode != "structured":
            reply = canon_message("assistant", "".join(buf))
        if reply is not None:
            sessions.store(req.messages, reply, capture["session_id"])

    usage = capture.get("usage")
    if usage:
        # drop internal breakdown keys; client gets clean OpenAI usage
        yield ("usage", {k: v for k, v in usage.items() if not k.startswith("_")})

    # structured output -> native JSON from the CLI, returned as message content
    if mode == "structured":
        if structured is not None:
            yield ("text", json.dumps(structured))
        else:
            # never return the model's preamble as if it were JSON — surface a clean error
            subtype = capture.get("subtype")
            logger.warning("structured output empty after %d attempt(s) (stop=%s subtype=%s); returning error",
                           max(1, attempts), capture.get("stop_reason"), subtype)
            if subtype == "error_max_structured_output_retries":
                message = ("Model output failed schema validation after the SDK's "
                           "retry limit. Simplify the schema or retry.")
            else:
                message = "Model did not produce output matching the requested schema."
            yield ("error", {
                "message": message, "type": "api_error",
                "code": subtype or "structured_output_failed"})
        yield ("done", None)
        return

    if calls:
        tool_calls = [
            {
                "id": _gen_id("call", i),
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c["args"]),
                },
            }
            for i, c in enumerate(calls)
        ]
        yield ("tool_calls", tool_calls)
        return

    # no tool captured. If this was a forced request, that's a failure worth flagging.
    if forced:
        logger.warning(
            "FORCED but no tool captured. mode=%s stop=%s blocked=%s buffered_chars=%d. "
            "Falling back to buffered text (may not satisfy the client's schema).",
            mode, capture.get("stop_reason"), blocked or "-", len(" ".join(buf)),
        )

    # emit buffered text (tool fallback) or, for chat, the fallback if no deltas streamed
    if buffering:
        yield ("text", "".join(buf))
    elif not emitted_text:
        # chat produced no StreamEvent deltas -> emit the complete text (or empty)
        yield ("text", "".join(buf))
    yield ("done", None)
