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

import json
import time
from typing import Any, AsyncIterator

from .log import logger
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
from .translate import build_prompt

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
    }


def _gen_id(prefix: str, n: int) -> str:
    # deterministic-ish id without Math.random/Date; good enough for a call id
    return f"{prefix}_{abs(hash((prefix, n))) % (10**12):012d}"


BUILTIN_TOOLS = ["Bash", "Read", "Edit", "Write", "WebFetch", "WebSearch",
                 "Glob", "Grep", "LS", "Task", "NotebookEdit"]


def _make_options(req: ChatCompletionRequest, capture: dict) -> ClaudeAgentOptions:
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
        disallowed_tools=BUILTIN_TOOLS,
    )
    if config.DISABLE_THINKING:
        base["thinking"] = {"type": "disabled"}
    if req.model:
        base["model"] = req.model

    def build(opts: dict) -> ClaudeAgentOptions:
        merged = {**base, **opts}
        return ClaudeAgentOptions(**{k: v for k, v in merged.items() if v is not None})

    # === structured output: native, no tools, one answer ===
    if is_structured:
        schema = {"type": "object"}
        if req.response_format.type == "json_schema" and req.response_format.json_schema:
            schema = req.response_format.json_schema.get("schema", schema)
        logger.debug("structured native output_format schema=%s", json.dumps(schema)[:300])
        return build(dict(
            output_format={"type": "json_schema", "schema": schema},
            allowed_tools=[],   # no tools -> model answers directly from the prompt
            max_turns=config.TOOL_MAX_TURNS,   # native StructuredOutput uses ~2 turns
            system_prompt=(system_prompt + "\n\nAnswer only from the conversation above. "
                           "Do not use any tools. Return the structured result directly.").strip(),
        ))

    # === passthrough function tools ===
    if req.tools:
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
            if tool_name.startswith(TOOL_PREFIX):
                short = tool_name[len(TOOL_PREFIX):]
                capture.setdefault("calls", []).append({"name": short, "args": tool_input})
                logger.info("captured tool_call: %s args=%s", short, json.dumps(tool_input)[:400])
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
        capture.setdefault("blocked", []).append(tool_name)
        return PermissionResultDeny(behavior="deny", message="tool disabled", interrupt=True)

    return build(dict(
        system_prompt=system_prompt or None,
        max_turns=1 if config.SINGLE_TURN else config.MAX_TURNS,
        can_use_tool=deny_all,
    ))


async def run(req: ChatCompletionRequest) -> AsyncIterator[tuple[str, Any]]:
    """Yield ("text", str) / ("tool_calls", list) / ("done", None)."""
    capture: dict = {}
    options = _make_options(req, capture)
    _, user_prompt = build_prompt(req.messages)

    n_tools = len(req.tools or [])
    mode = "structured" if req.response_format and req.response_format.type != "text" else \
           ("tools" if n_tools else "chat")
    forced = req.tool_choice in ("required", "any") or isinstance(req.tool_choice, dict) \
        or mode == "structured"
    # tools/structured use TOOL_MAX_TURNS; single-turn applies to plain chat only
    eff_turns = config.TOOL_MAX_TURNS if mode != "chat" else (1 if config.SINGLE_TURN else config.MAX_TURNS)
    logger.info(
        "request: model=%s mode=%s messages=%d tools=%d forced=%s max_turns=%d prompt_chars=%d",
        req.model, mode, len(req.messages), n_tools, forced, eff_turns, len(user_prompt),
    )
    logger.debug("tool_choice=%s response_format=%s prompt=%r",
                 req.tool_choice,
                 req.response_format.type if req.response_format else None,
                 user_prompt[:800])

    from claude_agent_sdk import (
        AssistantMessage, TextBlock, ThinkingBlock, ToolUseBlock,
        SystemMessage, ResultMessage,
    )

    # For tool/structured requests, buffer text instead of streaming it: any text
    # before a captured tool call is just ToolSearch preamble and must be dropped.
    buffering = mode != "chat"
    buf: list[str] = []

    t0 = time.perf_counter()
    text_chars = 0
    emitted_text = False
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text:
                        text_chars += len(block.text)
                        if buffering:
                            buf.append(block.text)
                        else:
                            emitted_text = True
                            yield ("text", block.text)
                    elif isinstance(block, ThinkingBlock):
                        logger.debug("thinking: %d chars", len(getattr(block, "thinking", "") or ""))
                    elif isinstance(block, ToolUseBlock):
                        logger.info("assistant tool_use block: name=%s input=%s",
                                    getattr(block, "name", "?"),
                                    json.dumps(getattr(block, "input", {}))[:300])
            elif isinstance(msg, SystemMessage):
                logger.debug("system message: %s", getattr(msg, "subtype", ""))
            elif isinstance(msg, ResultMessage):
                capture["usage"] = _extract_usage(getattr(msg, "usage", None))
                capture["stop_reason"] = getattr(msg, "stop_reason", None)
                capture["structured_output"] = getattr(msg, "structured_output", None)
                denials = getattr(msg, "permission_denials", None) or []
                result_text = getattr(msg, "result", None)
                logger.info(
                    "result: turns=%s stop=%s error=%s denials=%d cost=$%s tokens=%s",
                    getattr(msg, "num_turns", "?"), capture["stop_reason"],
                    getattr(msg, "is_error", "?"), len(denials),
                    getattr(msg, "total_cost_usd", "?"), capture.get("usage"),
                )
                if capture["structured_output"] is not None:
                    logger.info("native structured_output present: %s",
                                json.dumps(capture["structured_output"])[:300])
                if result_text:
                    logger.debug("result text: %r", str(result_text)[:400])
                break

    calls = capture.get("calls") or []
    blocked = capture.get("blocked") or []
    structured = capture.get("structured_output")
    elapsed = (time.perf_counter() - t0) * 1000
    outcome = "structured" if mode == "structured" else ("tool_calls" if calls else "text")
    logger.info("done: outcome=%s text_chars=%d captured=%d blocked=%d elapsed=%.0fms",
                outcome, text_chars, len(calls), len(blocked), elapsed)

    usage = capture.get("usage")
    if usage:
        yield ("usage", usage)

    # structured output -> native JSON from the CLI, returned as message content
    if mode == "structured":
        if structured is not None:
            yield ("text", json.dumps(structured))
        else:
            logger.warning("structured request produced no structured_output (stop=%s); "
                           "falling back to buffered text", capture.get("stop_reason"))
            yield ("text", "".join(buf))
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

    # emit buffered text (tool/structured fallback) or the empty terminator
    if buffering:
        text = "".join(buf)
        yield ("text", text)
    elif not emitted_text:
        yield ("text", "")
    yield ("done", None)
