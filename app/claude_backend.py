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
STRUCTURED_TOOL = "respond_with_structured_output"


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


def _make_options(req: ChatCompletionRequest, capture: dict) -> ClaudeAgentOptions:
    """Build SDK options: register tools + install the capture permission hook."""
    system_prompt, _ = build_prompt(req.messages)

    sdk_tools = []
    tool_names: list[str] = []

    # --- structured output mode: one forced schema tool ---
    if req.response_format and req.response_format.type in ("json_schema", "json_object"):
        schema = {"type": "object"}
        if req.response_format.type == "json_schema" and req.response_format.json_schema:
            schema = req.response_format.json_schema.get("schema", schema)

        @tool(STRUCTURED_TOOL, "Return the final answer as structured JSON.", schema)
        async def _structured(args):  # never actually runs; intercepted
            return {"content": [{"type": "text", "text": "ok"}]}

        sdk_tools.append(_structured)
        tool_names.append(TOOL_PREFIX + STRUCTURED_TOOL)
        system_prompt += (
            f"\n\nYou MUST answer by calling the `{STRUCTURED_TOOL}` tool exactly once "
            "with arguments matching its schema. Do not write a normal text reply."
        )

    # --- passthrough function tools ---
    for t in req.tools or []:
        name = t.function.name
        schema = t.function.parameters or {"type": "object"}

        @tool(name, t.function.description or name, schema)
        async def _passthrough(args):  # intercepted before execution
            return {"content": [{"type": "text", "text": "ok"}]}

        sdk_tools.append(_passthrough)
        tool_names.append(TOOL_PREFIX + name)

    forced = req.tool_choice in ("required", "any") or (
        isinstance(req.tool_choice, dict)
    )
    if forced and req.tools:
        system_prompt += "\n\nYou MUST call one of the provided tools to respond."

    server = create_sdk_mcp_server(name=MCP_SERVER, tools=sdk_tools) if sdk_tools else None
    logger.debug("registered tools: %s", tool_names or "(none)")

    async def can_use_tool(tool_name: str, tool_input: dict, ctx) -> Any:
        if tool_name.startswith(TOOL_PREFIX):
            short = tool_name[len(TOOL_PREFIX):]
            capture.setdefault("calls", []).append({"name": short, "args": tool_input})
            logger.info("captured tool_call: %s args=%s", short, json.dumps(tool_input)[:400])
            # stop the agent immediately; we hand control back to the OpenAI client
            return PermissionResultDeny(
                behavior="deny", message="captured by wrapper", interrupt=True
            )
        # a tool we did not register — Claude tried to use its own agent tooling
        capture.setdefault("blocked", []).append(tool_name)
        logger.warning("blocked non-passthrough tool: %s args=%s", tool_name, json.dumps(tool_input)[:200])
        return PermissionResultDeny(
            behavior="deny", message="tool disabled", interrupt=True
        )

    # Tools/structured need extra turns: this CLI loads MCP tools via a ToolSearch
    # round-trip first, so a hard max_turns=1 would cut off before the real call.
    needs_tools = bool(sdk_tools)
    max_turns = config.MAX_TURNS if needs_tools else (1 if config.SINGLE_TURN else config.MAX_TURNS)

    opts: dict[str, Any] = dict(
        system_prompt=system_prompt or None,
        # single-turn only applies to plain chat; don't load CLAUDE.md/user settings
        max_turns=max_turns,
        setting_sources=[],
        can_use_tool=can_use_tool,
        permission_mode="default",
        disallowed_tools=["Bash", "Read", "Edit", "Write", "WebFetch", "WebSearch"],
    )
    if config.DISABLE_THINKING:
        opts["thinking"] = {"type": "disabled"}  # SDK expects a dict, not the TypedDict ctor
    if server:
        opts["mcp_servers"] = {MCP_SERVER: server}
    if req.model:
        opts["model"] = req.model
    # keep falsy-but-meaningful values (e.g. setting_sources=[]) — only drop None
    return ClaudeAgentOptions(**{k: v for k, v in opts.items() if v is not None})


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
    # tools/structured always get full turns (ToolSearch round-trip); single-turn is chat-only
    eff_turns = config.MAX_TURNS if mode != "chat" else (1 if config.SINGLE_TURN else config.MAX_TURNS)
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
    elapsed = (time.perf_counter() - t0) * 1000
    outcome = "tool_calls" if (calls and calls[0]["name"] != STRUCTURED_TOOL) else "text"
    logger.info("done: outcome=%s text_chars=%d captured=%d blocked=%d elapsed=%.0fms",
                outcome, text_chars, len(calls), len(blocked), elapsed)

    usage = capture.get("usage")
    if usage:
        yield ("usage", usage)

    # structured output -> return JSON as content, not as a tool_call
    if calls and calls[0]["name"] == STRUCTURED_TOOL:
        yield ("text", json.dumps(calls[0]["args"]))
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
