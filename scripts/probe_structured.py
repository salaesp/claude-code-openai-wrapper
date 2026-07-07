#!/usr/bin/env python3
"""Probe: does native --json-schema structured output actually converge?

Isolates the structured path from the FastAPI stack. Drives ClaudeSDKClient
directly with the SAME options the wrapper builds (setting_sources=[], no tools,
output_format json_schema, allowed_tools=[]) across schemas of growing complexity.

For each case it prints: turns, stop_reason, is_error, elapsed, and — the thing
that matters — whether ResultMessage.structured_output populated. It also dumps
each StructuredOutput tool_use envelope the model emits, so you can see the
`{"parameter": {...}}` wrapping (or whatever the CLI is rejecting).

Read the verdict at the end:
  - trivial schema populates but complex does NOT  -> native path chokes on schema
    complexity/nesting. Simplify the schema, or switch complex schemas to the
    tool-capture fallback (register schema as one forced function tool).
  - NOTHING populates (even trivial)              -> native --json-schema is broken
    for this CLI/subscription. Use the tool-capture fallback for ALL structured.
  - everything populates                          -> native path is fine; the prod
    hang is purely the turn-budget-vs-timeout mismatch (fixes A/B already cover it).

Usage:
  export CLAUDE_CODE_OAUTH_TOKEN=...        # your subscription token
  python scripts/probe_structured.py [--model claude-haiku-4-5] [--turns 3]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

os.environ.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")

from claude_agent_sdk import (  # noqa: E402
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    ToolUseBlock,
    TextBlock,
)

DIRECTIVE = (
    "\n\nReturn your answer by producing a single JSON object that strictly matches "
    "the required schema — correct field names, types, enums, and all required "
    "fields. The object's top-level keys MUST be the schema's own properties. Do NOT "
    "wrap the result in any envelope key such as \"parameter\", \"input\", "
    "\"arguments\", \"value\", or \"result\" — emit the schema object directly. Do "
    "not write any explanation, preamble, or commentary; emit only the structured "
    "result on your first attempt."
)

# --- schemas, ascending complexity -----------------------------------------

TRIVIAL = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

FLAT_LIST = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "string"}},
        "count": {"type": "integer"},
    },
    "required": ["items", "count"],
    "additionalProperties": False,
}

# Mirrors the shape that hangs in prod: $defs + nested object array.
NESTED_DEFS = {
    "$defs": {
        "Scenario": {
            "type": "object",
            "properties": {
                "entry_point_class_fqn": {"type": "string"},
                "entry_point_method": {"type": "string"},
                "kind": {"type": "string", "enum": ["rest_endpoint", "service", "other"]},
                "http_method": {"type": "string"},
                "http_path": {"type": "string"},
                "target_class_fqn": {"type": "string"},
                "target_method": {"type": "string"},
                "why": {"type": "string"},
            },
            "required": ["entry_point_class_fqn", "entry_point_method", "kind",
                         "target_class_fqn", "target_method", "why"],
            "additionalProperties": False,
        }
    },
    "type": "object",
    "properties": {"scenarios": {"type": "array", "items": {"$ref": "#/$defs/Scenario"}}},
    "required": ["scenarios"],
    "additionalProperties": False,
}

CASES = [
    ("trivial", TRIVIAL,
     "What is the capital of France? Answer in one word."),
    ("flat_list", FLAT_LIST,
     "List three primary colors."),
    ("nested_defs", NESTED_DEFS,
     "COMPONENT: org.example.controller.CategoryController\n"
     "OBLIGATIONS: findAllCategories (GET /api/categories) happy path; "
     "createCategory (POST /api/categories) happy path. "
     "Emit one scenario per obligation."),
]


async def run_case(name: str, schema: dict, prompt: str, model: str, turns: int) -> dict:
    opts = ClaudeAgentOptions(
        setting_sources=[],
        permission_mode="default",
        tools=[],
        strict_mcp_config=True,
        env={"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
        model=model,
        output_format={"type": "json_schema", "schema": schema},
        allowed_tools=[],
        max_turns=turns,
        system_prompt=(
            "You are a helpful AI assistant accessed through a plain text API." + DIRECTIVE
        ),
    )
    client = ClaudeSDKClient(options=opts)
    envelopes: list[str] = []
    populated = None
    num_turns = stop = is_error = None
    t0 = time.perf_counter()
    try:
        await client.connect()
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, ToolUseBlock):
                        envelopes.append(json.dumps(getattr(b, "input", {}))[:200])
                    elif isinstance(b, TextBlock) and b.text.strip():
                        envelopes.append("TEXT: " + b.text.strip()[:120])
            elif isinstance(msg, ResultMessage):
                populated = getattr(msg, "structured_output", None)
                num_turns = getattr(msg, "num_turns", None)
                stop = getattr(msg, "stop_reason", None)
                is_error = getattr(msg, "is_error", None)
                break
    except Exception as e:
        stop = f"EXC:{type(e).__name__}:{e}"
    finally:
        try:
            async with asyncio.timeout(10):
                await client.disconnect()
        except Exception:
            pass
    elapsed = (time.perf_counter() - t0) * 1000
    ok = populated is not None
    print(f"\n=== {name} ===")
    print(f"  turns={num_turns} stop={stop} is_error={is_error} elapsed={elapsed:.0f}ms")
    print(f"  structured_output populated: {'YES' if ok else 'NO'}")
    if ok:
        print(f"    -> {json.dumps(populated)[:200]}")
    for i, env in enumerate(envelopes[:6]):
        print(f"  emit[{i}]: {env}")
    return {"name": name, "ok": ok, "turns": num_turns, "elapsed_ms": round(elapsed)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--turns", type=int, default=3)
    args = ap.parse_args()

    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        print("ERROR: set CLAUDE_CODE_OAUTH_TOKEN first.", file=sys.stderr)
        return 2

    print(f"model={args.model} max_turns={args.turns}")
    results = []
    for name, schema, prompt in CASES:
        results.append(await run_case(name, schema, prompt, args.model, args.turns))

    print("\n===== VERDICT =====")
    trivial_ok = next((r["ok"] for r in results if r["name"] == "trivial"), False)
    complex_ok = next((r["ok"] for r in results if r["name"] == "nested_defs"), False)
    if not trivial_ok:
        print("Native --json-schema is BROKEN even for a trivial schema on this CLI/"
              "subscription. Use the tool-capture fallback for ALL structured requests.")
    elif trivial_ok and not complex_ok:
        print("Native path works for simple schemas but CHOKES on $defs/nested. Either "
              "simplify the schema or route complex schemas through the tool-capture "
              "fallback (register the schema as one forced function tool).")
    else:
        print("Native path converges for all cases. The prod hang is purely the "
              "turn-budget-vs-REQUEST_TIMEOUT mismatch — fixes A/B cover it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
