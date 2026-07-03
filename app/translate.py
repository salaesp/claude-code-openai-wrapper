"""Translate between OpenAI wire format and a flat Claude prompt.

The endpoint is STATELESS: the OpenAI client resends the full message history on
every call. We reconstruct that history into (system_prompt, user_prompt) text.
Prior assistant tool_calls and their tool results are rendered as plain context so
Claude sees what already happened without us needing a live Claude session.
"""
from __future__ import annotations

import json
from .models import Message


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI multimodal array -> keep text parts only (images not supported here)
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def build_prompt(messages: list[Message]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    system_chunks: list[str] = []
    convo: list[str] = []

    for m in messages:
        if m.role == "system":
            system_chunks.append(_content_to_text(m.content))

        elif m.role == "user":
            convo.append(f"User: {_content_to_text(m.content)}")

        elif m.role == "assistant":
            text = _content_to_text(m.content)
            if text:
                convo.append(f"Assistant: {text}")
            for tc in m.tool_calls or []:
                convo.append(
                    f"Assistant (called tool `{tc.function.name}` "
                    f"with arguments {tc.function.arguments})"
                )

        elif m.role == "tool":
            # result of a previously requested tool call, as context
            convo.append(
                f"Tool result for `{m.name or m.tool_call_id}`: "
                f"{_content_to_text(m.content)}"
            )

    system_prompt = "\n\n".join(c for c in system_chunks if c).strip()

    if convo and convo[-1].startswith("User:"):
        # normal case: last turn is the user asking
        user_prompt = "\n".join(convo)
    else:
        # last turn was a tool result -> instruct Claude to continue
        convo.append("Assistant:")
        user_prompt = (
            "Continue the conversation below. Use the tool results already "
            "provided; call another tool only if you still need one.\n\n"
            + "\n".join(convo)
        )
    return system_prompt, user_prompt.strip()


def openai_tool_to_sdk(tool) -> tuple[str, str, dict]:
    """(sanitized_name, description, json_schema) for one OpenAI tool."""
    fn = tool.function
    return fn.name, (fn.description or fn.name), (fn.parameters or {"type": "object"})
