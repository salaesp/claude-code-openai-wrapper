"""OpenAI-compatible request/response schemas (subset we support)."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# ---- request ----

class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string, per OpenAI spec


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None      # assistant -> requested calls
    tool_call_id: str | None = None               # tool -> which call this answers


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None     # {name, schema, strict}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None   # "auto"|"none"|"required"|{...}
    response_format: ResponseFormat | None = None
    stream: bool = False
    temperature: float | None = None                  # accepted, forwarded where possible
    max_tokens: int | None = None
    reasoning_effort: str | None = None               # minimal|low|medium|high (+xhigh) -> CLI --effort
    # unsupported OpenAI fields are ignored, not rejected


# ---- response ----

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
