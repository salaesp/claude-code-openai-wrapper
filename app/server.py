"""FastAPI app exposing an OpenAI-compatible surface over the Claude subscription."""
from __future__ import annotations

import json
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

import time as _time
import uuid as _uuid

from . import config
from .log import configure as _log_configure, logger, request_id
from .setup import router as setup_router
from .claude_backend import run
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

_log_configure()
app = FastAPI(title="Claude OpenAI-Compatible Wrapper", version="0.1.0")
app.include_router(setup_router)


@app.middleware("http")
async def log_requests(request, call_next):
    request_id.set(_uuid.uuid4().hex[:8])
    start = _time.perf_counter()
    response = await call_next(request)
    dur = (_time.perf_counter() - start) * 1000
    # streaming responses log at first byte; full stream duration shown in backend logs
    logger.info("%s %s -> %s %.0fms",
                request.method, request.url.path, response.status_code, dur)
    return response


@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/setup" if not config.is_configured() else "/health")


async def verify_key(authorization: str | None = Header(None),
                     x_api_key: str | None = Header(None)):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != config.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(verify_key)])
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": m.strip(), "object": "model", "created": now, "owned_by": "anthropic"}
            for m in config.EXPOSED_MODELS
        ],
    }


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


@app.post("/v1/chat/completions", dependencies=[Depends(verify_key)])
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return StreamingResponse(
            _stream(req), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await _blocking(req)


async def _blocking(req: ChatCompletionRequest) -> ChatCompletionResponse:
    text_parts: list[str] = []
    tool_calls = None
    usage = Usage()
    async for kind, payload in run(req):
        if kind == "text":
            text_parts.append(payload)
        elif kind == "tool_calls":
            tool_calls = payload
        elif kind == "usage":
            usage = Usage(**payload)

    if tool_calls:
        msg = Message(role="assistant", content=None, tool_calls=tool_calls)
        finish = "tool_calls"
    else:
        msg = Message(role="assistant", content="".join(text_parts))
        finish = "stop"

    return ChatCompletionResponse(
        id=_completion_id(),
        created=int(time.time()),
        model=req.model,
        choices=[Choice(index=0, message=msg, finish_reason=finish)],
        usage=usage,
    )


async def _stream(req: ChatCompletionRequest):
    cid = _completion_id()
    created = int(time.time())

    def chunk(delta: dict, finish=None, usage=None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n"

    # opening chunk establishes the assistant role
    yield chunk({"role": "assistant"})

    finish = "stop"
    usage = None
    try:
        async for kind, payload in run(req):
            if kind == "text" and payload:
                yield chunk({"content": payload})
            elif kind == "tool_calls":
                for i, tc in enumerate(payload):
                    yield chunk({
                        "tool_calls": [{
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": tc["function"],
                        }]
                    })
                finish = "tool_calls"
            elif kind == "usage":
                usage = payload
    except Exception as e:  # surface backend errors inside the stream
        yield chunk({"content": f"\n[wrapper error: {e}]"})

    yield chunk({}, finish=finish)
    # final usage-only chunk (OpenAI include_usage style)
    yield chunk({}, usage=usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    yield "data: [DONE]\n\n"
