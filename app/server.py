"""FastAPI app exposing an OpenAI-compatible surface over the Claude subscription."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

import time as _time
import uuid as _uuid

from . import config
from .log import configure as _log_configure, logger, request_id
from .pool import pool
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


@contextlib.asynccontextmanager
async def _lifespan(app):
    sweeper = asyncio.create_task(pool.sweep_loop())
    try:
        yield
    finally:
        sweeper.cancel()
        await pool.shutdown()


app = FastAPI(title="Claude OpenAI-Compatible Wrapper", version="0.1.0",
              lifespan=_lifespan)
app.include_router(setup_router)


# ---- OpenAI-format errors (scoped to /v1; /setup keeps FastAPI defaults) ----

def _openai_error(status: int, message: str, err_type: str, code) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": err_type, "code": code}})


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException):
    if not request.url.path.startswith("/v1"):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    err_type = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
    return _openai_error(exc.status_code, str(exc.detail), err_type, exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_exc(request: Request, exc: RequestValidationError):
    if not request.url.path.startswith("/v1"):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []))
    return _openai_error(400, f"Invalid request: {loc}: {first.get('msg', 'validation error')}",
                         "invalid_request_error", 400)


@app.exception_handler(Exception)
async def _unhandled_exc(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    if not request.url.path.startswith("/v1"):
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    return _openai_error(500, str(exc), "api_error", "internal_error")


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


async def _blocking(req: ChatCompletionRequest):
    text_parts: list[str] = []
    tool_calls = None
    usage = Usage()
    try:
        async with asyncio.timeout(config.REQUEST_TIMEOUT):
            async for kind, payload in run(req):
                if kind == "text":
                    text_parts.append(payload)
                elif kind == "tool_calls":
                    tool_calls = payload
                elif kind == "usage":
                    usage = Usage(**payload)
    except TimeoutError:
        logger.warning("request timed out after %ss", config.REQUEST_TIMEOUT)
        return _openai_error(504, f"Request timed out after {config.REQUEST_TIMEOUT}s",
                             "timeout_error", "timeout")

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
        async with asyncio.timeout(config.REQUEST_TIMEOUT):
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
    except TimeoutError:
        logger.warning("stream timed out after %ss", config.REQUEST_TIMEOUT)
        yield "data: " + json.dumps({"error": {
            "message": f"Request timed out after {config.REQUEST_TIMEOUT}s",
            "type": "timeout_error", "code": "timeout"}}) + "\n\n"
    except Exception as e:  # surface backend errors inside the stream
        yield "data: " + json.dumps({"error": {
            "message": str(e), "type": "api_error", "code": "internal_error"}}) + "\n\n"

    yield chunk({}, finish=finish)
    # final usage-only chunk (OpenAI include_usage style)
    yield chunk({}, usage=usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    yield "data: [DONE]\n\n"
