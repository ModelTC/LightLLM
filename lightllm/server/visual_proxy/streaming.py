"""Public OpenAI SSE response for the visual agent."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import suppress
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import Response

from ..api_models import PromptTokensDetails, UsageInfo
from ..api_stream_obj import CustomStreamingResponse
from ..visual_trace import VisualTraceRecorder
from .agent import MainChatHandler, VisualHandler, _run_agent_choice, _usage_add
from .config import VisualChatProxyError, VisualProxyTimeoutError
from .evidence import ImageRegistry
from .runtime import VisualProxyRuntime


def _true_stream_response(
    *,
    request_payload: dict[str, Any],
    registry: ImageRegistry,
    raw_request: Request,
    runtime: VisualProxyRuntime,
    main_chat_handler: MainChatHandler,
    visual_handler: VisualHandler,
    max_agent_steps: int,
    trace_id: str,
    trace_recorder: VisualTraceRecorder,
    requested_n: int,
    public_reasoning_enabled: bool,
    deadline: float,
) -> CustomStreamingResponse:
    """Run lazily with buffered safe output, open-comment, and heartbeat SSE."""

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = str(request_payload.get("model") or "default")
    separate_reasoning = request_payload.get("separate_reasoning") is not False
    stream_options = request_payload.get("stream_options") or {}
    include_usage = (
        isinstance(stream_options, dict) and stream_options.get("include_usage") is True
    )

    def encode_choice(
        index: int, delta: dict[str, Any], finish_reason: Optional[str] = None
    ) -> str:
        return (
            "data: "
            + json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )

    async def chunks():
        aggregate_usage = UsageInfo(prompt_tokens_details=PromptTokensDetails())
        for index in range(requested_n):
            queue: asyncio.Queue[tuple[str, str, bool]] = asyncio.Queue()

            async def callback(kind: str, value: str, separate: bool) -> None:
                if value:
                    await queue.put((kind, value, separate))

            async def run_choice_with_timeout():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise VisualProxyTimeoutError("Visual agent loop timed out")
                try:
                    return await asyncio.wait_for(
                        _run_agent_choice(
                            request_payload=request_payload,
                            registry=registry,
                            raw_request=raw_request,
                            runtime=runtime,
                            main_chat_handler=main_chat_handler,
                            visual_handler=visual_handler,
                            max_agent_steps=max_agent_steps,
                            trace_id=f"{trace_id}-{index}",
                            trace_recorder=trace_recorder,
                            public_reasoning_enabled=public_reasoning_enabled,
                            stream_callback=callback,
                        ),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError as exc:
                    raise VisualProxyTimeoutError(
                        "Visual agent loop timed out"
                    ) from exc

            task = asyncio.create_task(run_choice_with_timeout())
            try:
                await asyncio.sleep(0)
                if index == 0:
                    yield ": nova-stream-open\n\n"
                role_emitted = False
                reasoning_started = False
                content_reasoning_started = False
                streamed_content = ""
                last_heartbeat = asyncio.get_running_loop().time()
                while not task.done() or not queue.empty():
                    try:
                        kind, value, separate = await asyncio.wait_for(
                            queue.get(), timeout=0.25
                        )
                    except asyncio.TimeoutError:
                        now = asyncio.get_running_loop().time()
                        if now - last_heartbeat >= 5.0:
                            yield ": heartbeat\n\n"
                            last_heartbeat = now
                        continue
                    if not role_emitted:
                        yield encode_choice(index, {"role": "assistant", "content": ""})
                        role_emitted = True
                    if kind == "ready":
                        continue
                    if kind in {"reasoning", "trace"}:
                        field = (
                            "reasoning"
                            if separate_reasoning or kind == "trace"
                            else "content"
                        )
                        if field == "reasoning":
                            if reasoning_started:
                                value = "\n" + value
                            reasoning_started = True
                        else:
                            if content_reasoning_started:
                                value = "\n" + value
                            content_reasoning_started = True
                        if field == "content":
                            streamed_content += value
                        yield encode_choice(index, {field: value})
                    elif kind == "content":
                        streamed_content += value
                        yield encode_choice(index, {"content": value})
                result = await task
                if not role_emitted:
                    yield encode_choice(index, {"role": "assistant", "content": ""})
            finally:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            if isinstance(result, Response):
                raise VisualChatProxyError(
                    f"Main model returned HTTP response status {result.status_code} during visual streaming"
                )
            choice, usage = result
            _usage_add(aggregate_usage, usage)
            public_message = choice.message
            remaining_content = public_message.content or ""
            if streamed_content and remaining_content.startswith(streamed_content):
                remaining_content = remaining_content[len(streamed_content) :]
            if remaining_content:
                yield encode_choice(index, {"content": remaining_content})
            for call_index, call in enumerate(public_message.tool_calls or []):
                call_data = call.model_dump(exclude_none=True)
                call_data["index"] = (
                    call.index if call.index is not None else call_index
                )
                yield encode_choice(index, {"tool_calls": [call_data]})
            yield encode_choice(index, {}, choice.finish_reason or "stop")

        if include_usage:
            yield "data: " + json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": aggregate_usage.model_dump(exclude_none=True),
                },
                ensure_ascii=False,
            ) + "\n\n"
        yield "data: [DONE]\n\n"

    return CustomStreamingResponse(chunks(), media_type="text/event-stream")
