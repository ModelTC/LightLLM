"""Opt-in OpenAI chat proxy that gives a text agent a builtin vision reader.

The local LightLLM model remains the agent model.  Image payloads are replaced
with ``<image_n/>`` text tags before the request reaches that model.  If the
model calls the injected ``vision_reader`` function, this module sends only the
selected image and task to a remote OpenAI-compatible multimodal service, adds
the result as an internal tool response, and continues the local agent loop.

This module deliberately sits above ``chat_completions_impl`` so the native
generation, routing, tokenization, and model code do not need proxy-specific
branches.  The public route calls it only when ``--visual_remote_url`` is set
and the request actually contains image content.
"""

from __future__ import annotations

import asyncio
import base64 as base64
import json
import socket as socket
import time
import traceback
import uuid
from contextlib import suppress
from typing import Any, Union

import httpx as httpx
from fastapi import Request
from fastapi.responses import Response

from lightllm.utils.error_utils import ClientDisconnected
from lightllm.utils.log_utils import init_logger

from .api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    PromptTokensDetails,
    UsageInfo,
)
from .api_stream_obj import CustomStreamingResponse
from .visual_proxy.agent import (
    MainChatHandler,
    _response_for_trace,
    _run_agent_choice,
    _usage_add,
)
from .visual_proxy.constants import (
    ANTHROPIC_SEQUENTIAL_TOOL_PROMPT,
    BUILTIN_VISION_READER_TOOL,
    MAX_AGENT_STEPS,
    MAX_REASONING_CONTEXT_BYTES,
    VISION_READER_NAME,
)
from .visual_proxy.config import (
    VisualChatProxyError,
    VisualProxyCapacityError,
    VisualProxySettings,
    VisualProxyTimeoutError,
    VisualProxyUpstreamError,
    VisualProxyUpstreamRejectedError,
    validate_visual_proxy_startup,
)
from .visual_proxy.evidence import (
    EvidenceLedger,
    ImageRegistry,
    RegisteredImage,
)
from .visual_proxy.inputs import (
    apply_visual_thinking_policy,
    replace_images_with_tags,
    request_has_images,
    requested_enable_thinking,
    resolve_public_reasoning_enabled,
    should_use_visual_proxy,
)
from .visual_proxy.model_stream import _consume_main_model_stream
from .visual_proxy.images import _pinned_remote_image_request
from .visual_proxy.sanitizer import (
    _merge_reasoning_text,
    reasoning_contains_raw_fc_serialization,
    sanitize_reasoning,
    strip_backend_special_tokens,
    text_exposes_private_multimodal_mechanism,
)
from .visual_proxy.traces import (
    _format_natural_builtin_trace,
    _format_public_builtin_projection,
    _format_xml_builtin_trace,
    expand_builtin_traces,
)
from .visual_proxy.provider import (
    VisualRemoteResult,
    _remote_content,
    call_visual_remote,
)
from .visual_proxy.runtime import (
    VisualProxyRuntime,
)
from .visual_proxy.streaming import _true_stream_response
from .visual_trace import (
    VisualTraceRecorder,
)


logger = init_logger(__name__)

# Stable facade for API routes and existing imports. Implementation details
# live in ``visual_proxy`` modules grouped by responsibility.
__all__ = (
    "BUILTIN_VISION_READER_TOOL",
    "ANTHROPIC_SEQUENTIAL_TOOL_PROMPT",
    "ClientDisconnected",
    "EvidenceLedger",
    "ImageRegistry",
    "RegisteredImage",
    "VisualChatProxyError",
    "VisualProxyCapacityError",
    "VisualProxyRuntime",
    "VisualProxySettings",
    "VisualProxyTimeoutError",
    "VisualProxyUpstreamError",
    "VisualProxyUpstreamRejectedError",
    "VisualRemoteResult",
    "VISION_READER_NAME",
    "_consume_main_model_stream",
    "_format_natural_builtin_trace",
    "_format_public_builtin_projection",
    "_format_xml_builtin_trace",
    "_message_with_reasoning",
    "_pinned_remote_image_request",
    "_remote_content",
    "_sanitize_model_message",
    "apply_visual_thinking_policy",
    "base64",
    "call_visual_remote",
    "expand_builtin_traces",
    "request_has_images",
    "resolve_public_reasoning_enabled",
    "sanitize_reasoning",
    "httpx",
    "socket",
    "should_use_visual_proxy",
    "text_exposes_private_multimodal_mechanism",
    "validate_visual_proxy_startup",
    "visual_chat_completions_impl",
)


def _sanitize_model_message(message: ChatMessage) -> ChatMessage:
    """Normalize one fresh model turn before it reaches history or the caller."""

    data = message.model_dump(exclude_none=True)
    raw_reasoning = _merge_reasoning_text(
        data.pop("reasoning", None), data.pop("reasoning_content", None)
    )
    clean_reasoning = sanitize_reasoning(raw_reasoning)
    if clean_reasoning:
        data["reasoning"] = clean_reasoning
    content = data.get("content")
    if isinstance(content, str):
        if reasoning_contains_raw_fc_serialization(content):
            raise VisualChatProxyError(
                "Main model content contained raw function-calling syntax; "
                "response quarantined."
            )
        data["content"] = strip_backend_special_tokens(content)
    return ChatMessage.model_validate(data)


def _message_with_reasoning(
    message: ChatMessage,
    reasoning_context: list[str],
) -> ChatMessage:
    data = message.model_dump(exclude_none=True)
    model_reasoning = sanitize_reasoning(
        _merge_reasoning_text(
            data.get("reasoning"), data.pop("reasoning_content", None)
        )
    )
    reasoning = _merge_reasoning_text(*reasoning_context, model_reasoning)
    if len(reasoning.encode("utf-8")) > MAX_REASONING_CONTEXT_BYTES:
        raise VisualChatProxyError(
            "Visual reasoning context exceeds the configured size limit"
        )
    if reasoning:
        data["reasoning"] = reasoning
    else:
        data.pop("reasoning", None)
    return ChatMessage.model_validate(data)


def _request_dict(request: ChatCompletionRequest) -> dict[str, Any]:
    return request.model_dump(by_alias=True, exclude_none=True)


async def _visual_chat_completions_impl(
    *,
    request: ChatCompletionRequest,
    raw_request: Request,
    runtime: VisualProxyRuntime,
    main_chat_handler: MainChatHandler,
    trace_id: str,
    trace_recorder: VisualTraceRecorder,
) -> Union[ChatCompletionResponse, Response]:
    requested_thinking = requested_enable_thinking(request)
    public_reasoning_enabled = resolve_public_reasoning_enabled(
        requested_thinking, runtime.settings
    )
    if request.seed is not None and not 0 <= request.seed <= 9_999_998:
        raise ValueError("seed must be an integer between 0 and 9999998")
    request = apply_visual_thinking_policy(request, runtime.settings)
    request_payload = _request_dict(request)
    deadline = asyncio.get_running_loop().time() + runtime.settings.agent_timeout
    registry = ImageRegistry(
        max_images=runtime.settings.max_images,
        max_total_image_bytes=runtime.settings.max_total_image_bytes,
    )
    request_payload["messages"] = replace_images_with_tags(
        request_payload["messages"], registry, runtime.settings
    )
    request_payload["messages"] = expand_builtin_traces(
        request_payload["messages"],
        runtime.settings.builtin_trace_format,
    )
    requested_n = max(1, int(request.n or 1))
    if requested_n > runtime.settings.max_choices:
        raise ValueError(
            f"Visual proxy accepts at most {runtime.settings.max_choices} choices per request"
        )
    if trace_recorder.enabled:
        trace_recorder.event(
            "proxy_agent_request_registered",
            request=request_payload,
            registered_images=[
                {"tag": tag, "origin": registry.resolve(tag).origin}
                for tag in registry.tags()
            ],
        )
    logger.info(
        "[visual-chat-proxy][external_request_registered] trace_id=%s image_count=%d tags=%s choices=%d",
        trace_id,
        len(registry),
        registry.available_tags(),
        requested_n,
    )

    if request.stream:
        return _true_stream_response(
            request_payload=request_payload,
            registry=registry,
            raw_request=raw_request,
            runtime=runtime,
            main_chat_handler=main_chat_handler,
            visual_handler=call_visual_remote,
            max_agent_steps=MAX_AGENT_STEPS,
            trace_id=trace_id,
            trace_recorder=trace_recorder,
            requested_n=requested_n,
            public_reasoning_enabled=public_reasoning_enabled,
            deadline=deadline,
        )

    async def run_choices() -> Union[
        tuple[list[ChatCompletionResponseChoice], UsageInfo], Response
    ]:
        choices: list[ChatCompletionResponseChoice] = []
        aggregate_usage = UsageInfo(prompt_tokens_details=PromptTokensDetails())
        for index in range(requested_n):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise VisualProxyTimeoutError("Visual agent loop timed out")
            try:
                result = await asyncio.wait_for(
                    _run_agent_choice(
                        request_payload=request_payload,
                        registry=registry,
                        raw_request=raw_request,
                        runtime=runtime,
                        main_chat_handler=main_chat_handler,
                        visual_handler=call_visual_remote,
                        max_agent_steps=MAX_AGENT_STEPS,
                        trace_id=f"{trace_id}-{index}",
                        trace_recorder=trace_recorder,
                        public_reasoning_enabled=public_reasoning_enabled,
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as exc:
                raise VisualProxyTimeoutError("Visual agent loop timed out") from exc
            if isinstance(result, Response):
                return result
            choice, usage = result
            choices.append(choice.model_copy(update={"index": index}))
            _usage_add(aggregate_usage, usage)
        return choices, aggregate_usage

    choices_result = await run_choices()
    if isinstance(choices_result, Response):
        return choices_result
    choices, aggregate_usage = choices_result

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=aggregate_usage,
    )
    if trace_recorder.enabled:
        trace_recorder.event(
            "proxy_final_response",
            response=response.model_dump(mode="json", exclude_none=True),
        )
    return response


async def visual_chat_completions_impl(
    *,
    request: ChatCompletionRequest,
    raw_request: Request,
    runtime: VisualProxyRuntime,
    main_chat_handler: MainChatHandler,
) -> Union[ChatCompletionResponse, Response]:
    """Run one OpenAI request through the opt-in visual agent adapter."""

    if runtime is None:
        raise VisualChatProxyError("Visual proxy runtime is not initialized")
    trace_id = f"visual-{uuid.uuid4().hex[:16]}"
    trace_request: dict[str, Any] = {}
    if runtime.settings.trace_dump_dir is not None:
        try:
            raw_payload = await raw_request.json()
        except Exception:
            raw_payload = None
        trace_request = {
            "method": raw_request.method,
            "path": raw_request.url.path,
            "payload": raw_payload,
            "normalized_openai_payload": request.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        }
    recorder = VisualTraceRecorder(
        trace_id,
        runtime.settings.trace_dump_dir,
        trace_request,
    )
    request_slot = runtime.request_slot(raw_request)
    await request_slot.__aenter__()
    deferred_stream = False
    try:
        response = await _visual_chat_completions_impl(
            request=request,
            raw_request=raw_request,
            runtime=runtime,
            main_chat_handler=main_chat_handler,
            trace_id=trace_id,
            trace_recorder=recorder,
        )
        if isinstance(response, CustomStreamingResponse):
            original_iterator = response.body_iterator

            async def traced_stream():
                try:
                    async for chunk in original_iterator:
                        yield chunk
                except ClientDisconnected as exc:
                    if recorder.enabled:
                        recorder.finish_error(exc, traceback.format_exc())
                    return
                except (GeneratorExit, asyncio.CancelledError) as exc:
                    if recorder.enabled:
                        recorder.finish_error(exc, traceback.format_exc())
                    return
                except BaseException as exc:
                    if recorder.enabled:
                        recorder.finish_error(exc, traceback.format_exc())
                    logger.exception("Visual proxy stream failed after response start")
                    error_data = json.dumps(
                        {
                            "error": {
                                "message": str(exc),
                                "type": "server_error",
                                "code": "visual_proxy_stream_error",
                            }
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {error_data}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                else:
                    if recorder.enabled:
                        recorder.finish_success({"stream": True, "completed": True})
                finally:
                    close = getattr(original_iterator, "aclose", None)
                    if close is not None:
                        with suppress(RuntimeError):
                            await close()
                    await recorder.flush()
                    await request_slot.__aexit__(None, None, None)

            response.body_iterator = traced_stream()
            deferred_stream = True
            return response
        if recorder.enabled:
            recorder.finish_success(_response_for_trace(response))
        return response
    except BaseException as exc:
        if recorder.enabled:
            recorder.finish_error(exc, traceback.format_exc())
        raise
    finally:
        if not deferred_stream:
            await recorder.flush()
            await request_slot.__aexit__(None, None, None)
