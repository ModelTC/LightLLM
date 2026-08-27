"""Visual agent loop and internal main-model response handling."""

from __future__ import annotations

import copy
import json
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Union

from fastapi import Request
from fastapi.responses import Response

from lightllm.utils.error_utils import ClientDisconnected
from lightllm.utils.log_utils import init_logger

from ..api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    PromptTokensDetails,
    UsageInfo,
)
from ..api_stream_obj import CustomStreamingResponse
from ..visual_trace import VisualTraceRecorder, bind_main_model_trace
from .config import (
    VisualChatProxyError,
    VisualProxyCapacityError,
    VisualProxyTimeoutError,
    VisualProxyUpstreamError,
    VisualProxyUpstreamRejectedError,
)
from .constants import (
    BUILTIN_VISION_READER_TOOL,
    MAX_REASONING_CONTEXT_BYTES,
    TRUNCATED_FINISH_REASONS,
    VISION_READER_NAME,
)
from .evidence import EvidenceLedger, ImageRegistry, RegisteredImage
from .inputs import request_has_images
from .model_stream import _consume_main_model_stream
from .sanitizer import StreamEventCallback
from .traces import _format_public_builtin_projection
from .runtime import (
    VisualProxyRuntime,
    _request_is_disconnected,
)


logger = init_logger(__name__)
MainChatHandler = Callable[
    [ChatCompletionRequest, Request], Awaitable[Union[ChatCompletionResponse, Response]]
]
VisualHandler = Callable[..., Awaitable[Any]]


def _builtin_vision_tool_error(
    code: str,
    message: str,
    *,
    retryable: bool,
) -> str:
    """Return a private, model-visible error for one builtin Vision call."""

    return json.dumps(
        {
            "ok": False,
            "error": {
                "type": code,
                "message": str(message),
                "retryable": retryable,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _vision_tool_error_from_exception(
    exc: VisualChatProxyError,
) -> tuple[str, bool]:
    message = str(exc)
    lowered = message.casefold()
    if isinstance(exc, VisualProxyTimeoutError):
        return "vision_timeout", True
    if isinstance(exc, VisualProxyCapacityError):
        return "vision_capacity_exceeded", True
    if isinstance(exc, VisualProxyUpstreamRejectedError):
        return "vision_upstream_rejected", False
    if "empty assistant message" in lowered or "empty" in lowered:
        return "vision_empty_output", True
    if any(
        marker in lowered
        for marker in (
            "invalid json",
            "non-object response",
            "no choices",
            "no choices[0].message",
            "stream failed",
            "non-streaming response",
        )
    ):
        return "vision_protocol_error", True
    return "vision_upstream_error", True


def _tool_arguments(tool_call: Any) -> dict[str, Any]:
    arguments = tool_call.function.arguments
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise ValueError("vision_reader arguments must be a JSON object")
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"vision_reader arguments are not valid JSON: {arguments!r}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("vision_reader arguments must decode to an object")
    return value


def _usage_add(total: UsageInfo, current: UsageInfo) -> None:
    total.prompt_tokens += current.prompt_tokens
    total.completion_tokens = (total.completion_tokens or 0) + (
        current.completion_tokens or 0
    )
    total.total_tokens += current.total_tokens
    current_cached = 0
    if current.prompt_tokens_details is not None:
        current_cached = current.prompt_tokens_details.cached_tokens
    if total.prompt_tokens_details is None:
        total.prompt_tokens_details = PromptTokensDetails()
    total.prompt_tokens_details.cached_tokens += current_cached


def _usage_add_mapping(total: UsageInfo, current: Any) -> None:
    if not isinstance(current, dict):
        return
    prompt_tokens = current.get("prompt_tokens", current.get("input_tokens", 0))
    completion_tokens = current.get(
        "completion_tokens", current.get("output_tokens", 0)
    )
    try:
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
    except (TypeError, ValueError):
        return
    _usage_add(
        total,
        UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(),
        ),
    )


def _has_external_vision_reader(tools: list[dict[str, Any]]) -> bool:
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == VISION_READER_NAME:
            return True
    return False


@dataclass(frozen=True)
class _BuiltinResult:
    call: Any
    call_id: str
    image_tag: str
    task: str
    content: str
    accepted: bool


async def _execute_builtin_call(
    *,
    tool_call: Any,
    call_index: int,
    step: int,
    model: str,
    registry: ImageRegistry,
    ledger: EvidenceLedger,
    aggregate_usage: UsageInfo,
    runtime: VisualProxyRuntime,
    raw_request: Request,
    trace_id: str,
    trace_recorder: VisualTraceRecorder,
    visual_handler: VisualHandler,
) -> _BuiltinResult:
    image_tag = ""
    task = ""
    call_id = tool_call.id or f"call_{uuid.uuid4().hex[:24]}"
    accepted = False
    image: Optional[RegisteredImage] = None

    try:
        arguments = _tool_arguments(tool_call)
        image_tag = str(arguments.get("image") or "")
        task = str(arguments.get("task") or "").strip()
        if not image_tag or not task:
            raise ValueError(
                "Builtin vision_reader requires both arguments: image and task. "
                f"Available tags: {registry.available_tags()}."
            )
        try:
            image = registry.resolve(image_tag)
        except ValueError as exc:
            raise ValueError(
                f"Builtin vision_reader cannot read image={image_tag!r}. "
                f"Available tags: {registry.available_tags()}."
            ) from exc
        image_tag = image.tag

        visual_result = await visual_handler(
            runtime=runtime,
            model=model,
            image=image,
            task=task,
            trace_id=f"{trace_id}-s{step}-c{call_index}",
            raw_request=raw_request,
            trace_recorder=trace_recorder,
            result_callback=None,
        )
        raw_result = str(visual_result)
        finish_reason = str(getattr(visual_result, "finish_reason", "") or "")
        image_digest = str(getattr(visual_result, "image_digest_sha256", "") or "")
        _usage_add_mapping(aggregate_usage, getattr(visual_result, "usage", {}))

        if finish_reason in TRUNCATED_FINISH_REASONS:
            ledger.record(
                image_tag=image_tag,
                result=raw_result,
                step=step,
                call_index=call_index,
                tool_call_id=call_id,
                task=task,
                finish_reason=finish_reason,
                image_digest_sha256=image_digest,
                terminal_failed="vision_result_truncated",
            )
            content = _builtin_vision_tool_error(
                "vision_result_truncated",
                f"Vision result ended with finish_reason={finish_reason!r}; "
                "the partial result was discarded.",
                retryable=True,
            )
        else:
            accepted = ledger.record(
                image_tag=image_tag,
                result=raw_result,
                step=step,
                call_index=call_index,
                tool_call_id=call_id,
                task=task,
                finish_reason=finish_reason,
                image_digest_sha256=image_digest,
            )
            content = raw_result
            if not accepted:
                code = (
                    "vision_empty_output"
                    if not raw_result.strip()
                    else "vision_protocol_error"
                )
                content = _builtin_vision_tool_error(
                    code,
                    "Vision returned no complete, verifiable result; "
                    "the result was discarded.",
                    retryable=True,
                )
    except ValueError as exc:
        content = _builtin_vision_tool_error(
            "vision_invalid_arguments", str(exc), retryable=False
        )
        if image_tag:
            with suppress(ValueError):
                image_tag = registry.resolve(image_tag).tag
                ledger.record(
                    image_tag=image_tag,
                    result=content,
                    step=step,
                    call_index=call_index,
                    tool_call_id=call_id,
                    task=task,
                    terminal_failed="vision_invalid_arguments",
                )
    except (
        VisualProxyUpstreamError,
        VisualProxyTimeoutError,
        VisualProxyCapacityError,
    ) as exc:
        code, retryable = _vision_tool_error_from_exception(exc)
        content = _builtin_vision_tool_error(code, str(exc), retryable=retryable)
        if image is not None:
            ledger.record(
                image_tag=image.tag,
                result="",
                step=step,
                call_index=call_index,
                tool_call_id=call_id,
                task=task,
                terminal_failed=code,
            )

    return _BuiltinResult(
        call=tool_call,
        call_id=call_id,
        image_tag=image_tag,
        task=task,
        content=content,
        accepted=accepted,
    )


async def _run_agent_choice(
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
    public_reasoning_enabled: bool = True,
    stream_callback: Optional[StreamEventCallback] = None,
) -> Union[tuple[ChatCompletionResponseChoice, UsageInfo], Response]:
    """Run one observational agent choice while keeping builtin execution private."""

    payload = copy.deepcopy(request_payload)
    public_separate_reasoning = payload.get("separate_reasoning")
    payload["separate_reasoning"] = True
    payload["stream"] = stream_callback is not None
    if stream_callback is not None:
        # Internal usage accounting must not depend on whether the caller asks
        # for the final public usage chunk.
        payload["stream_options"] = {"include_usage": True}
    payload["n"] = 1

    external_tools = copy.deepcopy(payload.get("tools") or [])
    builtin_enabled = not _has_external_vision_reader(external_tools)
    original_tool_choice = payload.get("tool_choice", "auto")

    ledger = EvidenceLedger(registry)
    aggregate_usage = UsageInfo(prompt_tokens_details=PromptTokensDetails())
    public_reasoning_parts: list[str] = []
    public_reasoning_is_trace: list[bool] = []
    public_trace_index = 0

    async def emit_public_reasoning(value: str, *, force: bool = False) -> None:
        if (not public_reasoning_enabled and not force) or not value:
            return
        if (
            len("\n".join((*public_reasoning_parts, value)).encode("utf-8"))
            > MAX_REASONING_CONTEXT_BYTES
        ):
            raise VisualChatProxyError(
                "Visual reasoning context exceeds the configured size limit"
            )
        separate = bool(public_reasoning_parts)
        public_reasoning_parts.append(value)
        public_reasoning_is_trace.append(force)
        if stream_callback is not None:
            await stream_callback("trace" if force else "reasoning", value, separate)

    def public_message_from_model(message: ChatMessage) -> ChatMessage:
        data = message.model_dump(exclude_none=True)
        data.pop("reasoning", None)
        data.pop("reasoning_content", None)
        reasoning = "\n".join(public_reasoning_parts)
        if public_separate_reasoning is False:
            model_reasoning = "\n".join(
                value
                for value, is_trace in zip(
                    public_reasoning_parts, public_reasoning_is_trace
                )
                if not is_trace
            )
            trace_reasoning = "\n".join(
                value
                for value, is_trace in zip(
                    public_reasoning_parts, public_reasoning_is_trace
                )
                if is_trace
            )
            if model_reasoning:
                data["content"] = model_reasoning + (data.get("content") or "")
            if trace_reasoning:
                data["reasoning"] = trace_reasoning
        elif reasoning:
            data["reasoning"] = reasoning
        return ChatMessage.model_validate(data)

    def configure_tools_for_step() -> None:
        payload["tools"] = copy.deepcopy(external_tools)
        if builtin_enabled:
            payload["tools"].append(copy.deepcopy(BUILTIN_VISION_READER_TOOL))
        payload["tool_choice"] = original_tool_choice

    for step in range(1, max_agent_steps + 1):
        if await _request_is_disconnected(raw_request):
            raise ClientDisconnected(
                reason="client disconnected during visual agent loop"
            )
        configure_tools_for_step()
        main_request = ChatCompletionRequest.model_validate(payload)
        if request_has_images(main_request):
            raise VisualChatProxyError(
                "Internal invariant failed: real image payload reached the main model request"
            )
        logger.info(
            "[visual-chat-proxy][main_model_request] trace_id=%s step=%d registered_images=%d image_payload_count=0",
            trace_id,
            step,
            len(registry),
        )
        if trace_recorder.enabled:
            trace_recorder.event(
                "main_model_request",
                choice_trace_id=trace_id,
                step=step,
                request=main_request.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
            )
        try:
            with bind_main_model_trace(trace_recorder, trace_id, step):
                response = await main_chat_handler(main_request, raw_request)
            if stream_callback is not None and isinstance(
                response, CustomStreamingResponse
            ):

                async def buffered_callback(
                    kind: str, value: str, separate: bool
                ) -> None:
                    if kind == "ready":
                        await stream_callback(kind, value, separate)

                response = await _consume_main_model_stream(
                    response,
                    model=str(payload.get("model") or "default"),
                    callback=buffered_callback,
                    separate_from_prior_reasoning=False,
                    stream_content=False,
                    preserve_model_text=True,
                )
            elif stream_callback is not None and isinstance(
                response, ChatCompletionResponse
            ):
                await stream_callback("ready", "1", False)
        except BaseException as exc:
            if trace_recorder.enabled:
                trace_recorder.event(
                    "main_model_error",
                    choice_trace_id=trace_id,
                    step=step,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            raise

        if trace_recorder.enabled:
            trace_recorder.event(
                "main_model_response",
                choice_trace_id=trace_id,
                step=step,
                response=_response_for_trace(response),
            )
        if not isinstance(response, ChatCompletionResponse):
            return response
        _usage_add(aggregate_usage, response.usage)
        if not response.choices:
            raise VisualChatProxyError("Main model returned no choices")

        choice = response.choices[0]
        message = choice.message.model_copy(deep=True)
        builtin_calls: list[Any] = []
        external_calls: list[Any] = []
        for tool_call in message.tool_calls or []:
            if builtin_enabled and tool_call.function.name == VISION_READER_NAME:
                builtin_calls.append(tool_call)
            else:
                external_calls.append(tool_call)

        step_reasoning = (
            message.reasoning
            if isinstance(message.reasoning, str)
            else message.reasoning_content
            if isinstance(message.reasoning_content, str)
            else ""
        )

        builtin_results = []
        for call_index, tool_call in enumerate(builtin_calls):
            builtin_results.append(
                await _execute_builtin_call(
                    tool_call=tool_call,
                    call_index=call_index,
                    step=step,
                    model=str(payload.get("model") or "default"),
                    registry=registry,
                    ledger=ledger,
                    aggregate_usage=aggregate_usage,
                    runtime=runtime,
                    raw_request=raw_request,
                    trace_id=trace_id,
                    trace_recorder=trace_recorder,
                    visual_handler=visual_handler,
                )
            )
        if builtin_results:
            if step_reasoning:
                await emit_public_reasoning(step_reasoning)
            for item in builtin_results:
                if item.accepted:
                    projection = _format_public_builtin_projection(
                        item.image_tag,
                        item.task,
                        item.content,
                        public_trace_index,
                        runtime.settings.builtin_trace_format,
                    )
                    public_trace_index += 1
                    await emit_public_reasoning(projection, force=True)

            internal_message = message.model_dump(exclude_none=True)
            internal_message["tool_calls"] = []
            for item in builtin_results:
                call_data = item.call.model_dump(exclude_none=True)
                call_data["id"] = item.call_id
                internal_message["tool_calls"].append(call_data)
            payload["messages"].append(internal_message)
            for item in builtin_results:
                payload["messages"].append(
                    {
                        "role": "tool",
                        "name": VISION_READER_NAME,
                        "tool_call_id": item.call_id,
                        "content": item.content,
                    }
                )
            if external_calls:
                message_data = message.model_dump(exclude_none=True)
                message_data["tool_calls"] = [
                    call.model_dump(exclude_none=True) for call in external_calls
                ]
                message_data.pop("reasoning", None)
                message_data.pop("reasoning_content", None)
                public_message = public_message_from_model(
                    ChatMessage.model_validate(message_data)
                )
                return (
                    ChatCompletionResponseChoice(
                        index=0,
                        message=public_message,
                        finish_reason="tool_calls",
                    ),
                    aggregate_usage,
                )
            continue

        if external_calls:
            if step_reasoning:
                await emit_public_reasoning(step_reasoning)
            message_data = message.model_dump(exclude_none=True)
            message_data["tool_calls"] = [
                call.model_dump(exclude_none=True) for call in external_calls
            ]
            message_data.pop("reasoning", None)
            message_data.pop("reasoning_content", None)
            public_message = public_message_from_model(
                ChatMessage.model_validate(message_data)
            )
            return (
                ChatCompletionResponseChoice(
                    index=0,
                    message=public_message,
                    finish_reason="tool_calls",
                ),
                aggregate_usage,
            )

        if step_reasoning:
            await emit_public_reasoning(step_reasoning)
        public_message = public_message_from_model(message)
        finish_reason = choice.finish_reason
        if trace_recorder.enabled:
            trace_recorder.event(
                "image_evidence_ledger",
                choice_trace_id=trace_id,
                evidence=ledger.snapshot(visual_answer_required=False),
            )
        return (
            ChatCompletionResponseChoice(
                index=0,
                message=public_message,
                finish_reason=finish_reason,
            ),
            aggregate_usage,
        )

    raise VisualChatProxyError(
        f"Visual agent loop exceeded max steps ({max_agent_steps})"
    )


def _response_for_trace(response: Any) -> Any:
    if isinstance(response, ChatCompletionResponse):
        return response.model_dump(mode="json", exclude_none=True)
    if not isinstance(response, Response):
        return response

    body = getattr(response, "body", None)
    parsed_body: Any = None
    if isinstance(body, bytes):
        try:
            parsed_body = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed_body = body.decode("utf-8", errors="replace")
    return {
        "status_code": response.status_code,
        "media_type": response.media_type,
        "body": parsed_body,
    }
