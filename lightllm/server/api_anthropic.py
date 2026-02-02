import json
import logging
import time
from typing import Any, AsyncGenerator, Literal

from pydantic import BaseModel, field_validator

from .api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Function,
    StreamOptions,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)

logger = logging.getLogger(__name__)


class AnthropicError(BaseModel):
    """Error structure for Anthropic API"""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API"""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicContentBlock(BaseModel):
    """Content block in message"""

    type: Literal["text", "image", "tool_use", "tool_result"]
    text: str | None = None
    # For image content
    source: dict[str, Any] | None = None
    # For tool use/result
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"  # Default to object type
        return v


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition"""

    type: Literal["auto", "any", "tool"]
    name: str | None = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: Literal["text_delta", "input_json_delta"] | None = None
    text: str | None = None
    partial_json: str | None = None

    # Message delta
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: "AnthropicMessagesResponse | None" = None
    delta: AnthropicDelta | None = None
    content_block: AnthropicContentBlock | None = None
    index: int | None = None
    error: AnthropicError | None = None
    usage: AnthropicUsage | None = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage | None = None

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


ANTHROPIC_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _convert_anthropic_to_chat_request(
    anthropic_request: AnthropicMessagesRequest,
) -> ChatCompletionRequest:
    """Convert Anthropic message format to OpenAI-style ChatCompletionRequest"""

    openai_messages: list[dict[str, Any]] = []

    # Add system message if provided
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            openai_messages.append(
                {"role": "system", "content": anthropic_request.system}
            )
        else:
            system_prompt = ""
            for block in anthropic_request.system:
                if block.type == "text" and block.text:
                    system_prompt += block.text
            openai_messages.append({"role": "system", "content": system_prompt})

    # Convert main messages
    for msg in anthropic_request.messages:
        openai_msg: dict[str, Any] = {"role": msg.role}
        if isinstance(msg.content, str):
            openai_msg["content"] = msg.content
        else:
            content_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in msg.content:
                if block.type == "text" and block.text:
                    content_parts.append({"type": "text", "text": block.text})
                elif block.type == "image" and block.source:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": block.source.get("data", "")},
                        }
                    )
                elif block.type == "tool_use":
                    tool_call = {
                        "id": block.id or f"call_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                    tool_calls.append(tool_call)
                elif block.type == "tool_result":
                    if msg.role == "user":
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.id or "",
                                "content": str(block.content) if block.content else "",
                            }
                        )
                    else:
                        tool_result_text = (
                            str(block.content) if block.content else ""
                        )
                        content_parts.append(
                            {
                                "type": "text",
                                "text": f"Tool result: {tool_result_text}",
                            }
                        )

            if tool_calls:
                openai_msg["tool_calls"] = tool_calls

            if content_parts:
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    openai_msg["content"] = content_parts[0]["text"]
                else:
                    openai_msg["content"] = content_parts
            elif not tool_calls:
                continue

        openai_messages.append(openai_msg)

    # Build kwargs, excluding None values to allow defaults to be used
    chat_req_kwargs = {
        "model": anthropic_request.model,
        "messages": openai_messages,
        "max_tokens": anthropic_request.max_tokens,
    }
    
    # Only include optional parameters if they are not None
    if anthropic_request.stop_sequences is not None:
        chat_req_kwargs["stop"] = anthropic_request.stop_sequences
    if anthropic_request.temperature is not None:
        chat_req_kwargs["temperature"] = anthropic_request.temperature
    if anthropic_request.top_p is not None:
        chat_req_kwargs["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k is not None:
        chat_req_kwargs["top_k"] = anthropic_request.top_k
    
    chat_req = ChatCompletionRequest(**chat_req_kwargs)

    # Streaming options: always include usage for Anthropic streaming
    if anthropic_request.stream:
        chat_req.stream = True
        chat_req.stream_options = StreamOptions(include_usage=True)
    else:
        chat_req.stream = False
        chat_req.stream_options = None

    # Tool choice mapping
    tools: list[Tool] = []
    if anthropic_request.tools is not None:
        for tool in anthropic_request.tools:
            tools.append(
                Tool(
                    type="function",
                    function=Function(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.input_schema,
                    ),
                )
            )
        chat_req.tools = tools

    # Map tool_choice
    if anthropic_request.tool_choice is None:
        if anthropic_request.tools:
            chat_req.tool_choice = "auto"
        else:
            chat_req.tool_choice = "none"
    else:
        tc = anthropic_request.tool_choice
        if tc.type == "auto":
            chat_req.tool_choice = "auto"
        elif tc.type == "any":
            chat_req.tool_choice = "required"
        elif tc.type == "tool" and tc.name:
            chat_req.tool_choice = ToolChoice(
                function=ToolChoiceFuncName(name=tc.name),
            )

    return chat_req


def _convert_chat_response_to_anthropic(
    chat_resp: ChatCompletionResponse,
) -> AnthropicMessagesResponse:
    """Convert ChatCompletionResponse to AnthropicMessagesResponse"""

    usage = AnthropicUsage(
        input_tokens=chat_resp.usage.prompt_tokens,
        output_tokens=chat_resp.usage.completion_tokens or 0,
    )

    content: list[AnthropicContentBlock] = [
        AnthropicContentBlock(
            type="text",
            text=chat_resp.choices[0].message.content or "",
        )
    ]

    for tool_call in chat_resp.choices[0].message.tool_calls or []:
        anthropic_tool_call = AnthropicContentBlock(
            type="tool_use",
            id=tool_call.id,
            name=tool_call.function.name if tool_call.function else None,
            input=json.loads(tool_call.function.arguments or "{}"),
        )
        content.append(anthropic_tool_call)

    finish_reason = chat_resp.choices[0].finish_reason
    stop_reason = ANTHROPIC_STOP_REASON_MAP.get(finish_reason) if finish_reason else None

    result = AnthropicMessagesResponse(
        id=str(chat_resp.id),
        content=content,
        model=chat_resp.model,
        stop_reason=stop_reason,
        usage=usage,
    )

    return result


def _wrap_data_with_event(data: str, event: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _anthropic_message_stream_from_chat(
    source,
) -> AsyncGenerator[str, None]:
    """Convert OpenAI-style SSE stream to Anthropic Messages streaming events"""

    buffer = ""
    first_item = True
    finish_reason: str | None = None
    content_block_index = 0
    content_block_started = False
    sent_message_stop = False

    try:
        async for chunk in source:
            if isinstance(chunk, bytes):
                text = chunk.decode("utf-8")
            else:
                text = str(chunk)

            buffer += text
            while "\n\n" in buffer:
                event, buffer = buffer.split("\n\n", 1)
                event = event.strip()
                if not event:
                    continue
                if not event.startswith("data:"):
                    error_response = AnthropicStreamEvent(
                        type="error",
                        error=AnthropicError(
                            type="internal_error",
                            message="Invalid data format received",
                        ),
                    )
                    data = error_response.model_dump_json(exclude_unset=True)
                    yield _wrap_data_with_event(data, "error")
                    sent_message_stop = True
                    yield "data: [DONE]\n\n"
                    return

                data_str = event[5:].strip().rstrip("\n")

                if data_str == "[DONE]":
                    stop_message = AnthropicStreamEvent(
                        type="message_stop",
                    )
                    data = stop_message.model_dump_json(
                        exclude_unset=True, exclude_none=True
                    )
                    yield _wrap_data_with_event(data, "message_stop")
                    sent_message_stop = True
                    yield "data: [DONE]\n\n"
                    return

                origin_chunk = ChatCompletionStreamResponse.model_validate_json(
                    data_str
                )

                if first_item:
                    chunk = AnthropicStreamEvent(
                        type="message_start",
                        message=AnthropicMessagesResponse(
                            id=origin_chunk.id,
                            content=[],
                            model=origin_chunk.model,
                            usage=AnthropicUsage(
                                input_tokens=origin_chunk.usage.prompt_tokens
                                if origin_chunk.usage
                                else 0,
                                output_tokens=0,
                            ),
                        ),
                    )
                    first_item = False
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield _wrap_data_with_event(data, "message_start")
                    continue

                # usage-only chunk (choices empty)
                if len(origin_chunk.choices) == 0:
                    if content_block_started:
                        stop_chunk = AnthropicStreamEvent(
                            index=content_block_index,
                            type="content_block_stop",
                        )
                        data = stop_chunk.model_dump_json(exclude_unset=True)
                        yield _wrap_data_with_event(data, "content_block_stop")
                        content_block_started = False

                    stop_reason = ANTHROPIC_STOP_REASON_MAP.get(
                        finish_reason or "stop"
                    )
                    chunk = AnthropicStreamEvent(
                        type="message_delta",
                        delta=AnthropicDelta(stop_reason=stop_reason),
                        usage=AnthropicUsage(
                            input_tokens=origin_chunk.usage.prompt_tokens
                            if origin_chunk.usage
                            else 0,
                            output_tokens=origin_chunk.usage.completion_tokens
                            if origin_chunk.usage
                            else 0,
                        ),
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield _wrap_data_with_event(data, "message_delta")
                    continue

                if origin_chunk.choices[0].finish_reason is not None:
                    finish_reason = origin_chunk.choices[0].finish_reason
                    continue

                # content tokens
                if origin_chunk.choices[0].delta.content is not None:
                    if not content_block_started:
                        chunk = AnthropicStreamEvent(
                            index=content_block_index,
                            type="content_block_start",
                            content_block=AnthropicContentBlock(
                                type="text", text=""
                            ),
                        )
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield _wrap_data_with_event(data, "content_block_start")
                        content_block_started = True

                    if origin_chunk.choices[0].delta.content == "":
                        continue

                    chunk = AnthropicStreamEvent(
                        index=content_block_index,
                        type="content_block_delta",
                        delta=AnthropicDelta(
                            type="text_delta",
                            text=origin_chunk.choices[0].delta.content,
                        ),
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield _wrap_data_with_event(data, "content_block_delta")
                    continue

                # tool calls
                elif (
                    origin_chunk.choices[0].delta.tool_calls
                    and len(origin_chunk.choices[0].delta.tool_calls) > 0
                ):
                    tool_call = origin_chunk.choices[0].delta.tool_calls[0]
                    if tool_call.id is not None:
                        if content_block_started:
                            stop_chunk = AnthropicStreamEvent(
                                index=content_block_index,
                                type="content_block_stop",
                            )
                            data = stop_chunk.model_dump_json(
                                exclude_unset=True
                            )
                            yield _wrap_data_with_event(
                                data, "content_block_stop"
                            )
                            content_block_started = False
                            content_block_index += 1

                        chunk = AnthropicStreamEvent(
                            index=content_block_index,
                            type="content_block_start",
                            content_block=AnthropicContentBlock(
                                type="tool_use",
                                id=tool_call.id,
                                name=tool_call.function.name
                                if tool_call.function
                                else None,
                                input={},
                            ),
                        )
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield _wrap_data_with_event(data, "content_block_start")
                        content_block_started = True
                    else:
                        chunk = AnthropicStreamEvent(
                            index=content_block_index,
                            type="content_block_delta",
                            delta=AnthropicDelta(
                                type="input_json_delta",
                                partial_json=tool_call.function.arguments
                                if tool_call.function
                                else None,
                            ),
                        )
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield _wrap_data_with_event(data, "content_block_delta")
                    continue

        # after source is exhausted, emit message_stop if not already sent
        if not sent_message_stop:
            stop_message = AnthropicStreamEvent(
                type="message_stop",
            )
            data = stop_message.model_dump_json(
                exclude_unset=True, exclude_none=True
            )
            yield _wrap_data_with_event(data, "message_stop")
            yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error in Anthropic message stream converter.")
        error_response = AnthropicStreamEvent(
            type="error",
            error=AnthropicError(type="internal_error", message=str(e)),
        )
        data = error_response.model_dump_json(exclude_unset=True)
        yield _wrap_data_with_event(data, "error")
        yield "data: [DONE]\n\n"
