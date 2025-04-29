import time

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union, Literal
import uuid


class ImageURL(BaseModel):
    url: str


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]


class Function(BaseModel):
    """Function descriptions."""

    description: Optional[str] = Field(default=None, examples=[None])
    name: Optional[str] = None
    parameters: Optional[object] = None


class Tool(BaseModel):
    """Function wrapper."""

    type: str = Field(default="function", examples=["function"])
    function: Function


class ToolChoiceFuncName(BaseModel):
    """The name of tool choice function."""

    name: Optional[str] = None


class ToolChoice(BaseModel):
    """The tool choice definition."""

    function: ToolChoiceFuncName
    type: Literal["function"] = Field(default="function", examples=["function"])


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    function_call: Optional[str] = "none"
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # OpenAI Adaptive parameters for tool call
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
        default="auto", examples=["none"]
    )  # noqa

    # Additional parameters supported by LightLLM
    do_sample: Optional[bool] = False
    top_k: Optional[int] = -1
    repetition_penalty: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False
    role_settings: Optional[Dict[str, str]] = None
    character_settings: Optional[List[Dict[str, str]]] = None


class FunctionResponse(BaseModel):
    """Function response."""

    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    """Tool call response."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionResponse


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)
