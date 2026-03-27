import time
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Dict, List, Optional, Union, Literal, ClassVar, TypeAlias
from transformers import GenerationConfig


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

    name: Optional[str] = None
    description: Optional[str] = Field(default=None, examples=[None])
    parameters: Optional[dict] = None
    response: Optional[dict] = None


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


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[dict[str, Any]] = Field(default=None, alias="schema")
    strict: Optional[bool] = None


class ResponseFormat(BaseModel):
    # type must be "json_schema", "json_object", or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class FunctionResponse(BaseModel):
    """Function response."""

    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    """Tool call response."""

    id: Optional[str] = None
    index: Optional[int] = None
    type: Literal["function"] = "function"
    function: FunctionResponse


class ChatCompletionMessageGenericParam(BaseModel):
    role: Literal["system", "assistant", "tool", "function"]
    content: Union[str, List[MessageContent], None] = Field(default=None)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v):
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in {"system", "assistant", "tool", "function"}:
                raise ValueError(
                    "'role' must be one of 'system', 'assistant', 'tool', or 'function' (case-insensitive)."
                )
            return v_lower
        raise ValueError("'role' must be a string")


ChatCompletionMessageParam = Union[ChatCompletionMessageGenericParam, Message]


class CompletionRequest(BaseModel):
    model: str
    # prompt: string or tokens
    prompt: Union[str, List[str], List[int], List[List[int]]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=(
            "Similar to chat completion, this parameter specifies the format "
            "of output. Only {'type': 'json_object'}, {'type': 'json_schema'}"
            ", or {'type': 'text' } is supported."
        ),
    )

    # Additional parameters supported by LightLLM
    do_sample: Optional[bool] = True
    top_k: Optional[int] = -1
    repetition_penalty: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False

    # Class variables to store loaded default values
    _loaded_defaults: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def load_generation_cfg(cls, weight_dir: str):
        """Load default values from model generation config."""
        try:
            generation_cfg = GenerationConfig.from_pretrained(weight_dir, trust_remote_code=True).to_dict()
            cls._loaded_defaults = {
                "do_sample": generation_cfg.get("do_sample", True),
                "presence_penalty": generation_cfg.get("presence_penalty", 0.0),
                "frequency_penalty": generation_cfg.get("frequency_penalty", 0.0),
                "repetition_penalty": generation_cfg.get("repetition_penalty", 1.0),
                "temperature": generation_cfg.get("temperature", 1.0),
                "top_p": generation_cfg.get("top_p", 1.0),
                "top_k": generation_cfg.get("top_k", -1),
            }
            # Remove None values
            cls._loaded_defaults = {k: v for k, v in cls._loaded_defaults.items() if v is not None}
        except Exception:
            pass

    @model_validator(mode="before")
    @classmethod
    def apply_loaded_defaults(cls, data: Any):
        """Apply loaded default values if field is not provided."""
        if isinstance(data, dict) and cls._loaded_defaults:
            for key, value in cls._loaded_defaults.items():
                if key not in data:
                    data[key] = value
        return data


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]
    function_call: Optional[str] = "none"
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 8192
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=(
            "Similar to chat completion, this parameter specifies the format "
            "of output. Only {'type': 'json_object'}, {'type': 'json_schema'}"
            ", or {'type': 'text' } is supported."
        ),
    )

    # OpenAI Adaptive parameters for tool call
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
        default="auto", examples=["none"]
    )  # noqa
    parallel_tool_calls: Optional[bool] = True

    # OpenAI parameters for reasoning and others
    chat_template_kwargs: Optional[Dict] = None
    separate_reasoning: Optional[bool] = True
    stream_reasoning: Optional[bool] = False

    # Additional parameters supported by LightLLM
    do_sample: Optional[bool] = True
    top_k: Optional[int] = -1
    repetition_penalty: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False
    role_settings: Optional[Dict[str, str]] = None
    character_settings: Optional[List[Dict[str, str]]] = None

    # Class variables to store loaded default values
    _loaded_defaults: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def load_generation_cfg(cls, weight_dir: str):
        """Load default values from model generation config."""
        try:
            generation_cfg = GenerationConfig.from_pretrained(weight_dir, trust_remote_code=True).to_dict()
            cls._loaded_defaults = {
                "do_sample": generation_cfg.get("do_sample", True),
                "presence_penalty": generation_cfg.get("presence_penalty", 0.0),
                "frequency_penalty": generation_cfg.get("frequency_penalty", 0.0),
                "repetition_penalty": generation_cfg.get("repetition_penalty", 1.0),
                "temperature": generation_cfg.get("temperature", 1.0),
                "top_p": generation_cfg.get("top_p", 1.0),
                "top_k": generation_cfg.get("top_k", -1),
            }
            # Remove None values
            cls._loaded_defaults = {k: v for k, v in cls._loaded_defaults.items() if v is not None}
        except Exception:
            pass

    @model_validator(mode="before")
    @classmethod
    def apply_loaded_defaults(cls, data: Any):
        """Apply loaded default values if field is not provided."""
        if isinstance(data, dict) and cls._loaded_defaults:
            for key, value in cls._loaded_defaults.items():
                if key not in data:
                    data[key] = value
        return data


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[Union[str, List[MessageContent]]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


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
    content: Optional[Union[str, List[MessageContent]]] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])
    reasoning_content: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]
    usage: Optional[UsageInfo] = None

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)


class CompletionLogprobs(BaseModel):
    tokens: List[str] = []
    token_logprobs: List[Optional[float]] = []
    top_logprobs: List[Optional[Dict[str, float]]] = []
    text_offset: List[int] = []


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional["CompletionLogprobs"] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)


class CompletionStreamChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionStreamChoice]
    usage: Optional[UsageInfo] = None

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)


class ChatCompletionStreamResponseV2(ChatCompletionStreamResponse):
    pass


class ChatCompletionResponseV2(ChatCompletionRequest):
    pass


# Supported values
AspectRatio: TypeAlias = Literal[
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9",
]

ImageSize: TypeAlias  = Literal["0.5K", "1K", "2K", "4K"]

Modality: TypeAlias = Literal["text", "image", "audio"]

ImageType: TypeAlias = Literal["png", "jpeg", "webp"]

class ImageConfig(BaseModel):
    aspect_ratio: AspectRatio = "1:1"
    image_size: ImageSize = "1K"
    image_type: ImageType = "jpeg"

    # Mapping to actual resolutions (base resolution for 1K)
    _aspect_ratio_to_resolution: ClassVar[dict] = {
        "1:1": (1024, 1024),
        "2:3": (832, 1248),
        "3:2": (1248, 832),
        "3:4": (864, 1184),
        "4:3": (1184, 864),
        "4:5": (896, 1152),
        "5:4": (1152, 896),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
        "21:9": (1536, 672),
    }

    _size_multiplier: ClassVar[dict] = {
        "0.5K": 0.5,
        "1K": 1.0,
        "2K": 2.0,
        "4K": 4.0,
    }

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v):
        if v not in cls._aspect_ratio_to_resolution:
            raise ValueError(f"Unsupported aspect ratio: {v}")
        return v

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v):
        if v not in cls._size_multiplier:
            raise ValueError(f"Unsupported image size: {v}")
        return v

    @field_validator("image_type")
    @classmethod
    def validate_image_type(cls, v):
        if v not in ['jpeg', 'png', 'webp']:
            raise ValueError(f"unsupported image type: {v}")
        return v

    def get_resolution(self):
        """Return scaled resolution (width, height)"""
        base = self._aspect_ratio_to_resolution[self.aspect_ratio]
        if base is None:
            return None  # extended ratios don't have fixed base

        scale = self._size_multiplier[self.image_size]
        w, h = base
        return int(w * scale), int(h * scale)


class ChatCompletionRequestV2(ChatCompletionRequest):
    modalities: List[Modality] = ["text"]
    image_config: Optional[ImageConfig] = None

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, v):
        if "text" not in v:
            raise ValueError("modalities must include 'text'")
        if len(v) != len(set(v)):
            raise ValueError("modalities must be unique")
        return v

    @model_validator(mode="after")
    def validate_image_config(self):
        if "image" in self.modalities:
            if self.image_config is None:
                self.image_config = ImageConfig()
        else:
            if self.image_config is not None:
                raise ValueError("image_config provided but 'image' not in modalities")
        return self