import time
from typing_extensions import deprecated
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Dict, List, Optional, Union, Literal, ClassVar, TypeAlias
from transformers import GenerationConfig


class ImageURL(BaseModel):
    url: str


class AudioURL(BaseModel):
    url: str


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None
    audio_url: Optional[AudioURL] = None


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
    max_tokens: Optional[int] = Field(
        default=256000, deprecated="max_tokens is deprecated, please use max_completion_tokens instead"
    )
    max_completion_tokens: Optional[int] = None
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
    seed: Optional[int] = -1

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
    max_tokens: Optional[int] = Field(
        default=256000, deprecated="max_tokens is deprecated, please use max_completion_tokens instead"
    )
    max_completion_tokens: Optional[int] = None
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
    seed: Optional[int] = -1
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
    # OpenRouter-style: generated images alongside text; content may include "<image>" placeholders
    images: Optional[List[MessageContent]] = None


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
    images: Optional[List[MessageContent]] = None


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


# Supported values
AspectRatio: TypeAlias = Literal[
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]

ImageSize: TypeAlias = Literal["0.5K", "1K", "1.5K", "2K", "4K"]

Modality: TypeAlias = Literal["text", "image", "audio"]

ImageType: TypeAlias = Literal["png", "jpeg", "webp"]


class ImageConfig(BaseModel):
    aspect_ratio: AspectRatio = "1:1"
    image_size: ImageSize = "1.5K"
    image_type: ImageType = "jpeg"
    height: Optional[int] = -1
    width: Optional[int] = -1
    # X2I / diffusion sampling (optional; server defaults apply when omitted)
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    image_guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    num_images: Optional[int] = None
    cfg_norm: Optional[Literal["none", "cfg_zero_star", "global", "text_channel", "channel"]] = None
    dynamic_resolution: Optional[bool] = True
    _aspect_ratio_to_resolution: ClassVar[dict] = {
        "1:1": {"1K": (1024, 1024), "1.5K": (1536, 1536), "2K": (2048, 2048)},
        "16:9": {"1.5K": (2048, 1152), "2K": (2720, 1536)},
        "9:16": {"1.5K": (1152, 2048), "2K": (1536, 2720)},
        "3:2": {"1.5K": (1888, 1248), "2K": (2496, 1664)},
        "2:3": {"1.5K": (1248, 1888), "2K": (1664, 2496)},
        "4:3": {"1.5K": (1760, 1312), "2K": (2368, 1760)},
        "3:4": {"1.5K": (1312, 1760), "2K": (1760, 2368)},
        "1:2": {"1.5K": (1088, 2144), "2K": (1440, 2880)},
        "2:1": {"1.5K": (2144, 1088), "2K": (2880, 1440)},
        "1:3": {"1.5K": (864, 2592), "2K": (1152, 3456)},
        "3:1": {"1.5K": (2592, 864), "2K": (3456, 1152)},
    }
    _size_set: ClassVar[set[str]] = {"1.5K", "2K"}

    @field_validator("image_size", mode="before")
    @classmethod
    def normalize_image_size(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v

    @model_validator(mode="after")
    def validate_resolution_config(self):
        has_custom_height = self.height is not None and self.height > 0
        has_custom_width = self.width is not None and self.width > 0
        has_any_custom = (self.height is not None and self.height != -1) or (
            self.width is not None and self.width != -1
        )

        # If custom resolution is provided, require both height/width and both must be positive.
        if has_any_custom:
            if not has_custom_height or not has_custom_width:
                raise ValueError("height and width must both be provided as positive integers")
            self.dynamic_resolution = False
            return self

        # Otherwise, validate ratio and logical image size.
        if self.aspect_ratio not in self._aspect_ratio_to_resolution:
            raise ValueError(f"Unsupported aspect ratio: {self.aspect_ratio}")
        if self.image_size not in self._size_set:
            raise ValueError(f"Unsupported image size: {self.image_size}")
        return self

    @field_validator("image_type")
    @classmethod
    def validate_image_type(cls, v):
        if v not in ["jpeg", "png", "webp"]:
            raise ValueError(f"unsupported image type: {v}")
        return v

    def get_resolution(self):
        """Return scaled resolution (width, height)"""
        from lightllm.models.neo_chat_moe.vision_process import smart_resize

        print(f"self.height: {self.height}, self.width: {self.width}", flush=True)
        if self.height > -1 and self.width > -1:
            w, h = self.width, self.height
        else:
            base = self._aspect_ratio_to_resolution[self.aspect_ratio][self.image_size]
            w, h = base

        h, w = smart_resize(h, w, factor=32, min_pixels=1024 * 1024, max_pixels=2048 * 2048)
        return w, h


class ChatCompletionRequestV2(ChatCompletionRequest):
    modalities: List[Modality] = ["text"]
    image_config: Optional[ImageConfig] = None

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, v):
        if "text" not in v and v != ["image"]:
            raise ValueError("modalities must include 'text', or be ['image'] for image-only generation")
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
