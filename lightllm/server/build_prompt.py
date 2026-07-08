import os
import json
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger
from functools import lru_cache
from lightllm.utils.config_utils import get_model_type_v1
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)

tokenizer = None


def init_tokenizer(args):
    global tokenizer

    tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)
    chat_path = args.chat_template
    if chat_path is not None:
        with open(chat_path, "r", encoding="utf-8") as f:
            chat_template_str = f.read()
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.chat_template = chat_template_str
        else:
            tokenizer.chat_template = chat_template_str
        return

    # 如果 tokenizer 目录下存在chat_template.json， 同时不存在 chat_template.jinja,
    # 则加载其并赋值给tokenizer 的 chat_template 对象。
    if not os.path.exists(os.path.join(args.model_dir, "chat_template.jinja")) and os.path.exists(
        os.path.join(args.model_dir, "chat_template.json")
    ):
        default_chat_template_path = os.path.join(args.model_dir, "chat_template.json")
        try:
            with open(default_chat_template_path, "r", encoding="utf-8") as f:
                template_data = json.load(f)
                if "chat_template" in template_data:
                    # Set it directly on the tokenizer object so apply_chat_template can use it
                    if hasattr(tokenizer, "tokenizer"):
                        # 多模态 tokenizer
                        tokenizer.tokenizer.chat_template = template_data["chat_template"]
                    else:
                        tokenizer.chat_template = template_data["chat_template"]

                    logger.info(f"Loaded chat_template.json from {default_chat_template_path}")
        except Exception as e:
            logger.warning(f"Failed to load chat_template.json from {default_chat_template_path}: {e}")
    return


@lru_cache(maxsize=1)
def tokenizer_supports_force_thinking() -> bool:
    """Whether this tokenizer supports thinking / reasoning."""

    assert tokenizer is not None

    try:
        ans = "thinking" in tokenizer.chat_template or "enable_thinking" in tokenizer.chat_template
        logger.debug(f"chat_template: {tokenizer.chat_template}")
        logger.info(f"tokenizer_supports_force_thinking : {ans}")
        return ans
    except:
        pass

    try:
        ans = "thinking" in tokenizer.tokenizer.chat_template or "enable_thinking" in tokenizer.tokenizer.chat_template
        logger.debug(f"tokenizer.tokenizer.chat_template: {tokenizer.tokenizer.chat_template}")
        logger.info(f"tokenizer_supports_force_thinking : {ans}")
        return ans
    except:
        pass

    logger.info("tokenizer_supports_force_thinking : False")
    return False


def _normalize_tool_call_arguments(messages: list) -> None:
    # Convert tool_calls function.arguments from JSON string to dict for Jinja template compatibility
    # Qwen35's chat template expects arguments to be a dict (uses |items filter)
    # but OpenAI format sends arguments as a JSON string
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tool_call in tool_calls:
                func = tool_call.get("function")
                if func and isinstance(func, dict):
                    args = func.get("arguments")
                    if isinstance(args, str) and args:
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            pass


def _alias_reasoning_to_reasoning_content(messages: list) -> None:
    # Clients (OpenRouter-style, claw-eval, and others) replay prior thinking on
    # assistant messages as `reasoning`, but Qwen3/Qwen3.5 chat templates read
    # `message.reasoning_content`. Without this alias the template falls back to
    # rendering every recent assistant turn as `<think>\n</think>` (empty think),
    # which teaches the model in-context to skip thinking on the current turn.
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if msg.get("reasoning_content"):
            continue
        reasoning = msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            msg["reasoning_content"] = reasoning


def _normalize_multimodal_content_types(messages: list) -> None:
    # OpenAI requests use content part types like `image_url` and `audio_url`.
    # Model chat templates generally render modality tokens from `image` and
    # `audio` parts while the raw media payload is carried separately in
    # MultimodalParams. Preserve the original fields and normalize only the
    # template-facing type to keep prompt tags aligned with media counts.
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                part["type"] = "image"
            elif part.get("type") == "audio_url":
                part["type"] = "audio"


def get_effective_chat_template_kwargs(request) -> dict:
    kwargs = {}
    default_kwargs = getattr(get_env_start_args(), "default_chat_template_kwargs", None)
    if default_kwargs:
        kwargs.update(default_kwargs)
    if request.chat_template_kwargs:
        kwargs.update(request.chat_template_kwargs)
    return kwargs


async def build_prompt(request, tools) -> str:
    # pydantic格式转成dict， 否则，当根据tokenizer_config.json拼template时，Jinja判断无法识别
    messages = [m.model_dump(by_alias=True, exclude_none=True) for m in request.messages]
    _normalize_tool_call_arguments(messages)
    _alias_reasoning_to_reasoning_content(messages)
    if get_model_type_v1() == "gemma4":
        # gemma4 的 tokenizer 不支持 multimodal 内容类型，所以需要手动转换
        _normalize_multimodal_content_types(messages)

    kwargs = {"conversation": messages}
    if request.character_settings:
        kwargs["character_settings"] = request.character_settings
    if request.role_settings:
        kwargs["role_setting"] = request.role_settings

    chat_template_kwargs = get_effective_chat_template_kwargs(request)
    if chat_template_kwargs:
        kwargs.update(chat_template_kwargs)

    # 修复一些parser类型是默认打开thinking，但是 tokenizer有时候不知道打开了thinking。导致
    # 构建的reasoning parser 和 tokenizer 的行为不对齐导致的问题。
    from .api_openai import _is_force_thinking_mode

    thinking = _is_force_thinking_mode(request)

    kwargs["thinking"] = thinking
    kwargs["enable_thinking"] = thinking

    # TODO thinking 模式应该是3种，一种是强制思考，一种是强制不思考，一种是模型自己决定的自适应
    # 的思考模式。当前的代码只是实现了强制思考和强制不思考两种模式。后续要根据模型的情况，从tokenizer
    # 上判断能支持的思考模式种类，再进行设置，才能具备更完备的处理。

    try:
        input_str = tokenizer.apply_chat_template(**kwargs, tokenize=False, add_generation_prompt=True, tools=tools)
    except Exception as e:
        raise ValueError(f"Failed to build prompt: {e}") from None
    return input_str
