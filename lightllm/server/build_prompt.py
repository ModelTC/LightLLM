import os
import json
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger

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


async def build_prompt(request, tools) -> str:
    global tokenizer
    # pydantic格式转成dict， 否则，当根据tokenizer_config.json拼template时，Jinja判断无法识别
    messages = [m.model_dump(by_alias=True, exclude_none=True) for m in request.messages]
    _normalize_tool_call_arguments(messages)
    _alias_reasoning_to_reasoning_content(messages)

    kwargs = {"conversation": messages}
    if request.character_settings:
        kwargs["character_settings"] = request.character_settings
    if request.role_settings:
        kwargs["role_setting"] = request.role_settings

    if request.chat_template_kwargs:
        kwargs.update(request.chat_template_kwargs)

    try:
        input_str = tokenizer.apply_chat_template(**kwargs, tokenize=False, add_generation_prompt=True, tools=tools)
    except BaseException as e:
        logger.error(f"Failed to build prompt: {e}")
        raise e
    return input_str
