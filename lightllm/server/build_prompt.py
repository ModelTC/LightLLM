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


async def build_prompt(request, tools) -> str:
    global tokenizer
    import json

    # pydantic格式转成dict， 否则，当根据tokenizer_config.json拼template时，Jinja判断无法识别
    messages = [m.model_dump(by_alias=True, exclude_none=True) for m in request.messages]

    # 当有工具调用时，content 被设置为None，被exclude_none=True排除，
    # 导致后续模板处理无法识别， 这里补齐content字段为""， 以兼容原有模板逻辑。
    for msg in messages:
        if "content" not in msg:
            msg["content"] = ""

    # 对于工具调用的消息，确保 tool_calls 字段存在且格式正确， 以兼容模板中对工具调用的处理逻辑。
    for msg in messages:
        if msg.get("role") != "assistant" or "tool_calls" not in msg:
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        # Drop empty tool_calls so templates take the normal assistant path
        if len(tool_calls) == 0:
            msg.pop("tool_calls", None)
            continue
        for tool_call in tool_calls:
            func = tool_call.get("function")
            if not func or not isinstance(func, dict):
                continue
            args = func.get("arguments")
            if args and not isinstance(args, (dict, list)):
                try:
                    func["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    func["arguments"] = {}
            elif not args:
                # Missing or empty arguments default to empty dict
                func["arguments"] = {}

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
