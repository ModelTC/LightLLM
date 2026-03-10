tokenizer = None


def init_tokenizer(args):
    global tokenizer
    from lightllm.server.tokenizer import get_tokenizer
    import os
    import json
    from lightllm.utils.log_utils import init_logger

    logger = init_logger(__name__)

    tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)
    chat_path = args.chat_template
    if chat_path is not None:
        with open(chat_path, "r", encoding="utf-8") as f:
            chat_template_str = f.read()
        tokenizer.chat_template = chat_template_str
    else:
        # Try to load from chat_template.json in model_dir if tokenizer doesn't have it
        default_chat_template_path = os.path.join(args.model_dir, "chat_template.json")
        if os.path.exists(default_chat_template_path):
            try:
                with open(default_chat_template_path, "r", encoding="utf-8") as f:
                    template_data = json.load(f)
                    if "chat_template" in template_data:
                        # Set it directly on the tokenizer object so apply_chat_template can use it
                        tokenizer.chat_template = template_data["chat_template"]
                        logger.info(f"Loaded chat_template from {default_chat_template_path}")
            except Exception as e:
                logger.warning(f"Failed to load chat_template from {default_chat_template_path}: {e}")


async def build_prompt(request, tools) -> str:
    global tokenizer
    # pydantic格式转成dict， 否则，当根据tokenizer_config.json拼template时，Jinja判断无法识别
    messages = [m.model_dump(by_alias=True, exclude_none=True) for m in request.messages]
    kwargs = {"conversation": messages}
    if request.character_settings:
        kwargs["character_settings"] = request.character_settings
    if request.role_settings:
        kwargs["role_setting"] = request.role_settings

    if request.chat_template_kwargs:
        kwargs.update(request.chat_template_kwargs)

    try:
        input_str = tokenizer.apply_chat_template(**kwargs, tokenize=False, add_generation_prompt=True, tools=tools)
    except:
        #  This except branch will be triggered when the chosen model
        #  has a different tools input format that is not compatiable
        #  with openAI's apply_chat_template tool_call format, like Mistral.
        tools = [t if "function" in t else {"function": t} for t in tools]
        input_str = tokenizer.apply_chat_template(
            **kwargs,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
        )
    return input_str
