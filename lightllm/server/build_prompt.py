tokenizer = None


def init_tokenizer(args):
    global tokenizer
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)
    chat_path = args.chat_template
    if chat_path is not None:
        with open(chat_path, "r", encoding="utf-8") as f:
            chat_template_str = f.read()
        tokenizer.chat_template = chat_template_str


async def build_prompt(request, tools) -> str:
    global tokenizer
    import json

    # pydantic格式转成dict， 否则，当根据tokenizer_config.json拼template时，Jinja判断无法识别
    messages = [m.model_dump(by_alias=True, exclude_none=True) for m in request.messages]

    # Convert tool_calls function.arguments from JSON string to dict for Jinja template compatibility
    # Qwen's chat template expects arguments to be a dict (uses |items filter)
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
                            # Keep original string if not valid JSON
                            pass

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
        if tools is not None:
            tools = [t if "function" in t else {"function": t} for t in tools]
        input_str = tokenizer.apply_chat_template(
            **kwargs,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
        )
    return input_str
