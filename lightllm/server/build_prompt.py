tokenizer = None


def init_tokenizer(args):
    global tokenizer
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)


async def build_prompt(request, tools) -> str:
    global tokenizer
    messages = request.messages
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
