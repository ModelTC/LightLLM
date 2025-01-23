tokenizer = None


def init_tokenizer(args):
    global tokenizer
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)


async def build_prompt(request) -> str:
    global tokenizer
    messages = request.messages
    kwargs = {"conversation": messages}
    if request.character_settings:
        kwargs["character_settings"] = request.character_settings
    if request.role_settings:
        kwargs["role_setting"] = request.role_settings
    input_str = tokenizer.apply_chat_template(**kwargs, tokenize=False, add_generation_prompt=True)
    return input_str
