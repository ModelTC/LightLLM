import json
import os

tokenizer = None
_model_type = None


def init_tokenizer(args):
    global tokenizer, _model_type
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)

    # Detect model type for specialized encoding (e.g. DeepSeek-V3.2)
    config_path = os.path.join(args.model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        _model_type = model_config.get("model_type", None)
        # Check architectures as fallback
        if _model_type is None:
            archs = model_config.get("architectures", [])
            if any("DeepseekV32" in a for a in archs):
                _model_type = "deepseek_v32"

    chat_path = args.chat_template
    if chat_path is not None:
        with open(chat_path, "r", encoding="utf-8") as f:
            chat_template_str = f.read()
        tokenizer.chat_template = chat_template_str


async def build_prompt(request, tools) -> str:
    global tokenizer, _model_type
    # pydantic格式转成dict， 否则，当根据tokenizer_config.json拼template时，Jinja判断无法识别
    messages = [m.model_dump(by_alias=True, exclude_none=True) for m in request.messages]

    # Use DeepSeek-V3.2 native encoding when applicable
    if _model_type == "deepseek_v32":
        return _build_prompt_dsv32(messages, tools, request)

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


def _build_prompt_dsv32(messages, tools, request):
    from lightllm.server.encoding_dsv32 import encode_messages

    # Inject tools into system message if present
    if tools is not None and len(tools) > 0:
        wrapped_tools = [t if "function" in t else {"function": t} for t in tools]
        if messages and messages[0].get("role") == "system":
            messages[0]["tools"] = wrapped_tools
        else:
            messages.insert(0, {"role": "system", "tools": wrapped_tools})

    # Determine thinking mode from request
    thinking = False
    if request.chat_template_kwargs:
        thinking = request.chat_template_kwargs.get("thinking", False) or request.chat_template_kwargs.get(
            "enable_thinking", False
        )

    thinking_mode = "thinking" if thinking else "chat"
    drop_thinking = messages[-1]["role"] == "user" if messages else True

    return encode_messages(messages, thinking_mode=thinking_mode, drop_thinking=drop_thinking)
