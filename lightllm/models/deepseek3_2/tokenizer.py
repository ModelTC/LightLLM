import copy


class DeepSeekV32Tokenizer:
    """Tokenizer wrapper for DeepSeek-V3.2 that uses the Python-based
    encoding_dsv32 module instead of Jinja chat templates.

    DeepSeek-V3.2's tokenizer_config.json does not ship with a Jinja chat
    template, so ``apply_chat_template`` would fail without either a manually
    supplied ``--chat_template`` file or this wrapper.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Cache added vocabulary for performance (HuggingFace can be slow).
        self._added_vocab = None

    # ------------------------------------------------------------------
    # Attribute delegation – everything not overridden goes to the inner
    # tokenizer so that encode/decode/vocab_size/eos_token_id/… all work.
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def get_added_vocab(self):
        if self._added_vocab is None:
            self._added_vocab = self.tokenizer.get_added_vocab()
        return self._added_vocab

    # ------------------------------------------------------------------
    # Core override: route apply_chat_template through encode_messages.
    # ------------------------------------------------------------------
    def apply_chat_template(
        self,
        conversation=None,
        messages=None,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
        thinking=None,
        **kwargs,
    ):
        from lightllm.models.deepseek3_2.encoding_dsv32 import encode_messages, render_tools

        msgs = conversation if conversation is not None else messages
        if msgs is None:
            raise ValueError("Either 'conversation' or 'messages' must be provided")

        # Deep copy to avoid mutating the caller's messages.
        msgs = copy.deepcopy(msgs)

        # Determine thinking mode.
        thinking_mode = "thinking" if thinking else "chat"

        # Inject tools into the first system message (or create one) so that
        # encode_messages / render_message picks them up.
        if tools:
            # build_prompt passes tools as bare function dicts:
            #   [{"name": "f", "description": "...", "parameters": {...}}]
            # encoding_dsv32's render_message expects OpenAI wrapper format:
            #   [{"type": "function", "function": {...}}]
            wrapped_tools = []
            for t in tools:
                if "function" in t:
                    wrapped_tools.append(t)
                else:
                    wrapped_tools.append({"type": "function", "function": t})

            injected = False
            for msg in msgs:
                if msg.get("role") == "system":
                    existing = msg.get("tools") or []
                    msg["tools"] = existing + wrapped_tools
                    injected = True
                    break

            if not injected:
                # Prepend a system message that carries the tools.
                msgs.insert(0, {"role": "system", "content": "", "tools": wrapped_tools})

        prompt = encode_messages(
            msgs,
            thinking_mode=thinking_mode,
            drop_thinking=kwargs.get("drop_thinking", True),
            add_default_bos_token=kwargs.get("add_default_bos_token", True),
        )

        if tokenize:
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        return prompt
