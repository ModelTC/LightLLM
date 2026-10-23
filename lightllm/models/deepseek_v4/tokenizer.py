import copy
import importlib.util
import json
import os


class DeepSeekV4Tokenizer:
    """Tokenizer wrapper for DeepSeek-V4's Python prompt encoding."""

    # DeepSeek-V4 has a per-request thinking mode (<think>...</think>) toggled via
    # chat_template_kwargs={"thinking": true}. It has no Jinja chat_template string,
    # so advertise thinking support explicitly for tokenizer_supports_force_thinking().
    supports_thinking = True

    def __init__(self, tokenizer, model_dir, model_config=None):
        self.tokenizer = tokenizer
        self.model_dir = model_dir
        self.model_config = {} if model_config is None else model_config
        self.has_vision = self.model_config.get("vision_n_layers", 0) > 0
        self.image_token_id = tokenizer.convert_tokens_to_ids("<｜deepseek_image｜>") if self.has_vision else None
        self._encoding_module = None
        self._added_vocab = None

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    @property
    def xgrammar_tokenizer(self):
        return self.tokenizer

    def get_added_vocab(self):
        if self._added_vocab is None:
            self._added_vocab = self.tokenizer.get_added_vocab()
        return self._added_vocab

    def init_imageitem_extral_params(self, img, multi_params, sampling_params):
        return

    def get_image_token_length(self, img):
        from lightllm.models.deepseek_v4.image_processor import get_image_grid

        cfg = self.model_config
        _, _, _, _, token_num = get_image_grid(
            img.image_h,
            img.image_w,
            cfg["vision_patch_size"],
            cfg["vision_downsample_ratio"],
            cfg["vision_max_n_token"],
            cfg["vision_min_pixels"],
            cfg.get("vision_max_wh_ratio"),
        )
        return token_num

    def encode(self, prompt, multimodal_params=None, **kwargs):
        if isinstance(prompt, str):
            origin_ids = self.tokenizer.encode(prompt, **kwargs)
        elif isinstance(prompt, list):
            origin_ids = prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

        images = [] if multimodal_params is None else multimodal_params.images
        if not images:
            return origin_ids

        placeholder_count = sum(token_id == self.image_token_id for token_id in origin_ids)
        if placeholder_count != len(images):
            raise ValueError(f"invalid image placeholder count: {placeholder_count} vs {len(images)}")

        input_ids = []
        image_index = 0
        for token_id in origin_ids:
            if token_id != self.image_token_id:
                input_ids.append(token_id)
                continue

            image = images[image_index]
            cache_skip = len(input_ids) % 4
            compress_pad = 3 - cache_skip
            image.block_start_idx = len(input_ids)
            image.start_idx = len(input_ids) + compress_pad
            input_ids.extend(range(image.token_id + cache_skip, image.token_id + image.token_num))
            image.block_end_idx = len(input_ids)
            image_index += 1

        return input_ids

    def decode(self, token_ids, **kwargs):
        if self.has_vision:
            # Prompt image embeddings use out-of-vocabulary cache IDs that the HF tokenizer cannot decode.
            vocab_size = self.model_config["vocab_size"]
            token_ids = [
                self.image_token_id if int(token_id) >= vocab_size else int(token_id) for token_id in token_ids
            ]
        return self.tokenizer.decode(token_ids, **kwargs)

    def _get_encoding_module(self):
        if self._encoding_module is not None:
            return self._encoding_module

        # Prefer the encoder shipped inside the model dir (respects any model-specific
        # customization); fall back to the copy vendored in this repo, because some
        # DeepSeek-V4 releases (e.g. the FP8 weights) do NOT ship an encoding/ dir.
        # vLLM/sglang likewise vendor this encoder in-tree instead of depending on the
        # model directory.
        encoding_path = os.path.join(self.model_dir, "encoding", "encoding_dsv4.py")
        if not os.path.exists(encoding_path):
            encoding_path = os.path.join(os.path.dirname(__file__), "encoding", "encoding_dsv4.py")
        if not os.path.exists(encoding_path):
            raise FileNotFoundError(f"DeepSeek-V4 encoding file not found: {encoding_path}")

        spec = importlib.util.spec_from_file_location("lightllm_deepseek_v4_encoding_dsv4", encoding_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load DeepSeek-V4 encoding module from {encoding_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._encoding_module = module
        return module

    def apply_chat_template(
        self,
        conversation=None,
        messages=None,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
        thinking=None,
        enable_thinking=None,
        **kwargs,
    ):
        msgs = conversation if conversation is not None else messages
        if msgs is None:
            raise ValueError("Either 'conversation' or 'messages' must be provided")

        msgs = copy.deepcopy(msgs)

        # The model's DSML encoder (encode_arguments_to_dsml in encoding_dsv4.py) expects
        # function.arguments as a JSON string and parses it internally. Upstream,
        # build_prompt._normalize_tool_call_arguments converts arguments from the OpenAI
        # JSON string to a dict (needed by Qwen3.x-style Jinja templates). A dict hits the
        # encoder's except-branch and gets wrapped under a single name="arguments" param,
        # which the model then imitates and amplifies across turns until required fields go
        # missing. Re-serialize dicts back to JSON strings so the encoder emits one
        # <parameter> per real arg.
        for msg in msgs:
            content = msg.get("content")
            if isinstance(content, list) and all(
                isinstance(part, dict) and part.get("type") == "text" for part in content
            ):
                msg["content"] = "".join(part.get("text") or "" for part in content)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), dict):
                    fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)

        if tools:
            wrapped_tools = []
            for tool in tools:
                if "function" in tool:
                    wrapped_tools.append(tool)
                else:
                    wrapped_tools.append({"type": "function", "function": tool})

            injected = False
            for msg in msgs:
                if msg.get("role") == "system":
                    existing = msg.get("tools") or []
                    msg["tools"] = existing + wrapped_tools
                    injected = True
                    break

            if not injected:
                msgs.insert(0, {"role": "system", "content": "", "tools": wrapped_tools})

        if thinking is None:
            thinking = bool(enable_thinking) if enable_thinking is not None else True
        thinking_mode = "thinking" if thinking else "chat"
        effort = kwargs.get("reasoning_effort")
        if thinking and effort is None:
            effort = os.getenv("LIGHTLLM_DSV4_THINKING_EFFORT", "high")
        if effort not in ("low", "max", "high", None):
            effort = None
        encoding = self._get_encoding_module()
        prompt = encoding.encode_messages(
            msgs,
            thinking_mode=thinking_mode,
            drop_thinking=kwargs.get("drop_thinking", True),
            add_default_bos_token=kwargs.get("add_default_bos_token", True),
            reasoning_effort=effort,
        )

        if tokenize:
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        return prompt
