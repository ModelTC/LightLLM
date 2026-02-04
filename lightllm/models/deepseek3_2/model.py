import copy
import json
import logging
import os

from lightllm.models.registry import ModelRegistry
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.utils.envs_utils import get_env_start_args

_logger = logging.getLogger(__name__)

# When ENABLE_NSA is set, use the full V32 NSA (Native Sparse Attention) pipeline
# including the indexer, custom memory manager, and NSA-aware attention kernels.
# When not set, fall back to the DeepSeek V3 (Deepseek2) inference path while
# keeping V32-specific tokenizer/parser support intact.
_ENABLE_NSA = os.environ.get("ENABLE_NSA", "0").lower() in ("1", "true")

if _ENABLE_NSA:
    from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
    from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
    from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2InferStateInfo
    from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager, Deepseek3_2FP8KVMemoryManager


class DeepSeekV32Tokenizer:
    """Tokenizer wrapper for DeepSeek-V3.2 that uses the Python-based
    encoding_dsv32 module instead of Jinja chat templates.

    DeepSeek-V3.2's tokenizer_config.json does not ship with a Jinja chat
    template, so ``apply_chat_template`` would fail without either a manually
    supplied ``--chat_template`` file or this wrapper.  Activate it with
    ``--tokenizer_mode deepseek_v32``.
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


@ModelRegistry(["deepseek_v32"])
class Deepseek3_2TpPartModel(Deepseek2TpPartModel):
    # When ENABLE_NSA is set, override with V32-specific NSA classes.
    # Otherwise, inherit the V3/V2 classes from Deepseek2TpPartModel.
    if _ENABLE_NSA:
        transformer_weight_class = Deepseek3_2TransformerLayerWeight
        transformer_layer_infer_class = Deepseek3_2TransformerLayerInfer
        infer_state_class = Deepseek3_2InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        if _ENABLE_NSA:
            self.index_topk = self.config["index_topk"]
        else:
            _logger.info("ENABLE_NSA is not set, using DeepSeek V3 inference path (no NSA indexer).")
        return

    def _init_inferstate_cls(self):
        if _ENABLE_NSA:
            self.infer_state_class = Deepseek3_2InferStateInfo
        else:
            super()._init_inferstate_cls()

    def _init_mem_manager(self):
        if not _ENABLE_NSA:
            # Fall back to the standard V3/V2 memory manager (no indexer buffer).
            return super()._init_mem_manager()

        manager_class = Deepseek3_2MemoryManager
        if get_env_start_args().llm_kv_type == "fp8kv":
            manager_class = Deepseek3_2FP8KVMemoryManager

        # mtp 模式下需要在mem manger上扩展draft model使用的layer
        added_mtp_layer_num = 0
        if get_env_start_args().mtp_mode == "deepseekv3_eagle":
            added_mtp_layer_num += 1
        elif get_env_start_args().mtp_mode == "deepseekv3_vanilla":
            added_mtp_layer_num += get_env_start_args().mtp_step

        self.mem_manager = manager_class(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=1,
            head_dim=self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"],
            layer_num=self.config["num_hidden_layers"] + added_mtp_layer_num,
            mem_fraction=self.mem_fraction,
        )
        return
