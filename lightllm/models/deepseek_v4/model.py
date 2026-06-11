import copy
import importlib.util
import os

import torch
from lightllm.models.registry import ModelRegistry
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
from lightllm.models.deepseek_v4.layer_weights.pre_and_post_layer_weight import (
    DeepseekV4PreAndPostLayerWeight,
)
from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import (
    DeepseekV4TransformerLayerWeight,
)
from lightllm.models.deepseek_v4.layer_infer.pre_layer_infer import (
    DeepseekV4PreLayerInfer,
)
from lightllm.models.deepseek_v4.layer_infer.post_layer_infer import (
    DeepseekV4PostLayerInfer,
)
from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import (
    DeepseekV4TransformerLayerInfer,
)
from lightllm.common.basemodel.attention.create_utils import nsa_data_type_to_backend
from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.llama.yarn_rotary_utils import (
    find_correction_range,
    linear_ramp_mask,
)
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager

logger = init_logger(__name__)
DSV4_DECODE_CUDAGRAPH_MAX_LEN = 8192


@ModelRegistry("deepseek_v4")
class DeepseekV4TpPartModel(LlamaTpPartModel):
    req_manager: DeepseekV4ReqManager
    mem_manager: DeepseekV4MemoryManager

    pre_and_post_weight_class = DeepseekV4PreAndPostLayerWeight
    transformer_weight_class = DeepseekV4TransformerLayerWeight

    pre_layer_infer_class = DeepseekV4PreLayerInfer
    post_layer_infer_class = DeepseekV4PostLayerInfer
    transformer_layer_infer_class = DeepseekV4TransformerLayerInfer

    infer_state_class = DeepseekV4InferStateInfo

    def _verify_params(self):
        assert self.load_way == "HF", "only support HF format weights"
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        assert self.config["o_groups"] % self.tp_world_size_ == 0
        assert self.config["index_n_heads"] % self.tp_world_size_ == 0
        return

    def _init_req_manager(self):
        create_max_seq_len = 0
        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self._dsv4_req_manager_seq_len = create_max_seq_len
        layer_num = self.config["n_layer"] + get_added_mtp_kv_layer_num()
        self._dsv4_compress_rates = self._get_compress_rates(layer_num)
        self.req_manager = DeepseekV4ReqManager(
            self.max_req_num,
            create_max_seq_len,
            compress_rates=self._dsv4_compress_rates,
            head_dim=self.config["head_dim"],
            indexer_head_dim=self.config["index_head_dim"],
            sliding_window=self.config["sliding_window"],
        )
        return

    def _get_compress_rates(self, layer_num):
        rates = list(self.config["compress_ratios"])
        return rates[:layer_num]

    def _init_mem_manager(self):
        layer_num = self.config["n_layer"] + get_added_mtp_kv_layer_num()
        compress_rates = getattr(self, "_dsv4_compress_rates", self._get_compress_rates(layer_num))
        sliding_window = int(self.config["sliding_window"])
        # 活跃窗口之外的 swa 余量: 在途 prefill chunk 的瞬时占用(出窗槽位到下一次 prep 才回收)
        # + radix cache 持有的窗口尾部(每条缓存序列约一个 window)。
        swa_extra_token_num = int(self.batch_max_tokens or 0) + self.max_req_num * sliding_window
        self.mem_manager = DeepseekV4MemoryManager(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=1,
            head_dim=self.config["head_dim"],
            layer_num=layer_num,
            compress_rates=compress_rates,
            indexer_head_dim=self.config["index_head_dim"],
            max_request_num=self.max_req_num,
            sliding_window=sliding_window,
            swa_extra_token_num=swa_extra_token_num,
            mem_fraction=self.mem_fraction,
        )
        assert isinstance(self.req_manager, DeepseekV4ReqManager)
        self.req_manager.bind_mem_manager(self.mem_manager)
        return

    def _init_cudagraph(self):
        if not self.disable_cudagraph and self.graph_max_len_in_batch > DSV4_DECODE_CUDAGRAPH_MAX_LEN:
            logger.info(
                "DeepSeek-V4 caps decode cudagraph max_len_in_batch from %s to %s for the current "
                "graph-safe sparse-attention path; longer decode batches run eager.",
                self.graph_max_len_in_batch,
                DSV4_DECODE_CUDAGRAPH_MAX_LEN,
            )
            self.graph_max_len_in_batch = DSV4_DECODE_CUDAGRAPH_MAX_LEN
        return super()._init_cudagraph()

    def _can_run_prefill_cudagraph(self, infer_state: DeepseekV4InferStateInfo, handle_token_num):
        if infer_state.prefix_total_token_num == 0:
            return True
        return False

    def _init_att_backend(self):
        args = get_env_start_args()
        if args.llm_kv_type == "None":
            args.llm_kv_type = "fp8kv_dsa"
        if args.llm_kv_type != "fp8kv_dsa":
            raise RuntimeError("DeepSeek-V4 requires llm_kv_type=fp8kv_dsa for packed FlashMLA sparse attention")
        backend_cls = nsa_data_type_to_backend["fp8kv_dsa"]["flashmla_sparse"]
        self.prefill_att_backend = backend_cls(model=self)
        self.decode_att_backend = backend_cls(model=self)
        return

    def _init_custom(self):
        self._init_to_get_rotary()
        dist_group_manager.new_deepep_group(
            self.config["n_routed_experts"],
            self.config["hidden_size"],
            self.config.get("num_experts_per_tok", 1),
            self.config.get("moe_intermediate_size", self.config.get("intermediate_size")),
        )
        return

    def _init_to_get_rotary(self):
        # Interleaved (GPT-J) rope. Build complex64 freqs_cis tables (_freqs_cis_*) following the
        # gemma4 two-variant convention; the fused sglang q kernel consumes them directly, while
        # _cos_cached_*/_sin_cached_* are .real/.imag views of the same storage for the kv rope,
        # inverse rope and compressor paths (apply_rotary_emb: interleaved, NOT the NeoX
        # rotary_emb_fwd). Sliding-window layers use base rope_theta (no YaRN); compressed (CSA/HCA)
        # layers use compress_rope_theta with YaRN. Kept fp32 for accuracy (the apply upcasts anyway).
        cfg = self.config
        rs = cfg.get("rope_scaling", {}) or {}
        dim = cfg["qk_rope_head_dim"]
        beta_fast = rs.get("beta_fast", 32)
        beta_slow = rs.get("beta_slow", 1)
        max_seq = max(int(self.max_seq_length), int(cfg.get("max_position_embeddings", 8192)))
        max_seq = min(max_seq, 1 << 18)  # cap table size (256K) for correctness-first

        def build(base, factor, orig_max):
            freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cuda") / dim))
            if orig_max > 0:
                low, high = find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
                smooth = 1 - linear_ramp_mask(low, high, dim // 2).cuda()
                freqs = freqs / factor * (1 - smooth) + freqs * smooth
            f = torch.outer(torch.arange(max_seq, dtype=torch.float32, device="cuda"), freqs)  # [max_seq, dim//2]
            return torch.complex(f.cos(), f.sin())

        self._freqs_cis_sliding = build(
            cfg["rope_theta"],
            rs.get("factor", 16),
            rs.get("original_max_position_embeddings", 65536),
        )
        self._freqs_cis_compress = build(
            cfg["compress_rope_theta"],
            rs.get("factor", 16),
            rs.get("original_max_position_embeddings", 65536),
        )
        self._cos_cached_sliding = self._freqs_cis_sliding.real
        self._sin_cached_sliding = self._freqs_cis_sliding.imag
        self._cos_cached_compress = self._freqs_cis_compress.real
        self._sin_cached_compress = self._freqs_cis_compress.imag
        # Each layer uses exactly one rope variant; wire its table once here (layers are already
        # built: _init_infer_layer runs before _init_custom) instead of relaying via infer_state.
        # The compressor needs the full compress tables (entry rope positions != token positions).
        for layer in self.layers_infer:
            layer.freqs_cis = self._freqs_cis_compress if layer.compress_ratio else self._freqs_cis_sliding
            layer.cos_compress_table = self._cos_cached_compress
            layer.sin_compress_table = self._sin_cached_compress
        return


class DeepSeekV4Tokenizer:
    """Tokenizer wrapper for DeepSeek-V4's Python prompt encoding."""

    def __init__(self, tokenizer, model_dir):
        self.tokenizer = tokenizer
        self.model_dir = model_dir
        self._encoding_module = None
        self._added_vocab = None

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def get_added_vocab(self):
        if self._added_vocab is None:
            self._added_vocab = self.tokenizer.get_added_vocab()
        return self._added_vocab

    def _get_encoding_module(self):
        if self._encoding_module is not None:
            return self._encoding_module

        encoding_path = os.path.join(self.model_dir, "encoding", "encoding_dsv4.py")
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
            thinking = bool(enable_thinking) if enable_thinking is not None else False
        thinking_mode = "thinking" if thinking else "chat"
        encoding = self._get_encoding_module()
        prompt = encoding.encode_messages(
            msgs,
            thinking_mode=thinking_mode,
            drop_thinking=kwargs.get("drop_thinking", True),
            add_default_bos_token=kwargs.get("add_default_bos_token", True),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )

        if tokenize:
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        return prompt
