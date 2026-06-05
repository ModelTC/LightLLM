import importlib.util
import os

import torch
from lightllm.models.registry import ModelRegistry
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
from lightllm.common.basemodel.attention.base_att import (
    BaseAttBackend,
    BasePrefillAttState,
    BaseDecodeAttState,
)
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
from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek3_2.model import DeepSeekChatTokenizerBase
from lightllm.models.llama.yarn_rotary_utils import (
    find_correction_range,
    linear_ramp_mask,
)
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager

logger = init_logger(__name__)


class DeepseekV4DirectSparseAttBackend(BaseAttBackend):
    """Lifecycle placeholder for V4 direct attention.

    V4 attention is currently driven inside the layer by `vllm_sparse_attn()`, not by the generic
    `infer_state.prefill_att_state.prefill_att()` / `decode_att()` backend selector.
    """

    def create_att_prefill_state(self, infer_state):
        return DeepseekV4DirectSparsePrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return DeepseekV4DirectSparseDecodeAttState(backend=self, infer_state=infer_state)


class DeepseekV4DirectSparsePrefillAttState(BasePrefillAttState):
    def init_state(self):
        return

    def prefill_att(self, *args, **kwargs):
        raise RuntimeError("DeepSeek-V4 attention is executed directly by vllm_sparse_attn() in layer_infer.")


class DeepseekV4DirectSparseDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        return

    def decode_att(self, *args, **kwargs):
        raise RuntimeError("DeepSeek-V4 attention is executed directly by vllm_sparse_attn() in layer_infer.")


@ModelRegistry("deepseek_v4")
class DeepseekV4TpPartModel(LlamaTpPartModel):
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

    def _init_some_value(self):
        super()._init_some_value()
        self.head_dim_ = self.config["head_dim"]
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
        )
        return

    def _get_compress_rates(self, layer_num):
        rates = list(self.config["compress_ratios"])
        assert (
            len(rates) >= layer_num
        ), f"DeepSeek-V4 compress_ratios length {len(rates)} is shorter than layer_num {layer_num}"
        return rates[:layer_num]

    def _init_mem_manager(self):
        layer_num = self.config["n_layer"] + get_added_mtp_kv_layer_num()
        compress_rates = getattr(self, "_dsv4_compress_rates", self._get_compress_rates(layer_num))
        self.mem_manager = DeepseekV4MemoryManager(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=1,
            head_dim=self.config["head_dim"],
            layer_num=layer_num,
            compress_rates=compress_rates,
            indexer_head_dim=self.config["index_head_dim"],
            max_request_num=self.max_req_num,
            sliding_window=self.config["sliding_window"],
            mem_fraction=self.mem_fraction,
        )
        assert isinstance(self.req_manager, DeepseekV4ReqManager)
        self.req_manager.bind_mem_manager(self.mem_manager)
        return

    def _init_att_backend(self):
        self.prefill_att_backend = DeepseekV4DirectSparseAttBackend(model=self)
        self.decode_att_backend = DeepseekV4DirectSparseAttBackend(model=self)
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
        # Interleaved (GPT-J) rope. Build real cos/sin tables (_cos_cached_*/_sin_cached_*) following the
        # gemma4 two-variant convention; the infer-struct slices them into position_cos_*/position_sin_*
        # and apply_rotary_emb (interleaved, NOT the NeoX rotary_emb_fwd) applies them. Sliding-window
        # layers use base rope_theta (no YaRN); compressed (CSA/HCA) layers use compress_rope_theta with
        # YaRN. Tables kept fp32 for accuracy (the apply upcasts anyway).
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
            return f.cos(), f.sin()

        self._cos_cached_sliding, self._sin_cached_sliding = build(
            cfg["rope_theta"],
            rs.get("factor", 16),
            rs.get("original_max_position_embeddings", 65536),
        )
        self._cos_cached_compress, self._sin_cached_compress = build(
            cfg["compress_rope_theta"],
            rs.get("factor", 16),
            rs.get("original_max_position_embeddings", 65536),
        )
        return


class DeepSeekV4Tokenizer(DeepSeekChatTokenizerBase):
    """Tokenizer wrapper for DeepSeek-V4's Python prompt encoding."""

    def __init__(self, tokenizer, model_dir):
        super().__init__(tokenizer)
        self.model_dir = model_dir
        self._encoding_module = None

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

    def _encode_messages(self, msgs, thinking_mode, kwargs):
        encoding = self._get_encoding_module()
        return encoding.encode_messages(
            msgs,
            thinking_mode=thinking_mode,
            drop_thinking=kwargs.get("drop_thinking", True),
            add_default_bos_token=kwargs.get("add_default_bos_token", True),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )
