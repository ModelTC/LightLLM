import os
import json
import torch
from lightllm.models.registry import ModelRegistry
from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend
from lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager import HybridSlidingMemoryManager
from lightllm.common.req_manager import ReqManagerForSlidingWindow
from lightllm.common.build_utils import repair_config
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.gemma4.infer_struct import Gemma4InferStateInfo
from lightllm.models.gemma4.kv_layout import build_sliding_cache_config
from lightllm.models.gemma4.layer_infer.pre_layer_infer import Gemma4PreLayerInfer
from lightllm.models.gemma4.layer_infer.post_layer_infer import Gemma4PostLayerInfer
from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer
from lightllm.models.gemma4.layer_weights.pre_and_post_layer_weight import Gemma4PreAndPostLayerWeight
from lightllm.models.gemma4.layer_weights.transformer_layer_weight import Gemma4TransformerLayerWeight
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager

logger = init_logger(__name__)


@ModelRegistry("gemma4", is_multimodal=True)
class Gemma4TpPartModel(LlamaTpPartModel):
    pre_and_post_weight_class = Gemma4PreAndPostLayerWeight
    transformer_weight_class = Gemma4TransformerLayerWeight

    pre_layer_infer_class = Gemma4PreLayerInfer
    transformer_layer_infer_class = Gemma4TransformerLayerInfer
    post_layer_infer_class = Gemma4PostLayerInfer

    infer_state_class = Gemma4InferStateInfo

    def __init__(self, kvargs):
        # head_dim_ is used by the default _init_to_get_rotary which we
        # override; still set it to the sliding-layer head_dim for consistency
        # with the mem manager and any generic helpers.
        self.head_dim_ = 256
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # The shipped checkpoint is a multimodal config wrapping a Gemma4TextConfig
        # under text_config; flatten it so downstream code sees text-model fields
        # at the top level (mirrors the gemma3 approach).
        if "text_config" in self.config:
            self.config = self.config["text_config"].copy()

        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        self._reset_num_key_value_heads()

        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

        if self.config.get("enable_moe_block", False):
            # LightLLM's MoE helpers use Qwen/DeepSeek-style field names.
            # Gemma-4 checkpoints expose equivalent values as top_k_experts
            # and moe_intermediate_size.
            self.config.setdefault("num_experts_per_tok", self.config["top_k_experts"])
            self.config.setdefault("norm_topk_prob", True)
            self.config.setdefault("scoring_func", "softmax")
        return

    def _verify_params(self):
        args = get_env_start_args()
        assert self.load_way == "HF", "Gemma-4 only supports HF format."
        self.sliding_cache_config = build_sliding_cache_config(self.config, self.tp_world_size_, self.data_type)
        if self.config.get("hidden_size_per_layer_input"):
            # PLE uses one static buffer, not independent microbatch storage.
            assert not (
                args.enable_prefill_microbatch_overlap or args.enable_decode_microbatch_overlap
            ), "Gemma-4 PLE does not support microbatch overlap yet"
        assert args.mtp_step == 0, "Gemma-4 hybrid sliding-window cache does not support MTP yet"
        if args.enable_cpu_cache:
            assert not args.disable_dynamic_prompt_cache, "Gemma-4 CPU cache requires GPU prefix cache"
        assert not args.disable_chunked_prefill, "Gemma-4 hybrid sliding-window cache requires chunked prefill"
        assert args.run_mode == "normal", "Gemma-4 hybrid sliding-window cache does not support PD mode yet"
        assert args.llm_kv_type == "None", "Gemma-4 hybrid sliding-window cache does not support quantized KV yet"
        assert not args.enable_dp_prompt_cache_fetch, "Gemma-4 sliding-window state does not support DP cache fetch yet"
        assert not args.diverse_mode, "Gemma-4 sliding-window state does not support diverse mode yet"
        return

    def _init_req_manager(self):
        self.req_manager = ReqManagerForSlidingWindow(
            max_request_num=self.max_req_num,
            max_sequence_length=max(self.batch_max_tokens, self.max_seq_length),
            mem_manager=None,
            sliding_config=self.sliding_cache_config,
        )

    def _init_mem_manager(self):
        self.mem_manager = HybridSlidingMemoryManager(
            size=self.max_total_token_num,
            sliding_config=self.sliding_cache_config,
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_att_backend(self):
        # Full attention uses the standard backend. Sliding attention uses
        # Gemma's local kernels for the compact page table and image mask.
        self.prefill_att_backend = TritonAttBackend(model=self)
        self.decode_att_backend = TritonAttBackend(model=self)

    def _init_custom(self):
        self._init_to_get_rotary_gemma4()
        if self.config.get("enable_moe_block", False):
            dist_group_manager.new_deepep_group(
                n_routed_experts=self.config["num_experts"],
                hidden_size=self.config["hidden_size"],
                expert_quant_method_names=dist_group_manager.get_moe_quant_methods(self.trans_layers_weight),
                num_experts_per_tok=self.config.get("num_experts_per_tok", self.config.get("top_k_experts", 1)),
                moe_intermediate_size=self.config.get("moe_intermediate_size", self.config.get("intermediate_size")),
            )
        self._init_ple_static_buffer()

    def _init_ple_static_buffer(self):
        ple_dim = self.config.get("hidden_size_per_layer_input") or 0
        if ple_dim <= 0:
            return
        args = get_env_start_args()
        max_tokens = max(
            int(self.batch_max_tokens or 0),
            int(self.graph_max_batch_size or 0),
            int(getattr(args, "prefill_cudagraph_max_handle_token", 0) or 0),
        )
        assert max_tokens > 0, "PLE static buffer needs a positive max-token bound"
        num_layers = self.config["num_hidden_layers"]
        buf = torch.zeros((max_tokens, num_layers, ple_dim), dtype=self.data_type, device="cuda")
        self.pre_infer.ple_static_buffer = buf
        for layer_infer in self.layers_infer:
            layer_infer.ple_static_buffer = buf
        logger.info(
            f"Allocated PLE static buffer: tokens={max_tokens}, layers={num_layers}, "
            f"ple_dim={ple_dim}, dtype={self.data_type}"
        )

    def _init_to_get_rotary_gemma4(self):
        # gemma4 当前不支持 dp prefill balance
        assert self.args.enable_dp_prefill_balance is False, "Gemma-4 does not support dp prefill balance"

        rope_params = self.config["rope_parameters"]

        # Cap the rotary table at something we can fit in memory — Gemma-4's
        # advertised max_position_embeddings is 262144 which would require
        # ~200MB per table in fp32. Rely on the server's max_seq_length instead.
        max_seq_len = max(self.max_seq_length + 1024, 16384)

        t = torch.arange(max_seq_len, dtype=torch.float32, device="cpu")

        # Sliding layers: default RoPE, theta=10000, full rotation over head_dim=256.
        sliding_params = rope_params["sliding_attention"]
        sliding_head_dim = self.config["head_dim"]
        sliding_theta = sliding_params["rope_theta"]
        sliding_partial = sliding_params.get("partial_rotary_factor", 1.0)
        sliding_rot_dim = int(sliding_head_dim * sliding_partial)
        inv_freq_sliding = 1.0 / (
            sliding_theta ** (torch.arange(0, sliding_rot_dim, 2, dtype=torch.float32) / sliding_rot_dim)
        )
        freqs_s = torch.outer(t, inv_freq_sliding)
        self._cos_cached_sliding = torch.cos(freqs_s).to(self.data_type).cuda()
        self._sin_cached_sliding = torch.sin(freqs_s).to(self.data_type).cuda()

        # Full-attention layers: proportional RoPE, theta=1_000_000,
        # partial_rotary_factor=0.25 over global_head_dim=512.
        # Proportional semantics (HF transformers):
        #   rope_angles = int(partial * head_dim // 2)   -> 64
        #   inv_freq[0:rope_angles] = 1 / base ** (arange(0, 2*rope_angles, 2) / head_dim)
        #   inv_freq[rope_angles:head_dim//2] = 0   (identity rotation for "no-pe" dims)
        full_params = rope_params["full_attention"]
        full_head_dim = self.config["global_head_dim"]
        full_theta = full_params["rope_theta"]
        full_partial = full_params.get("partial_rotary_factor", 1.0)
        rope_type = full_params.get("rope_type", "default")
        if rope_type == "proportional":
            rope_angles = int(full_partial * full_head_dim // 2)
            inv_freq_rot = 1.0 / (
                full_theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / full_head_dim)
            )
            nope_angles = full_head_dim // 2 - rope_angles
            if nope_angles > 0:
                inv_freq_full = torch.cat([inv_freq_rot, torch.zeros(nope_angles, dtype=torch.float32)])
            else:
                inv_freq_full = inv_freq_rot
        else:
            full_rot_dim = int(full_head_dim * full_partial)
            inv_freq_full = 1.0 / (full_theta ** (torch.arange(0, full_rot_dim, 2, dtype=torch.float32) / full_rot_dim))

        freqs_f = torch.outer(t, inv_freq_full)
        self._cos_cached_full = torch.cos(freqs_f).to(self.data_type).cuda()
        self._sin_cached_full = torch.sin(freqs_f).to(self.data_type).cuda()
        return
