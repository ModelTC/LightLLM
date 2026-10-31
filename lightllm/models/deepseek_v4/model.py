from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from lightllm.models.llama.model import LlamaTpPartModel
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
from lightllm.common.basemodel.attention import get_nsa_prefill_att_backend_class, get_nsa_decode_att_backend_class
from lightllm.common.basemodel.attention.nsa.dsv4_fp8_flashmla_sparse import (
    DSV4_NSA_BACKENDS,
    DeepseekV4FlashMlaFp8SparseAttBackend,
)
from lightllm.common.basemodel.hidden_collector import LayerHiddenCollector
from lightllm.common.basemodel.triton_kernel.mhc import hc_contract, hc_post
from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek_v4.workspace import DeepseekV4Workspace
from lightllm.models.llama.yarn_rotary_utils import (
    find_correction_range,
    linear_ramp_mask,
)
from lightllm.utils.envs_utils import get_env_start_args, get_redundancy_expert_num
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att.base_state import FullAttentionState
    from lightllm.common.attention_state.bounded_att.base_state import BoundedAttentionState

logger = init_logger(__name__)


class DeepseekV4LayerHiddenCollector(LayerHiddenCollector):
    """Materialize deferred mHC streams before collecting draft-model features."""

    def _load_layer_ids(self):
        layer_ids = self.model.config.get("dspark_target_layer_ids")
        if layer_ids is None:
            return super()._load_layer_ids()
        layer_ids = frozenset(map(int, layer_ids))
        assert layer_ids and all(0 <= layer < self.layer_num for layer in layer_ids)
        return layer_ids

    def add(self, layer_index: int, hidden):
        if layer_index not in self.layer_ids:
            return
        hc_mult = self.model.config["hc_mult"]
        if isinstance(hidden, tuple):
            hidden = hc_post(*hidden, streams=hc_mult)
        super().add(layer_index=layer_index, hidden=hc_contract(hidden, hc_mult))


class DeepseekV4TpPartModel(LlamaTpPartModel):
    is_hybrid_att_model = True
    has_vision = False

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
        if self.args.enable_dp_prefill_balance:
            raise ValueError(
                "DeepSeek-V4 does not support --enable_dp_prefill_balance: KV slots remain in request order"
            )
        if self.args.enable_ep_moe:
            if self.args.enable_tpsp_mix_mode and self.tp_world_size_ > 1:
                raise ValueError("DeepSeek-V4 EP with TP-SP is not supported: FFN and mHC token layouts differ")
            if get_redundancy_expert_num():
                raise ValueError("DeepSeek-V4 preselected MoE routing does not yet support redundant experts")
        return

    @classmethod
    def create_full_att_states(
        cls,
        model_config: dict[str, object],
        args: StartArgs | EasyDict,
        model_logical_layer_ids: tuple[int, ...],
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> list[FullAttentionState]:
        """分别创建 C4/C128 定义；逻辑索引保留原始 token 域，不提前分配存储。"""
        from lightllm.common.attention_state.full_att.create_compressed_att_state import create_compressed_att_state
        from .config import create_attention_config

        assert args.llm_kv_type in ("None", "fp8kv_dsa"), f"unsupported V4 KV encoding: {args.llm_kv_type}"
        config = create_attention_config(
            model_config=model_config,
            model_logical_layer_ids=model_logical_layer_ids,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        states = []
        for ratio in (4, 128):
            if ratio not in config.compress_ratios:
                continue
            states.append(create_compressed_att_state(
                config=config.create_compressed_config(compress_ratio=ratio),
                page_size=args.page_size,
                device="meta",
            ))
        return states

    @classmethod
    def create_bounded_att_states(
        cls,
        model_config: dict[str, object],
        args: StartArgs | EasyDict,
        model_logical_layer_ids: tuple[int, ...],
        request_capacity: int = 1,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        device: str = "meta",
    ) -> list[BoundedAttentionState]:
        """创建本模型的 SWA/compressor 请求运行态，大小页容量由管理器后续确定。"""
        from lightllm.common.attention_state.bounded_att import (
            WindowCompressorStateConfig,
            create_window_compressor_state,
            create_sliding_window_state,
        )
        from .config import create_attention_config

        config = create_attention_config(
            model_config=model_config,
            model_logical_layer_ids=model_logical_layer_ids,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        states = [create_sliding_window_state(
            config=config.create_swa_config(),
            request_capacity=request_capacity,
            mtp_step=args.mtp_step,
            device=device,
        )]
        # 模型决定窗口组合：CSA 使用 W8/S4，HCA 使用 W128/S128；State 本身不识别 C4/C128。
        for ratio, window_size in ((4, 8), (128, 128)):
            layers = tuple(
                layer for layer, compress_ratio in zip(model_logical_layer_ids, config.compress_ratios)
                if compress_ratio == ratio
            )
            if not layers:
                continue
            targets = (
                (("kv", config.head_dim), ("indexer", config.indexer_head_dim))
                if ratio == 4 else (("kv", config.head_dim),)
            )
            for target, head_dim in targets:
                states.append(create_window_compressor_state(
                    config=WindowCompressorStateConfig(
                        logical_layer_ids=layers,
                        window_size=window_size,
                        compress_stride=ratio,
                        candidate_dim=head_dim * (2 if window_size > ratio else 1),
                        target=target,
                        tp_rank=tp_rank,
                        tp_world_size=tp_world_size,
                    ),
                    request_capacity=request_capacity,
                    mtp_step=args.mtp_step,
                    device=device,
                ))
        return states

    def prepare_attention_states(self) -> None:
        super().prepare_attention_states()
        # 在统一容量规划前预留长 prefill 的 SWA 临时区，避免 KV 分配挤占计算所需空间。
        DeepseekV4FlashMlaFp8SparseAttBackend.get_swa_prefill_workspace(self)

    def _init_quant(self):
        if not self.is_mtp_draft_model:
            return super()._init_quant()
        from lightllm.common.quantization import Quantcfg

        # 权重量化按 checkpoint 层号查询；draft 的资源层数仍只描述自身。
        config = dict(self.config, n_layer=len(self.config["compress_ratios"]))
        self.quant_cfg = Quantcfg(config, self.quant_type, self.quant_cfg_path, self.expert_dtype)

    def _init_att_backend(self):
        args = get_env_start_args()
        if args.llm_kv_type == "None":
            args.llm_kv_type = "fp8kv_dsa"
        # TODO: 支持其他 kv type
        if args.llm_kv_type != "fp8kv_dsa":
            raise RuntimeError("DeepSeek-V4 requires llm_kv_type=fp8kv_dsa for packed FlashMLA sparse attention")
        self.prefill_att_backend = get_nsa_prefill_att_backend_class(index=0, backend_map=DSV4_NSA_BACKENDS)(model=self)
        self.decode_att_backend = get_nsa_decode_att_backend_class(index=0, backend_map=DSV4_NSA_BACKENDS)(model=self)

        tp_q_head_num = self.prefill_att_backend.tp_q_head_num
        padded_q_head_num = self.prefill_att_backend.padded_q_head_num
        if not self.is_mtp_draft_model:
            # Drafts share this buffer after the main model has captured its CUDA graphs.
            self.dsv4_workspace.init_flashmla_prefill_q(
                tp_q_head_num=tp_q_head_num,
                padded_q_head_num=padded_q_head_num,
                head_dim=self.config["head_dim"],
                dtype=self.data_type,
            )
        for layer_infer in self.layers_infer:
            layer_infer.flashmla_q_head_num_ = padded_q_head_num
        return

    def _init_custom(self):
        self._init_to_get_rotary()
        prefill_aux_stream = None
        if os.getenv("LIGHTLLM_DSV4_PREFILL_OVERLAP", "1") == "1" and not self.args.enable_prefill_microbatch_overlap:
            prefill_aux_stream = torch.cuda.Stream()
        self.dsv4_workspace = DeepseekV4Workspace(self)
        if prefill_aux_stream is not None and self.run_mode == "prefill" and 4 in self.config["compress_ratios"]:
            with torch.cuda.stream(prefill_aux_stream):
                self.dsv4_workspace.init_c4_prefill_aux(self)
        for layer in self.layers_infer:
            layer.dsv4_prefill_aux_stream = prefill_aux_stream
        dist_group_manager.new_deepep_group(
            n_routed_experts=self.config["n_routed_experts"],
            hidden_size=self.config["hidden_size"],
            expert_quant_method_names=dist_group_manager.get_moe_quant_methods(self.trans_layers_weight),
            num_experts_per_tok=self.config["num_experts_per_tok"],
            moe_intermediate_size=self.config["moe_intermediate_size"],
        )
        return

    def _init_hidden_collector(self):
        if not self.is_mtp_draft_model and self.mtp_manager.args.mtp_mode == "dspark":
            self.hidden_collector_prototype = DeepseekV4LayerHiddenCollector(self)
        else:
            super()._init_hidden_collector()
            if isinstance(self.hidden_collector_prototype, LayerHiddenCollector):
                self.hidden_collector_prototype = DeepseekV4LayerHiddenCollector(self)

    def _init_cudagraph(self):
        if self.args.run_mode == "prefill":
            self.graph = None
        else:
            super()._init_cudagraph()

    def _init_prefill_cuda_graph(self):
        if self.args.run_mode == "decode":
            self.prefill_graph = None
        else:
            super()._init_prefill_cuda_graph()

    def _autotune_warmup(self):
        if self.args.run_mode != "decode":
            super()._autotune_warmup()

    def _check_max_len_infer(self):
        if self.args.run_mode != "decode":
            super()._check_max_len_infer()

    def _init_to_get_rotary(self):
        # Interleaved (GPT-J) rope. Build complex64 freqs_cis tables (_freqs_cis_*) following the
        # gemma4 two-variant convention; the fused CUDA Q/K kernels consume them directly, while
        # _cos_cached_*/_sin_cached_* are .real/.imag views of the same storage for the inverse
        # rope and compressor paths.
        # Pure sliding-window layers use unscaled rope_theta. Compressed layers use
        # compress_rope_theta with YaRN for both local SWA and compressed attention.
        # Kept fp32 for accuracy (the apply upcasts anyway).
        cfg = self.config
        rs = cfg.get("rope_scaling", {}) or {}
        dim = cfg["qk_rope_head_dim"]
        # The rope tables MUST span every absolute position any request can produce (the served
        # max_req_total_len / max_position_embeddings, up to 1M). Capping them shorter makes
        # init_some_extra_state's index_select(cos/sin, position_ids) read OOB past the table at
        # contexts beyond the cap (device-side assert / crash). ~268MB total at 1M, fp32x32 x4 views.
        max_seq = max(self.max_seq_length, cfg["max_position_embeddings"])
        freq_exponents = torch.arange(0, dim, 2, dtype=torch.float32, device="cuda") / dim
        positions = torch.arange(max_seq, dtype=torch.float32, device="cuda")

        rope_type = rs.get("rope_type", rs.get("type", "default"))
        orig_max = rs.get("original_max_position_embeddings", 0)

        def build_inv_freq(base):
            freqs = 1.0 / (base ** freq_exponents)
            if rope_type == "yarn" and orig_max > 0:
                beta_fast = rs.get("beta_fast", 32)
                beta_slow = rs.get("beta_slow", 1)
                factor = rs.get("factor", 1)
                if factor is None:
                    factor = cfg["max_position_embeddings"] / orig_max
                low, high = find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
                smooth = 1 - linear_ramp_mask(low, high, dim // 2).cuda()
                freqs = freqs / factor * (1 - smooth) + freqs * smooth
            return freqs

        sliding_freqs = 1.0 / (cfg["rope_theta"] ** freq_exponents)
        f = torch.outer(positions, sliding_freqs)  # [max_seq, dim//2]
        self._freqs_cis_sliding = torch.complex(f.cos(), f.sin())

        compress_freqs = build_inv_freq(cfg["compress_rope_theta"])
        f = torch.outer(positions, compress_freqs)  # [max_seq, dim//2]
        self._freqs_cis_compress = torch.complex(f.cos(), f.sin())
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
            # the indexer-Q fused kernel (compress rope) needs the complex compress freqs table.
            layer.index_infer.freqs_cis = self._freqs_cis_compress
        return


class DeepseekV4VisionTpPartModel(DeepseekV4TpPartModel):
    has_vision = True
