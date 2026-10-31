import json
import os
from typing import TYPE_CHECKING

import torch
import triton

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.attention.linear.kda import KDALinearAttBackend
from lightllm.common.basemodel.attention.nsa.glm5_next import Glm5NextSparseAttBackend
from lightllm.models.glm5_next.config import Glm5NextResourceConfig
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from .layer_infer.pre_layer_infer import Glm5NextPreLayerInfer
from .layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from .layer_weights.pre_and_post_layer_weight import Glm5NextPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState
    from lightllm.common.attention_state.bounded_att import BoundedAttentionState
    from lightllm.common.attention_state.att_tail import AttentionTailState


class Glm5NextTpPartModel(TpPartBaseModel):
    is_hybrid_att_model = True
    pre_and_post_weight_class = Glm5NextPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    pre_layer_infer_class = Glm5NextPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    @classmethod
    def create_bounded_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        model_logical_layer_ids: tuple[int, ...],
        request_capacity: int = 1,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        device: str = "meta",
    ) -> list["BoundedAttentionState"]:
        """创建当前模型的 bounded 运行态；CPU 布局查询传入 meta 设备。"""
        from lightllm.common.attention_state.bounded_att import create_kda_state, ConvSSMStateConfig

        config = cls.normalize_model_config(model_config)
        assert not (args.mtp_step and args.enable_prefill_decode_mixed), "MTP linear attention does not support mixed prefill/decode"
        cfg = Glm5NextResourceConfig.from_model_config(config, args)
        logical_layer_ids = tuple(layer for i, layer in enumerate(model_logical_layer_ids) if (i + 1) % cfg.full_attention_interval)
        if not logical_layer_ids:
            return []
        state = create_kda_state(
            config=ConvSSMStateConfig(
                logical_layer_ids=logical_layer_ids,
                global_k_heads=cfg.global_linear_k_heads,
                global_v_heads=cfg.global_linear_v_heads,
                key_dim=cfg.head_linear_k_dim,
                value_dim=cfg.head_linear_v_dim,
                conv_width=cfg.conv_kernel_size - 1,
                conv_dtype=cfg.conv_state_dtype,
                ssm_dtype=cfg.ssm_state_dtype,
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            request_capacity=request_capacity,
            mtp_step=args.mtp_step,
            device=device,
        )
        return [state]

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        """保留外层量化配置，并统一 GLM 文本配置的别名及 autotune 层数。"""
        config = super().normalize_model_config(model_config)
        if "quantization_config" in model_config:
            config["quantization_config"] = dict(model_config["quantization_config"])
        config["autotune_layer_num"] = 4
        return config

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json")) as f:
            self.config = self.normalize_model_config(json.load(f))

    def _verify_params(self):
        super()._verify_params()
        assert self.config["qk_rope_head_dim"] == 0, "GLM-5.3 Flash uses NoPE attention"
        assert not self.args.enable_tpsp_mix_mode, "GLM-5.3 Flash does not support TP/SP mixed mode"

    def autotune_layers(self):
        return 4

    def _init_some_value(self):
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        # MLA stores one replicated latent KV vector per token on every TP rank.
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 0
        self.qk_nope_head_dim = self.config["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.config["qk_rope_head_dim"]
        self.q_lora_rank = self.config["q_lora_rank"]
        self.kv_lora_rank = self.config["kv_lora_rank"]
        self.v_head_dim = self.config.get("v_head_dim", self.qk_nope_head_dim)
        self.head_dim_ = self.kv_lora_rank + self.qk_rope_head_dim

    @classmethod
    def create_full_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        model_logical_layer_ids: tuple[int, ...],
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """GLM full-attention 层采用 NoPE packed MLA，并保存压缩 indexer 字段。"""
        from lightllm.common.attention_state.full_att import FullAttentionConfig, create_mla_state
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        cfg = Glm5NextResourceConfig.from_model_config(config, args)
        logical_layer_groups = tuple((model_logical_layer_ids[index],) for index in cfg.get_full_att_layer_indices())
        if not logical_layer_groups:
            return []
        state = create_mla_state(
            config=FullAttentionConfig(
                logical_layer_groups=logical_layer_groups,
                global_head_num=1,
                head_dim=cfg.full_att_head_dim,
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            kv_type=args.llm_kv_type,
            packed=True,
            fp8_format="group64_e8m0",
            page_size=args.page_size,
            device="meta",
        )
        return [state]

    @classmethod
    def create_att_tail_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        model_logical_layer_ids: tuple[int, ...],
        request_capacity: int = 1,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        device: str = "meta",
    ) -> list["AttentionTailState"]:
        """直接选择本模型需要 K-pool tail 的层，不依赖 full State 创建过程。"""
        from lightllm.common.attention_state.att_tail import create_kpool_state, KPoolStateConfig
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        cfg = Glm5NextResourceConfig.from_model_config(config, args)
        logical_layer_ids = tuple(model_logical_layer_ids[index] for index in cfg.get_full_att_layer_indices())
        if not logical_layer_ids:
            return []
        assert args.linear_att_hash_page_size % config["index_kpool"] == 0, "GLM cache pages must align to index_kpool"
        state = create_kpool_state(
            config=KPoolStateConfig(
                logical_layer_ids=logical_layer_ids,
                pool_size=config["index_kpool"],
                head_dim=config["index_head_dim"],
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            request_capacity=request_capacity,
            mtp_step=args.mtp_step,
            device=device,
        )
        return [state]

    def _init_att_backend(self):
        from lightllm.common.basemodel.attention.nsa.fp8_glm5_next import Fp8Glm5NextSparseAttBackend

        backend_class = Fp8Glm5NextSparseAttBackend if self.args.llm_kv_type == "fp8kv_dsa" else Glm5NextSparseAttBackend
        self.prefill_att_backend = backend_class(model=self)
        self.decode_att_backend = self.prefill_att_backend

    def _init_att_backend1(self):
        self.prefill_att_backend1 = KDALinearAttBackend(model=self)
        self.decode_att_backend1 = self.prefill_att_backend1

    def _init_custom(self):
        triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
        if self.args.enable_ep_moe:
            dist_group_manager.new_deepep_group(
                n_routed_experts=self.config["n_routed_experts"],
                hidden_size=self.config["hidden_size"],
                expert_quant_method_names=dist_group_manager.get_moe_quant_methods(self.trans_layers_weight),
                num_experts_per_tok=self.config["num_experts_per_tok"],
                moe_intermediate_size=self.config["moe_intermediate_size"],
            )
