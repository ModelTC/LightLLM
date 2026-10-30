import json
import os
from typing import TYPE_CHECKING

import torch
import triton

from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.attention.linear.kda import KDALinearAttBackend
from lightllm.common.basemodel.attention.nsa.glm5_next import Glm5NextSparseAttBackend
from lightllm.models.glm5_next.att_resource_config import Glm5NextResourceConfig
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from .layer_infer.pre_layer_infer import Glm5NextPreLayerInfer
from .layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from .layer_weights.pre_and_post_layer_weight import Glm5NextPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.att_resources.full_att import FullAttentionState
    from lightllm.common.att_resources.bounded_att import BoundedAttentionState
    from lightllm.common.att_resources.att_tail import AttentionTailState


class Glm5NextTpPartModel(TpPartBaseModel):
    is_hybrid_att_model = True
    pre_and_post_weight_class = Glm5NextPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    pre_layer_infer_class = Glm5NextPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    @classmethod
    def _create_bounded_att_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], request_capacity: int = 1,
        tp_rank: int = 0, tp_world_size: int = 1, device: str = "meta",
    ) -> list["BoundedAttentionState"]:
        """创建当前模型的 bounded 运行态；CPU 布局查询传入 meta 设备。"""
        from lightllm.common.att_resources.bounded_att import create_kda_state, ConvSSMStateConfig
        from lightllm.common.basemodel.basemodel_utils import get_resource_config

        config = get_resource_config(model_config, model_class=cls)
        assert not (args.mtp_step and args.enable_prefill_decode_mixed), "MTP linear attention does not support mixed prefill/decode"
        cfg = Glm5NextResourceConfig.from_model_config(config, args, include_draft=False)
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
            request_capacity=request_capacity, mtp_step=args.mtp_step, device=device,
        )
        return [state]

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json")) as f:
            outer_config = json.load(f)
        self.config = dict(outer_config.get("text_config", outer_config))
        if "quantization_config" in outer_config:
            self.config["quantization_config"] = dict(outer_config["quantization_config"])
        self.config["autotune_layer_num"] = 4
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])

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
    def _create_full_att_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], tp_rank: int = 0, tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """按当前模型的方案创建 full-attention 定义，不分配 KV 容量或注册管理器。"""
        from lightllm.common.basemodel.basemodel_utils import create_full_attention_definition

        state = create_full_attention_definition(
            model_class=cls, model_config=model_config, args=args,
            model_logical_layer_ids=model_logical_layer_ids, tp_rank=tp_rank, tp_world_size=tp_world_size,
        )
        return [] if state is None else [state]

    @classmethod
    def _create_att_tail_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], request_capacity: int = 1,
        tp_rank: int = 0, tp_world_size: int = 1, device: str = "meta",
    ) -> list["AttentionTailState"]:
        """按本模型的 full-attention 层创建 K-pool tail，不依赖其他创建接口的副作用。"""
        from lightllm.models.glm5_next.att_resource_config import create_model_att_tail_states

        return create_model_att_tail_states(
            model_class=cls, model_config=model_config, args=args,
            model_logical_layer_ids=model_logical_layer_ids, request_capacity=request_capacity,
            tp_rank=tp_rank, tp_world_size=tp_world_size, device=device,
        )

    @classmethod
    def _full_attention_options(cls, config, args):
        if args.llm_kv_type != "None":
            raise ValueError(f"GLM K-pool does not support llm_kv_type={args.llm_kv_type}")
        cfg = Glm5NextResourceConfig.from_model_config(config, args, include_draft=False)
        return dict(heads=1, dim=config["kv_lora_rank"], logical_layer_groups=tuple((logical_layer_id,) for logical_layer_id in cfg.get_full_att_logical_layer_ids()), packed=True)

    def _init_att_backend(self):
        self.prefill_att_backend = Glm5NextSparseAttBackend(model=self)
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
