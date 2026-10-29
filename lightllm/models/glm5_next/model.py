import json
import os

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


class Glm5NextTpPartModel(TpPartBaseModel):
    is_hybrid_att_model = True
    pre_and_post_weight_class = Glm5NextPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    pre_layer_infer_class = Glm5NextPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    @classmethod
    def create_bounded_attention_definition(cls, model_config, args, *, model_layer_ids, tp_rank=0, tp_world_size=1):
        from lightllm.common.att_resources.bounded_att import create_kda_state, ConvSSMStateConfig
        from lightllm.common.basemodel.basemodel_utils import get_resource_config

        config = get_resource_config(model_config, model_class=cls)
        cfg = Glm5NextResourceConfig.from_model_config(config, args, include_draft=False)
        layers = tuple(layer for i, layer in enumerate(model_layer_ids) if (i + 1) % cfg.full_attention_interval)
        if not layers:
            return None
        state = create_kda_state(
            config=ConvSSMStateConfig(
                layer_ids=layers,
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
            request_capacity=1, device="meta",
        )
        return state

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

    def _init_att_resource_manager(self):
        from .att_resource_manager import Glm5NextAttResourceManager, init_model_att_tail
        from lightllm.common.basemodel.basemodel_utils import init_model_full_attention, init_model_bounded_attention

        self.linear_config = Glm5NextResourceConfig.from_model_config(self.config, self.args, include_draft=False)
        if self.args.linear_att_hash_page_size % self.linear_config.index_kpool:
            raise ValueError("GLM cache pages must align to index_kpool")
        if self.args.mtp_step and self.args.enable_prefill_decode_mixed:
            raise ValueError("MTP linear attention does not support mixed prefill/decode")
        if not self.is_mtp_draft_model:
            self.att_resource_manager = Glm5NextAttResourceManager(
                self.linear_config, self.max_req_num + 1, self.args.mtp_step,
                tp_rank=self.tp_rank_, linear_layer_ids=(), full_layer_ids=(),
            )
        init_model_bounded_attention(self)
        init_model_att_tail(self)
        init_model_full_attention(self)

    @classmethod
    def _full_attention_options(cls, config, args):
        if args.llm_kv_type != "None":
            raise ValueError(f"GLM K-pool does not support llm_kv_type={args.llm_kv_type}")
        cfg = Glm5NextResourceConfig.from_model_config(config, args, include_draft=False)
        return dict(heads=1, dim=config["kv_lora_rank"], layer_ids=cfg.get_full_att_layer_ids(), packed=True)

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
