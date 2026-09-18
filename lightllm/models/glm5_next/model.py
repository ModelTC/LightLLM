import json
import os

import torch
import triton

from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.attention.linear.kda import KDALinearAttBackend
from lightllm.common.basemodel.attention.nsa.glm5_next import Glm5NextSparseAttBackend
from lightllm.common.kv_cache_mem_manager import Glm5NextMemManager
from lightllm.common.req_manager import Glm5NextReqManager
from lightllm.common.state_cache_manager import Glm5NextCacheConfig
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.registry import ModelRegistry
from .layer_infer.pre_layer_infer import Glm5NextPreLayerInfer
from .layer_infer.post_layer_infer import Glm5NextPostLayerInfer
from .layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from .layer_weights.pre_and_post_layer_weight import Glm5NextPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight


@ModelRegistry("glm5_next", is_multimodal=True)
@ModelRegistry("glm5_next_text")
class Glm5NextTpPartModel(TpPartBaseModel):
    pre_and_post_weight_class = Glm5NextPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    pre_layer_infer_class = Glm5NextPreLayerInfer
    post_layer_infer_class = Glm5NextPostLayerInfer
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json")) as f:
            outer_config = json.load(f)
        self.config = dict(outer_config.get("text_config", outer_config))
        if "quantization_config" in outer_config:
            self.config["quantization_config"] = dict(outer_config["quantization_config"])
        if "quantization_config" in self.config:
            self.config["quantization_config"].setdefault("scale_fmt", "ue8m0")
        self.config["autotune_layer_num"] = 4
        for names in (
            ["num_attention_heads", "n_head"],
            ["hidden_size", "n_embd", "n_embed"],
            ["num_hidden_layers", "n_layer"],
        ):
            repair_config(self.config, same_names=names)

    def _verify_params(self):
        assert self.load_way in ("HF", "DS"), "GLM-5.3 Flash only supports HF and DS weight loading"
        assert self.config["linear_attn_config"]["num_heads"] % self.tp_world_size_ == 0
        assert self.config["qk_rope_head_dim"] == 0, "GLM-5.3 Flash uses NoPE attention"
        args = self.args
        assert not args.enable_tpsp_mix_mode, "GLM-5.3 Flash does not support TP/SP mixed mode"
        assert args.dp == 1 or args.enable_ep_moe, "GLM-5.3 Flash data parallelism requires expert parallelism"
        assert args.mtp_mode in (None, "eagle_with_att", "vanilla_with_att"), "Unsupported GLM NextN mode"
        assert not args.enable_prefill_cudagraph, "GLM-5.3 Flash v1 supports decode CUDA graphs"

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

    def _init_req_manager(self):
        self.linear_config = Glm5NextCacheConfig.from_model_config(self.config, self.args)
        self.req_manager = Glm5NextReqManager(
            self.max_req_num,
            max(self.batch_max_tokens or 0, self.max_seq_length or 0),
            None,
            linear_config=self.linear_config,
        )

    def _init_mem_manager(self):
        self.mem_manager = Glm5NextMemManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=1,
            head_dim=self.linear_config.full_att_head_dim,
            full_att_layer_num=self.linear_config.get_full_att_kv_layer_num_with_draft_model(),
            linear_config=self.linear_config,
            mem_fraction=self.mem_fraction,
        )

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
