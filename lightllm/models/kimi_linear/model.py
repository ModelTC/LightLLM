import json
import os

import torch

from lightllm.common.build_utils import repair_config
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.common.req_manager import ReqManagerForMamba
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.kimi_linear.infer_struct import KimiLinearInferStateInfo
from lightllm.models.kimi_linear.layer_infer.transformer_layer_infer import (
    KimiLinearTransformerLayerInfer,
)
from lightllm.models.kimi_linear.layer_weights.transformer_layer_weight import (
    KimiLinearTransformerLayerWeight,
)
from lightllm.models.kimi_linear.mem_manager import KimiLinearMemManager
from lightllm.models.registry import ModelRegistry
from lightllm.utils.envs_utils import get_env_start_args


@ModelRegistry(["kimi_linear", "kimi_k3"])
class KimiLinearTpPartModel(Deepseek2TpPartModel):
    transformer_weight_class = KimiLinearTransformerLayerWeight
    transformer_layer_infer_class = KimiLinearTransformerLayerInfer
    infer_state_class = KimiLinearInferStateInfo

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            config = json.load(json_file)
        self.is_kimi_k3 = config.get("model_type") == "kimi_k3"
        self.weight_prefix = ""
        if self.is_kimi_k3:
            config = config["text_config"].copy()
            self.weight_prefix = "language_model."
        self.config = config
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        self.config.setdefault("rope_scaling", None)
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        self.config["n_routed_experts"] = self.config["num_experts"]
        self.config["n_shared_experts"] = self.config["num_shared_experts"]
        self.config["num_experts_per_tok"] = self.config["num_experts_per_token"]
        self.config["norm_topk_prob"] = self.config["moe_renormalize"]
        self.config["n_group"] = self.config["num_expert_group"]
        self.config["scoring_func"] = self.config["moe_router_activation_func"]

    def _verify_params(self):
        super()._verify_params()
        start_args = get_env_start_args()
        assert start_args.mtp_step == 0, "Kimi Linear does not provide MTP layers"
        if getattr(start_args, "moe_ep_backend", "deepep") == "moonep":
            assert start_args.enable_ep_moe, "--moe_ep_backend moonep requires --enable_ep_moe"
            assert start_args.nnodes == 1, "MoonEP currently supports only a single NVLink-connected node"
            assert self.is_kimi_k3, "MoonEP is currently supported only for the released Kimi K3 architecture"
            assert self.data_type == torch.bfloat16, "MoonEP currently requires --data_type bfloat16"
            assert self.quant_type == "none", "MoonEP currently requires BF16 (non-quantized) expert weights"
            assert getattr(start_args, "moonep_prefetch_slots", 4) > 0, "--moonep_prefetch_slots must be positive"
            assert (
                start_args.ep_redundancy_expert_config_path is None and not start_args.auto_update_redundancy_expert
            ), "MoonEP dynamically duplicates experts and is incompatible with DeepEP redundancy settings"
            assert not (
                start_args.enable_prefill_microbatch_overlap
                or start_args.enable_decode_microbatch_overlap
                or start_args.enable_prefill_decode_mixed
            ), "MoonEP buffers cannot be shared by overlapping inference microbatches"

    def _init_linear_config(self):
        linear_config = self.config["linear_attn_config"]
        full_attention_layers = tuple(layer - 1 for layer in linear_config["full_attn_layers"])
        self.linear_config = LinearAttCacheConfig(
            tp_world_size=self.tp_world_size_,
            full_att_all_num_kv_heads=1,
            full_att_dtype=self.data_type,
            full_att_num_kv_heads=1,
            full_att_head_dim=self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"],
            global_linear_k_heads=linear_config["num_heads"],
            global_linear_v_heads=linear_config["num_heads"],
            num_linear_k_heads=linear_config["num_heads"] // self.tp_world_size_,
            num_linear_v_heads=linear_config["num_heads"] // self.tp_world_size_,
            head_linear_k_dim=linear_config["head_dim"],
            head_linear_v_dim=linear_config["head_dim"],
            conv_kernel_size=linear_config["short_conv_kernel_size"],
            linear_layer_num=len(linear_config["kda_layers"]),
            conv_state_dtype=self.data_type,
            ssm_state_dtype=torch.float32,
            full_attention_interval=1,
            all_layer_num=self.config["num_hidden_layers"],
            full_attention_layers=full_attention_layers,
            full_att_kv_factor=1,
        )

    def _init_req_manager(self):
        create_max_seq_len = max(value for value in (self.batch_max_tokens, self.max_seq_length) if value is not None)
        self._init_linear_config()
        self.req_manager = ReqManagerForMamba(
            self.max_req_num,
            create_max_seq_len,
            None,
            linear_config=self.linear_config,
        )

    def _init_mem_manager(self):
        self.mem_manager = KimiLinearMemManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=1,
            head_dim=self.linear_config.full_att_head_dim,
            full_att_layer_num=self.linear_config.get_full_att_kv_layer_num_with_draft_model(),
            linear_config=self.linear_config,
            mem_fraction=self.mem_fraction,
        )

    def _init_custom(self):
        dist_group_manager.new_deepep_group(
            self.config["num_experts"],
            self.config.get("routed_expert_hidden_size", self.config["hidden_size"]),
            self.config["num_experts_per_token"],
            self.config["moe_intermediate_size"],
        )
