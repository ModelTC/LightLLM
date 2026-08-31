# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import torch

from lightllm.common.basemodel.hidden_collector import FinalHiddenCollector
from lightllm.common.basemodel.attention.linear import KDALinearAttBackend
from lightllm.common.build_utils import repair_config
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.common.req_manager import ReqManagerForMamba
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.deepseek3_2.model import Deepseek3_2TpPartModel
from lightllm.models.glm5_next.infer_struct import Glm5NextInferStateInfo
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import (
    Glm5NextTransformerLayerInfer,
)
from lightllm.models.glm5_next.layer_infer.pre_layer_infer import (
    Glm5NextMultimodalPreLayerInfer,
)
from lightllm.models.glm5_next.layer_weights.pre_and_post_layer_weight import (
    Glm5NextPreAndPostLayerWeight,
)
from lightllm.models.glm5_next.layer_weights.transformer_layer_weight import (
    Glm5NextTransformerLayerWeight,
)
from lightllm.models.glm5_next.mem_manager import Glm5NextMemManager
from lightllm.models.registry import ModelRegistry
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args


class Glm5NextPostNormHiddenCollector(FinalHiddenCollector):
    """Expose the post-final-norm hidden state expected by GLM NextN.

    The generic EAGLE collector returns the decoder output before the model's
    final RMSNorm.  GLM's NextN block was trained against the normalized target
    hidden and also recycles its own normalized hidden between recurrent draft
    steps (matching the vLLM and SGLang implementations).
    """

    def __init__(self, norm_weight, eps: float):
        super().__init__()
        self.norm_weight = norm_weight
        self.eps = eps
        self.draft_token_ids = None
        self.draft_token_probs = None

    def new_instance(self):
        return Glm5NextPostNormHiddenCollector(self.norm_weight, self.eps)

    def add_final_hidden(self, final_hidden: torch.Tensor) -> None:
        self.final_hidden = self.norm_weight(input=final_hidden, eps=self.eps)

    def add_mtp_outputs(
        self,
        draft_token_ids: torch.Tensor | None,
        confidence_logits: torch.Tensor | None,
        draft_token_probs: torch.Tensor | None = None,
    ) -> None:
        assert confidence_logits is None
        self.draft_token_ids = draft_token_ids
        self.draft_token_probs = draft_token_probs

    def finish_output(self, infer_state):
        output = super().finish_output(infer_state)
        output.draft_token_ids = self.draft_token_ids
        output.draft_token_probs = self.draft_token_probs
        self.draft_token_ids = None
        self.draft_token_probs = None
        return output


@ModelRegistry(["glm5_next", "glm5_next_text"])
class Glm5NextTpPartModel(Deepseek3_2TpPartModel):
    # Keep attention/mHC rows replicated so their row-parallel projections can
    # use the fast custom all-reduce.  Only shard rows around EP MoE, then
    # all-gather once.  The recurrent draft model opts out below.
    replicated_attention_ep = True
    pre_and_post_weight_class = Glm5NextPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer
    infer_state_class = Glm5NextInferStateInfo

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as config_file:
            outer_config = json.load(config_file)
        self.outer_config = outer_config
        self.config = dict(outer_config.get("text_config", outer_config))
        self.vision_config = outer_config.get("vision_config")
        if "quantization_config" in outer_config:
            self.config["quantization_config"] = outer_config["quantization_config"]
            # GLM-5.3's index cache follows the official ue8m0-scale path;
            # the released HF config does not spell this implementation detail
            # out in quantization_config.
            self.config["quantization_config"].setdefault("scale_fmt", "ue8m0")
        # The checkpoint uses the standard clamped SwiGLU, not GPT-OSS's
        # clamped (up + 1) variant supported by the shared kernel.
        self.config["swiglu_clamp_up_add_one"] = False
        # The generic autotune warmup executes only a representative prefix.
        # Let the last layer in that prefix contract mHC's residual streams
        # before the shared LM head consumes its output.
        self.config["autotune_layer_num"] = 4
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

    def autotune_layers(self):
        return 4

    def _init_hidden_collector(self):
        collector = self.mtp_manager.create_hidden_collector(model=self)
        if isinstance(collector, FinalHiddenCollector):
            collector = Glm5NextPostNormHiddenCollector(
                norm_weight=self.pre_post_weight.final_norm_weight_,
                eps=self.config["rms_norm_eps"],
            )
        self.hidden_collector_prototype = collector

    def _make_linear_config(self):
        linear = self.config["linear_attn_config"]
        start_args: StartArgs = get_env_start_args()
        state_dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        return LinearAttCacheConfig(
            tp_world_size=self.tp_world_size_,
            full_att_all_num_kv_heads=1,
            full_att_dtype=self.data_type,
            full_att_num_kv_heads=1,
            full_att_head_dim=self.config["kv_lora_rank"],
            global_linear_k_heads=linear["num_heads"],
            global_linear_v_heads=linear["num_heads"],
            num_linear_k_heads=linear["num_heads"] // self.tp_world_size_,
            num_linear_v_heads=linear["num_heads"] // self.tp_world_size_,
            head_linear_k_dim=linear["head_dim"],
            head_linear_v_dim=linear["head_dim"],
            conv_kernel_size=linear["short_conv_kernel_size"],
            linear_layer_num=len(linear["kda_layers"]),
            conv_state_dtype=self.data_type,
            ssm_state_dtype=state_dtypes[start_args.linear_att_ssm_data_type],
            full_attention_interval=4,
            all_layer_num=self.config["n_layer"],
            draft_full_att_kv_layer_num=get_added_mtp_kv_layer_num(),
        )

    def _init_req_manager(self):
        max_sequence_length = 0
        if self.batch_max_tokens is not None:
            max_sequence_length = max(max_sequence_length, self.batch_max_tokens)
        if self.max_seq_length is not None:
            max_sequence_length = max(max_sequence_length, self.max_seq_length)
        self.linear_config = self._make_linear_config()
        self.req_manager = ReqManagerForMamba(
            self.max_req_num,
            max_sequence_length,
            None,
            linear_config=self.linear_config,
        )

    def _init_mem_manager(self):
        self.linear_config = getattr(self, "linear_config", self._make_linear_config())
        self.mem_manager = Glm5NextMemManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=1,
            head_dim=self.config["kv_lora_rank"],
            full_att_layer_num=self.linear_config.get_full_att_kv_layer_num_with_draft_model(),
            linear_config=self.linear_config,
            mem_fraction=self.mem_fraction,
        )

    def _init_att_backend1(self):
        if getattr(self, "is_mtp_draft_model", False):
            self.prefill_att_backend1 = None
            self.decode_att_backend1 = None
            return
        self.prefill_att_backend1 = KDALinearAttBackend(model=self)
        self.decode_att_backend1 = self.prefill_att_backend1

    def _init_custom(self):
        # GLM-5 sparse MLA is entirely NoPE.  Keep zero-width tables so the
        # generic infer-state position setup remains valid without allocating
        # a million-token rotary cache.
        max_length = max(self.config["max_position_embeddings"], self.max_seq_length or 0)
        self._cos_cached = torch.empty((max_length, 0), dtype=self.data_type, device="cuda")
        self._sin_cached = torch.empty_like(self._cos_cached)
        dist_group_manager.new_deepep_group(
            n_routed_experts=self.config["n_routed_experts"],
            hidden_size=self.config["hidden_size"],
            expert_quant_method_names=dist_group_manager.get_moe_quant_methods(self.trans_layers_weight),
            num_experts_per_tok=self.config["num_experts_per_tok"],
            moe_intermediate_size=self.config["moe_intermediate_size"],
        )


@ModelRegistry(
    "glm5_next",
    is_multimodal=True,
    condition=lambda model_cfg: model_cfg.get("vision_config") is not None,
)
class Glm5NextMultimodalTpPartModel(Glm5NextTpPartModel):
    pre_layer_infer_class = Glm5NextMultimodalPreLayerInfer
