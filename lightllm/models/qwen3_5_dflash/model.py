from lightllm.common.speculative.config import normalize_speculative_draft_config
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_5_dflash.layer_weights.pre_and_post_layer_weight import (
    Qwen35DFlashPreAndPostLayerWeight,
)


class Qwen3_5DFlashModel(Qwen3DFlashModel):
    """Qwen3.5 DFlash draft model.

    The checkpoint stores DFlash parameters under `dflash_config` and omits
    token embedding / LM head weights. Those two weights are shared from the
    already-loaded target model. Its query/proposal layout is resolved from the
    checkpoint schema by the generic speculative runtime.
    """

    pre_and_post_weight_class = Qwen35DFlashPreAndPostLayerWeight
    share_target_embedding_and_lm_head = True

    def _init_config(self):
        super()._init_config()
        self._normalize_dflash_config()
        return

    def _normalize_dflash_config(self):
        normalize_speculative_draft_config(self.config)

        rope_parameters = self.config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            if "rope_theta" in rope_parameters and "rope_theta" not in self.config:
                self.config["rope_theta"] = rope_parameters["rope_theta"]
            if (
                "partial_rotary_factor" in rope_parameters
                and "partial_rotary_factor" not in self.config
            ):
                self.config["partial_rotary_factor"] = rope_parameters[
                    "partial_rotary_factor"
                ]
            if "rope_scaling" not in self.config:
                self.config["rope_scaling"] = rope_parameters
        return

    def _init_custom(self):
        # The Qwen3.5 target uses mrope/partial rotary with a different head
        # shape from the ordinary DFlash draft. Build draft-owned rotary caches
        # from the draft config instead of reusing main_model._cos/_sin.
        LlamaTpPartModel._init_custom(self)
        self.dist_group = dist_group_manager.get_default_group()
        self.block_size = int(self.config["block_size"])
        self.mask_token_id = int(self.config["mask_token_id"])
        return

    def _init_mem_manager(self):
        target_mem_manager = self.main_model.mem_manager
        target_linear_config = getattr(target_mem_manager, "linear_config", None)
        assert (
            target_linear_config is not None
        ), "Qwen3_5DFlashModel requires a Qwen3Next target"

        draft_head_dim = self.config.get("head_dim")
        if draft_head_dim is None:
            draft_head_dim = (
                self.config["hidden_size"] // self.config["num_attention_heads"]
            )
        draft_kv_heads = int(self.config["num_key_value_heads"])
        target_kv_heads = int(target_linear_config.full_att_all_num_kv_heads)
        draft_layer_num = int(self.config["n_layer"])
        reserved_draft_layer_num = int(target_linear_config.draft_full_att_kv_layer_num)
        assert (
            int(draft_head_dim) == int(target_mem_manager.head_dim)
            and draft_kv_heads == target_kv_heads
        ), (
            "Qwen3.5 block draft currently requires draft and target full-attention KV shapes to match: "
            f"draft=({draft_kv_heads}, {draft_head_dim}), "
            f"target=({target_kv_heads}, {target_mem_manager.head_dim})"
        )
        assert reserved_draft_layer_num >= draft_layer_num, (
            "Qwen3Next target KV cache did not reserve enough draft layers: "
            f"required={draft_layer_num}, reserved={reserved_draft_layer_num}"
        )
        return super()._init_mem_manager()

    def _init_weights(self, start_layer_index=None):
        super()._init_weights(start_layer_index=start_layer_index)
        if self.share_target_embedding_and_lm_head:
            self.pre_post_weight.wte_weight_ = (
                self.main_model.pre_post_weight.wte_weight_
            )
            self.pre_post_weight.lm_head_weight_ = (
                self.main_model.pre_post_weight.lm_head_weight_
            )
        return
