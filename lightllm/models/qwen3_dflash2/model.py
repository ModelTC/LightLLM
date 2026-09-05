from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dflash2.layer_infer.post_layer_infer import Qwen3DFlash2PostLayerInfer
from lightllm.models.qwen3_dflash2.layer_infer.transformer_layer_infer import Qwen3DFlash2TransformerLayerInfer
from lightllm.models.qwen3_dflash2.layer_weights.pre_and_post_layer_weight import (
    Qwen3DFlash2PreAndPostLayerWeight,
)
from lightllm.models.qwen3_dflash2.layer_weights.transformer_layer_weight import (
    Qwen3DFlash2TransformerLayerWeight,
)


@DraftModelRegistry(model_type="qwen3", spec_modes="dflash2")
class Qwen3DFlash2Model(Qwen3DFlashModel):
    """Qwen3 DFlash2 draft model."""

    pre_and_post_weight_class = Qwen3DFlash2PreAndPostLayerWeight
    transformer_weight_class = Qwen3DFlash2TransformerLayerWeight
    post_layer_infer_class = Qwen3DFlash2PostLayerInfer
    transformer_layer_infer_class = Qwen3DFlash2TransformerLayerInfer

    def _init_config(self):
        super()._init_config()
        dflash_config = self.config.get("dflash_config", {})
        if not isinstance(dflash_config, dict):
            raise ValueError("dflash_config must be an object in the DFlash2 checkpoint config")
        self.config.update(dflash_config)

        rope_parameters = self.config.get("rope_parameters", {})
        if "rope_theta" in rope_parameters and "rope_theta" not in self.config:
            self.config["rope_theta"] = rope_parameters["rope_theta"]
        if "partial_rotary_factor" in rope_parameters and "partial_rotary_factor" not in self.config:
            self.config["partial_rotary_factor"] = rope_parameters["partial_rotary_factor"]
        if rope_parameters and "rope_scaling" not in self.config:
            self.config["rope_scaling"] = rope_parameters

    def _verify_params(self):
        LlamaTpPartModel._verify_params(self)
        assert not self.enable_tpsp_mix_mode, "Qwen3 DFlash2 draft model does not support TP-SP"

        if self.args.llm_kv_type == "fp8kv_sph":
            raise NotImplementedError(
                "DFlash2 sliding-window attention does not support fp8kv_sph; use --llm_kv_type None."
            )

        selector_top_k = self.config.get("selector_top_k")
        if selector_top_k is None:
            raise ValueError("selector_top_k is required in the DFlash2 checkpoint config")
        selector_top_k = int(selector_top_k)
        # 请求采样状态的候选缓冲区在 ReqSamplingParamsManager._init_dflash2_buffers 中固定分配为 16。
        if selector_top_k != 16:
            raise ValueError(f"DFlash2 requires selector_top_k=16, got {selector_top_k}")
        self.config["selector_top_k"] = selector_top_k

        physical_block_size = self.args.mtp_step + 1
        assert physical_block_size <= self.config["block_size"]
        self.config["block_size"] = physical_block_size

    def _init_custom(self):
        # The released Qwen3.8 drafter uses its own Qwen3 rotary layout.
        LlamaTpPartModel._init_custom(self)
        self.block_size = int(self.config["block_size"])
        self.mask_token_id = int(self.config["mask_token_id"])

    def _init_mem_manager(self):
        main_mem_manager = self.main_model.mem_manager
        draft_head_num = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)
        draft_width = draft_head_num * self.config["head_dim"]
        target_width = main_mem_manager.head_num * main_mem_manager.head_dim
        assert draft_width == target_width, (
            "DFlash2 draft and target KV must have equal flat widths: "
            f"draft=({draft_head_num}, {self.config['head_dim']}), "
            f"target=({main_mem_manager.head_num}, {main_mem_manager.head_dim})"
        )
        self.mem_manager = main_mem_manager

    def _init_weights(self, start_layer_index=None):
        super()._init_weights(start_layer_index=start_layer_index)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
