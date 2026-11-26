from lightllm.common.attention.base_backend import AttentionBackend
from lightllm.common.attention.backends.gqa import selector as gqa_selector
from lightllm.common.attention.backends.gqa import GQAAttConfig, get_gqa_kv_quant
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


class Qwen3DFlash2Model(Qwen3DFlashModel):
    """Qwen3 DFlash2 draft model."""

    pre_and_post_weight_class = Qwen3DFlash2PreAndPostLayerWeight
    transformer_weight_class = Qwen3DFlash2TransformerLayerWeight
    post_layer_infer_class = Qwen3DFlash2PostLayerInfer
    transformer_layer_infer_class = Qwen3DFlash2TransformerLayerInfer

    @classmethod
    def create_att_backends(cls, model: "Qwen3DFlash2Model") -> tuple[AttentionBackend, ...]:
        if not model.layers_infer:
            return ()
        # Prefill only writes draft KV; attention over the draft block runs during decode.
        config = model.config
        layer_ids = tuple(layer.layer_num_ for layer in model.layers_infer)
        window = max(int(config.get("sliding_window") or 0), 0) - 1
        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        state = model.att_state_manager.get_state(
            model_role=model.att_state_role, state_type_filter="gqa", logical_layer_id=layer_ids[0]
        )
        kv_quant, group_size, has_q_scale = get_gqa_kv_quant(
            llm_kv_type=model.args.llm_kv_type,
            kv_scales=state.get_layer_kv_scales(logical_layer_id=layer_ids[0]),
            head_dim=head_dim,
        )
        att_config = GQAAttConfig(
            num_heads=config["num_attention_heads"] // model.tp_world_size_,
            num_kv_heads=max(config["num_key_value_heads"] // model.tp_world_size_, 1),
            head_dim=head_dim,
            dtype=model.data_type,
            causal=not model.is_mtp_draft_model or model.args.mtp_mode not in ("dspark", "dflash", "dflash2"),
            kv_quant=kv_quant,
            kv_quant_group_size=group_size,
            has_q_scale=has_q_scale,
            sliding_window=(window, window),
        )
        return (
            gqa_selector.select_gqa_decode_backend(att_config=att_config, model=model, logical_layer_ids=layer_ids),
        )

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        """合并 draft 专有配置，再同步别名及 rotary 参数；启动与 CPU cache 使用同一结果。"""
        config = model_config.get("text_config", model_config).copy()
        dflash_config = config.get("dflash_config", {})
        if not isinstance(dflash_config, dict):
            raise ValueError("dflash_config must be an object in the DFlash2 checkpoint config")
        config.update(dflash_config)
        config = super().normalize_model_config(config)

        rope_parameters = config.get("rope_parameters", {})
        if "rope_theta" in rope_parameters and "rope_theta" not in config:
            config["rope_theta"] = rope_parameters["rope_theta"]
        if "partial_rotary_factor" in rope_parameters and "partial_rotary_factor" not in config:
            config["partial_rotary_factor"] = rope_parameters["partial_rotary_factor"]
        if rope_parameters and "rope_scaling" not in config:
            config["rope_scaling"] = rope_parameters
        return config

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
        # This implementation targets the released top-16 selector checkpoint.
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

    def _init_weights(self, start_layer_index=None):
        super()._init_weights(start_layer_index=start_layer_index)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
