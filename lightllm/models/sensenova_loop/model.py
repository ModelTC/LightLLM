"""SenseNova Loop: shared checkpoint weights and independent execution caches."""

from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from .config import get_loop_layer_indices, get_loop_layer_types
from .layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer
from .layer_weights.transformer_layer_weight import SenseNovaLoopTransformerLayerWeight
from .layer_weights.pre_and_post_layer_weight import SenseNovaLoopPreAndPostLayerWeight


class SenseNovaLoopTpPartModel(Qwen3_5TpPartModel):
    """Share physical weights, with separate KV/conv/SSM slots per execution layer."""

    transformer_weight_class = SenseNovaLoopTransformerLayerWeight
    transformer_layer_infer_class = SenseNovaLoopTransformerLayerInfer
    pre_and_post_weight_class = SenseNovaLoopPreAndPostLayerWeight

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        config = super().normalize_model_config(model_config)
        # Quantization metadata lives at the top level in the released
        # SenseNova-Loop AWQ checkpoint.  Keep supporting nested text_config
        # metadata because Transformers has emitted both layouts over time.
        quantization = model_config.get("quantization_config")
        if quantization is None:
            text_config = model_config.get("text_config")
            if isinstance(text_config, dict):
                quantization = text_config.get("quantization_config")
        if quantization is not None:
            config["quantization_config"] = quantization

        if "loop_physical_indices" not in config:
            settings = model_config.get("loop_settings")
            if settings is None:
                text_config = model_config.get("text_config")
                settings = text_config.get("loop_settings") if isinstance(text_config, dict) else None
            if settings is None:
                raise ValueError("SenseNova Loop config is missing loop_settings")
            indices = get_loop_layer_indices(settings, config["num_hidden_layers"])
            config.update(
                layer_types=get_loop_layer_types(settings, config),
                n_layer=len(indices),
                num_hidden_layers=len(indices),
                loop_physical_indices=indices,
            )
        mhc = model_config.get("mhc")
        if mhc is None and isinstance(model_config.get("text_config"), dict):
            mhc = model_config["text_config"].get("mhc")
        if mhc is not None:
            config["mhc"] = mhc
        return config

    def _init_config(self):
        super()._init_config()
        self.loop_layer_indices = self.config["loop_physical_indices"]
        self.num_physical_layers = max(self.loop_layer_indices) + 1

    def _init_weights(self, start_layer_index=0):
        # Allocate and preprocess each checkpoint layer once, then alias weights.
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, network_config=self.config)
        self.physical_layer_weights = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(start_layer_index, start_layer_index + self.num_physical_layers)
        ]
        self.trans_layers_weight = [self.physical_layer_weights[i] for i in self.loop_layer_indices]

    def _init_infer_layer(self, start_layer_index=0):
        super()._init_infer_layer(start_layer_index)
        # Bind split projections once for each visit to a shared physical layer.
        for layer, weight in zip(self.layers_infer, self.trans_layers_weight):
            if getattr(weight, "linear_gate_proj", None) is not None:
                layer._linear_in_proj = layer._linear_in_proj_with_gates

    def load_weights(self, weight_dict):
        assert weight_dict is None or isinstance(weight_dict, dict), "weight_dict must be a dict or None"
        execution_weights = self.trans_layers_weight
        self.trans_layers_weight = self.physical_layer_weights
        try:
            super().load_weights(weight_dict)
        finally:
            self.trans_layers_weight = execution_weights
