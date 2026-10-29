"""SenseNova Loop: shared checkpoint weights and independent execution caches."""

from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.utils.config_utils import get_config_json
from .config import get_loop_layer_indices, get_loop_layer_types
from .layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer
from .layer_weights.transformer_layer_weight import SenseNovaLoopTransformerLayerWeight


class SenseNovaLoopTpPartModel(Qwen3_5TpPartModel):
    """Share physical weights, with separate KV/conv/SSM slots per execution layer."""

    transformer_weight_class = SenseNovaLoopTransformerLayerWeight
    transformer_layer_infer_class = SenseNovaLoopTransformerLayerInfer

    def _init_config(self):
        super()._init_config()
        model_config = get_config_json(self.weight_dir_)
        settings = model_config["loop_settings"]
        self.num_physical_layers = self.config["n_layer"]
        self.loop_layer_indices = get_loop_layer_indices(settings, self.num_physical_layers)
        self.config.update(
            layer_types=get_loop_layer_types(settings, self.config),
            n_layer=len(self.loop_layer_indices),
            num_hidden_layers=len(self.loop_layer_indices),
            loop_physical_indices=self.loop_layer_indices,
        )
        if model_config.get("mhc") is not None:
            self.config["mhc"] = model_config["mhc"]

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

    def load_weights(self, weight_dict):
        execution_weights = self.trans_layers_weight
        self.trans_layers_weight = self.physical_layer_weights
        try:
            super().load_weights(weight_dict)
        finally:
            self.trans_layers_weight = execution_weights
