"""SenseNova Loop: shared checkpoint weights and independent execution caches."""

from pathlib import Path

from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from .config import get_loop_layer_indices, get_loop_layer_types
from .layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer
from .layer_infer.pre_layer_infer import SenseNovaLoopPreLayerInfer
from .layer_weights.transformer_layer_weight import SenseNovaLoopTransformerLayerWeight
from .layer_weights.pre_and_post_layer_weight import SenseNovaLoopPreAndPostLayerWeight


class SenseNovaLoopTpPartModel(Qwen3_5TpPartModel):
    """Share physical weights, with separate KV/conv/SSM slots per execution layer."""

    transformer_weight_class = SenseNovaLoopTransformerLayerWeight
    transformer_layer_infer_class = SenseNovaLoopTransformerLayerInfer
    pre_and_post_weight_class = SenseNovaLoopPreAndPostLayerWeight
    pre_layer_infer_class = SenseNovaLoopPreLayerInfer

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        config = super().normalize_model_config(model_config)
        awq = model_config.get("loop_awq_config")
        if awq is not None:
            expected = {
                "bits": 4,
                "zero_point": True,
                "linear_layout": "awq_gemm",
                "input_scale_mode": "runtime",
                "embedding_layout": "row_major_int4",
            }
            if any(awq.get(key) != value for key, value in expected.items()) or awq.get("group_size") != 128:
                raise ValueError("Unsupported Loop AWQ export; expected runtime-scale W4A16, group 128")
            config["loop_awq_config"] = awq
            config["quantization_config"] = {
                "quant_method": "awq",
                "bits": 4,
                "group_size": awq["group_size"],
                "zero_point": True,
                "version": "gemm",
            }
        if "loop_physical_indices" not in config:
            settings = model_config["loop_settings"]
            indices = get_loop_layer_indices(settings, config["num_hidden_layers"])
            config.update(
                layer_types=get_loop_layer_types(settings, config),
                n_layer=len(indices),
                num_hidden_layers=len(indices),
                loop_physical_indices=indices,
            )
        if model_config.get("mhc") is not None:
            config["mhc"] = model_config["mhc"]
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
        args = self.args
        if args.visual_infer_in_llm:
            from .inline_vision import InlineLoopVision
            from .config import validate_inline_vision

            validate_inline_vision(args, "sensenova_loop")
            self.pre_infer.inline_vision = InlineLoopVision(self.weight_dir_, self.vision_config, args)
        # Bind split projections once for each visit to a shared physical layer.
        for layer, weight in zip(self.layers_infer, self.trans_layers_weight):
            if getattr(weight, "linear_gate_proj", None) is not None:
                layer._linear_in_proj = layer._linear_in_proj_with_gates

    def load_weights(self, weight_dict):
        assert weight_dict is None or isinstance(weight_dict, dict), "weight_dict must be a dict or None"
        execution_weights = self.trans_layers_weight
        self.trans_layers_weight = self.physical_layer_weights
        try:
            module_dir = Path(self.weight_dir_) / "modules" if self.config.get("loop_awq_config") is not None else None
            if module_dir is not None and module_dir.is_dir() and not weight_dict:
                # Original Loop AWQ exports store one module per safetensors file.
                # Load sequentially: repacking is bounded and shared physical layers load once.
                from lightllm.common.basemodel.layer_weights.hf_load_utils import load_func
                from tqdm import tqdm

                files = [
                    Path("unquantized.safetensors"),
                    *(file.relative_to(self.weight_dir_) for file in sorted(module_dir.glob("*.safetensors"))),
                ]
                for file in tqdm(files, desc="Loading Loop AWQ weights"):
                    load_func(str(file), True, self.pre_post_weight, self.physical_layer_weights, self.weight_dir_)
                self.pre_post_weight.verify_load()
                for weight in self.physical_layer_weights:
                    weight.verify_load()
            else:
                super().load_weights(weight_dict)
        finally:
            self.trans_layers_weight = execution_weights
