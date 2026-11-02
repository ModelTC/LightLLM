import os
import yaml
import collections
from copy import deepcopy
from .registry import QUANTMETHODS
from .plan import CheckpointFormat, WeightPlan
from .w8a8 import *
from .w8a8gx import *
from .deepgemm import *
from .awq import *
from .no_quant import *
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import is_sm100_gpu

logger = init_logger(__name__)

EXPERT_DTYPE_TO_QUANT_TYPE = {
    "fp8": "fp8w8a8-b128-deepgemm",
    "fp4": "fp4fp8-b32-deepgemm",
}
SUPPORTED_EXPERT_DTYPES = tuple(EXPERT_DTYPE_TO_QUANT_TYPE)


class Quantcfg:
    def __init__(self, network_config, quant_type="none", custom_cfg_path=None, expert_dtype=None):
        self.layer_num = network_config["n_layer"]
        self.quant_type = deepcopy(quant_type)
        self._explicit_selection = isinstance(quant_type, dict)
        self.start_args_expert_dtype = expert_dtype
        self.config_expert_dtype = network_config.get("expert_dtype", None)
        # Parse quant_cfg first so model config only fills missing per-layer fused_moe entries;
        # an explicit startup argument is applied afterward and overrides every layer.
        self._parse_custom_cfg(custom_cfg_path)
        self._parse_network_config(network_config)
        self.checkpoint_format = CheckpointFormat.from_config(self.hf_quantization_config)
        self.use_ue8m0_scales = (
            self.hf_quantization_config is not None and self.hf_quantization_config.get("scale_fmt") == "ue8m0"
        ) or os.getenv("LIGHTLLM_USE_UE8M0_SCALES", "0").upper() in ("ON", "TRUE", "1")

    def _get_expert_quant_type(self, expert_dtype):
        quant_type = EXPERT_DTYPE_TO_QUANT_TYPE.get(expert_dtype)
        if quant_type is None:
            raise ValueError(
                f"unsupported expert_dtype `{expert_dtype}`; expected one of {list(SUPPORTED_EXPERT_DTYPES)}"
            )
        if expert_dtype == "fp4" and not is_sm100_gpu():
            raise RuntimeError("expert_dtype `fp4` requires an SM100 GPU; please use `fp8` on non-SM100 GPUs.")
        return quant_type

    def _parse_network_config(self, network_config):
        hf_quantization_config = network_config.get("quantization_config", None)
        if hf_quantization_config is None:
            self.quantized_weight = False
            self.static_activation = False
            self.hf_quantization_config = None
            self._mapping_expert_quant_method()
            return
        self.quantized_weight = True
        activation_scheme = network_config.get("activation_scheme", "dynamic")
        self.static_activation = activation_scheme == "static"
        self.hf_quantization_config = deepcopy(hf_quantization_config)
        self.hf_quantization_method = hf_quantization_config["quant_method"]
        self._mapping_quant_method()

    def _mapping_expert_quant_method(self):
        expert_dtype = self.start_args_expert_dtype or self.config_expert_dtype
        if expert_dtype is None:
            return

        target = self._get_expert_quant_type(expert_dtype)
        for layer_num in range(self.layer_num):
            layer_quant_cfg = self.quant_cfg[layer_num]
            if self.start_args_expert_dtype is not None:
                # 优先级 1：启动参数显式指定 expert_dtype，覆盖其他来源的配置。
                layer_quant_cfg["fused_moe"] = target
            elif "fused_moe" in layer_quant_cfg:
                # 优先级 2：未指定启动参数时，保留 quant_cfg 中已有的逐层配置。
                continue
            else:
                # 优先级 3：前两个来源均未指定时，使用 config.json 中的 expert_dtype。
                layer_quant_cfg["fused_moe"] = target

        source = "startup arguments" if self.start_args_expert_dtype is not None else "model config"
        logger.info(
            f"select fused_moe quant way from {source} expert_dtype=`{expert_dtype}`: {target}; "
            "priority is startup arguments > quant_cfg > model config."
        )

    def _mapping_quant_method(self):
        if self.hf_quantization_method == "fp8":
            block_size = self.hf_quantization_config.get("weight_block_size", None)
            if block_size == [128, 128] and not self._explicit_selection:
                from lightllm.common.quantization.deepgemm import HAS_DEEPGEMM

                if HAS_DEEPGEMM:
                    self.quant_type = "fp8w8a8-b128-deepgemm"
                else:
                    self.quant_type = "fp8w8a8-b128-vllm"
                logger.info(f"select fp8w8a8-b128 quant way: {self.quant_type}")
            self._mapping_expert_quant_method()

        elif self.hf_quantization_method == "awq":
            if not self._explicit_selection:
                self.quant_type = "awq"
                if is_awq_marlin_compatible(self.hf_quantization_config):
                    self.quant_type = "awq_marlin"
            logger.info(f"select awq quant way: {self.quant_type}")
        else:
            # TODO: more quant method
            pass

    def _parse_custom_cfg(self, custom_cfg_path):
        self.quant_cfg = collections.defaultdict(dict)
        if custom_cfg_path is None:
            return

        with open(custom_cfg_path, "r") as file:
            data = yaml.safe_load(file)

        self.quant_type = data["quant_type"]
        self._explicit_selection = isinstance(self.quant_type, dict)
        for layer_quant_cfg in data.get("mix_bits", []):
            name = layer_quant_cfg["name"]
            layer_nums = layer_quant_cfg.get("layer_nums", range(self.layer_num))
            layer_quant_type = layer_quant_cfg["quant_type"]
            for layer_num in layer_nums:
                self.quant_cfg[layer_num].update({name: layer_quant_type})

    def get_quant_type(self, layer_num, name):
        layer_config = self.quant_cfg.get(layer_num, None)
        if layer_config is None:
            return QUANTMETHODS.resolve(self.quant_type)
        quant_type = layer_config.get(name, self.quant_type)
        return QUANTMETHODS.resolve(quant_type)

    def get_weight_plan(self, layer_num, name):
        key = self.get_quant_type(layer_num, name)
        selection = QUANTMETHODS.describe(key)
        requested = self.quant_cfg.get(layer_num, {}).get(name, self.quant_type)
        if isinstance(requested, dict):
            checkpoint = self.checkpoint_format.encoding
            if selection.weight_format.startswith("awq_"):
                if checkpoint != "awq":
                    raise ValueError(
                        "AWQ representations require an AWQ checkpoint; online AWQ encoding is unavailable"
                    )
                if selection.implementation == "marlin" and not is_awq_marlin_compatible(self.hf_quantization_config):
                    raise ValueError("the AWQ checkpoint is incompatible with Marlin packing")
            elif checkpoint == "awq":
                raise ValueError("AWQ checkpoints require an explicitly supported AWQ representation")
            elif checkpoint == "fp8" and self.hf_quantization_config.get("weight_block_size") == [128, 128]:
                if selection.weight_format != "fp8_block128":
                    raise ValueError("block FP8 checkpoint scales require the matching weight representation")
            if selection.weight_format == "fp4_group32" and not is_sm100_gpu():
                raise ValueError("FP4 group32 execution requires SM100")
        return WeightPlan(self.checkpoint_format, selection, key, self.use_ue8m0_scales)

    def get_quant_method(self, layer_num, name):
        return self.get_weight_plan(layer_num, name).bind(QUANTMETHODS)

    def content_identity(self):
        def describe(layer, name):
            plan = self.get_weight_plan(layer, name)
            result = dict(semantics=QUANTMETHODS.content_key(plan.method_key))
            if plan.method_key == "fp8w8a8-b128-deepgemm":
                result["power_of_two_scales"] = plan.use_ue8m0_scales
            return result

        default = describe(-1, "")
        overrides = {}
        for layer, values in self.quant_cfg.items():
            fields = {name: describe(layer, name) for name in values}
            fields = {name: value for name, value in fields.items() if value != default}
            if fields:
                overrides[layer] = fields
        return dict(checkpoint=self.checkpoint_format, default=default, overrides=overrides)
