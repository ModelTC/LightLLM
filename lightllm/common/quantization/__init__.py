import yaml
import collections
from .registry import QUANTMETHODS
from .w8a8 import *
from .w8a8gx import *
from .deepgemm import *
from .awq import *
from .no_quant import *
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import is_sm100_gpu

logger = init_logger(__name__)

EXPERT_DTYPE_TO_QUANT_TYPE = {
    "fp8": "deepgemm-fp8w8a8-b128",
    "fp4": "deepgemm-fp4fp8-b32",
}
SUPPORTED_EXPERT_DTYPES = tuple(EXPERT_DTYPE_TO_QUANT_TYPE)


class Quantcfg:
    def __init__(self, network_config, quant_type="none", custom_cfg_path=None, expert_dtype=None):
        self.layer_num = network_config["n_layer"]
        self.quant_type = quant_type
        self.expert_dtype = expert_dtype
        self.network_config_ = network_config
        self._parse_custom_cfg(custom_cfg_path)
        self._parse_network_config(network_config)

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
            return
        self.quantized_weight = True
        activation_scheme = network_config.get("activation_scheme", "dynamic")
        self.static_activation = activation_scheme == "static"
        self.hf_quantization_config = hf_quantization_config
        self.hf_quantization_method = hf_quantization_config["quant_method"]
        self._mapping_quant_method()

    def _mapping_quant_method(self):
        if self.hf_quantization_method == "fp8":
            block_size = self.hf_quantization_config.get("weight_block_size", None)
            if block_size == [128, 128]:
                from lightllm.common.quantization.deepgemm import HAS_DEEPGEMM

                if HAS_DEEPGEMM:
                    self.quant_type = "deepgemm-fp8w8a8-b128"
                else:
                    self.quant_type = "vllm-fp8w8a8-b128"
                logger.info(f"select fp8w8a8-b128 quant way: {self.quant_type}")

            # fp8 量化下，部分 MoE 模型（如 DeepSeek-V4），可以单独声明 expert 权重精度，
            # 按其值给 fused_moe 选用对应的 deepgemm 量化方法。
            expert_dtype = self.expert_dtype or self.network_config_.get("expert_dtype", None)
            if expert_dtype is None:
                return
            target = self._get_expert_quant_type(expert_dtype)
            for layer_num in range(self.layer_num):
                if self.expert_dtype is not None:
                    self.quant_cfg[layer_num]["fused_moe"] = target
                else:
                    self.quant_cfg[layer_num].setdefault("fused_moe", target)
            logger.info(f"select fused_moe quant way from expert_dtype=`{expert_dtype}`: {target}")
        elif self.hf_quantization_method == "awq":
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
        for layer_quant_cfg in data.get("mix_bits", []):
            name = layer_quant_cfg["name"]
            layer_nums = layer_quant_cfg.get("layer_nums", range(self.layer_num))
            layer_quant_type = layer_quant_cfg["quant_type"]
            for layer_num in layer_nums:
                self.quant_cfg[layer_num].update({name: layer_quant_type})

    def get_quant_type(self, layer_num, name):
        layer_config = self.quant_cfg.get(layer_num, None)
        if layer_config is None:
            return self.quant_type
        quant_type = layer_config.get(name, self.quant_type)
        return quant_type

    def get_quant_method(self, layer_num, name):
        quant_type = self.get_quant_type(layer_num, name)
        quant_method = QUANTMETHODS.get(quant_type)
        quant_method.hf_quantization_config = self.hf_quantization_config
        return quant_method
