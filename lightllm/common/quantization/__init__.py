import yaml
import collections
from .registry import QUANTMETHODS
from .ppl_quant import *
from .torchao_quant import *
from .vllm_quant import *
from .triton_quant.triton_quant import *


class Quantcfg:
    def __init__(self, network_config, quant_type=None, custom_cfg_path=None):
        self.layer_num = network_config["n_layer"]
        self.quant_type = quant_type
        self.network_config_ = network_config
        self._parse_custom_cfg(custom_cfg_path)
        self._parse_network_config(network_config)

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
                self.quant_type = "vllm-fp8w8a8-b128"
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
            layer_name = layer_quant_cfg["layer_name"]
            layer_nums = layer_quant_cfg.get("layer_nums", range(self.layer_num))
            layer_quant_type = layer_quant_cfg["quant_type"]
            for layer_num in layer_nums:
                self.quant_cfg[layer_num].update({layer_name: layer_quant_type})

    def get_quant_type(self, layer_num, layer_name):
        return self.quant_cfg[layer_num][layer_name]

    def set_quant_type(self, layer_num, layer_name, quant_type):
        self.quant_cfg[layer_num][layer_name] = quant_type

    def get_mixed_list(self, layer_num):
        return self.quant_cfg[layer_num].keys()

    def get_default_quant_method(self):
        if self.quant_type is None:
            return None
        return QUANTMETHODS.get(self.quant_type)

    def get_quant_method(self, layer_num, layer_name):
        if self.quant_type is None:
            return None
        layer_cfg = self.quant_cfg[layer_num]
        return QUANTMETHODS.get(layer_cfg[layer_name])
