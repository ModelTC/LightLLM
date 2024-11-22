import yaml
import collections
from .registry import QUANTMETHODS
from .ppl_quant import *
from .torchao_quant import *
from .vllm_quant import *


class Quantcfg:
    def __init__(self, layer_num, quant_type=None, cfg_path=None):
        self.layer_num = layer_num
        self.quant_type = quant_type
        self.parse_cfg(cfg_path)

    def parse_cfg(self, cfg_path):
        self.quant_cfg = collections.defaultdict(dict)
        if cfg_path is None:
            return

        with open(cfg_path, "r") as file:
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
