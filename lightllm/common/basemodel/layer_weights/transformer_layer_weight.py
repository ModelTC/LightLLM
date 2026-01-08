from functools import partial

# from lightllm.common.layers.mm import MM
from .base_layer_weight import BaseLayerWeight
from .meta_weights import BaseWeight, MMWeightTpl
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class TransformerLayerWeight(BaseLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg):
        super().__init__()
        self.layer_num_ = layer_num
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.quant_cfg = quant_cfg
        self._parse_config()
        self._init_weight_names()
        self._init_weight()
        return

    def _parse_config(self):
        pass

    def _init_weight_names(self):
        pass

    def _init_weight(self):
        pass

    def load_hf_weights(self, weights):
        """
        load weights
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, MMWeightTpl) and len(attr.weight_names) >= 2:
                with self.lock:
                    attr.load_hf_weights(weights)
            elif isinstance(attr, BaseWeight):
                attr.load_hf_weights(weights)
