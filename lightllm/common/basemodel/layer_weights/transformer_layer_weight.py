from functools import partial

# from lightllm.common.layers.mm import MM
from .base_layer_weight import BaseLayerWeight
from .meta_weights import BaseWeight, MultiMMWeight, MMWeightTpl, FusedMoeWeight
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class TransformerLayerWeight(BaseLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg):
        super().__init__()
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.quant_cfg = quant_cfg
        self._parse_config()
        self._init_weight_names()
        self._init_qweight_names()
        self._init_weight()
        self._set_quantization()
        return

    def _parse_config(self):
        pass

    def _init_weight_names(self):
        pass

    def _init_qweight_names(self):
        pass

    def _init_weight(self):
        pass

    def load_hf_weights(self, weights):
        """
        load weights
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, MultiMMWeight):
                with self.lock:
                    attr.load_hf_weights(weights)
            elif isinstance(attr, BaseWeight):
                attr.load_hf_weights(weights)

    def _set_quantization(self):
        if self.quant_cfg.quant_type is None:
            return
        mix_quant_list = self.quant_cfg.get_mixed_list(self.layer_num_)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, MMWeightTpl) or isinstance(attr, FusedMoeWeight):
                if attr_name in mix_quant_list:
                    attr.set_quant_method(self.quant_cfg.get_quant_method(self.layer_num_, attr_name))
                    attr_quant_type = self.quant_cfg.get_quant_type(self.layer_num_, attr_name)
                    logger.info(f"Layer {self.layer_num_} {attr_name} is set to {attr_quant_type}")
                else:
                    attr.set_quant_method(self.quant_cfg.get_default_quant_method())
