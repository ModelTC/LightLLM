from .base_layer_weight import BaseLayerWeight
from .meta_weights import BaseWeight, MMWeightTpl


class PreAndPostLayerWeight(BaseLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__()
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.init_static_params()
        return

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
