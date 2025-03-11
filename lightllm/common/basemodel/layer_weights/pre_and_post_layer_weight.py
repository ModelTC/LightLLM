from .base_layer_weight import BaseLayerWeight


class PreAndPostLayerWeight(BaseLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__()
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.init_static_params()
        return
