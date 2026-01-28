from .base_layer_infer import BaseLayerInfer


class TransformerLayerInfer(BaseLayerInfer):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__()
        self.layer_num_ = layer_num
        self.network_config_ = network_config
        return
