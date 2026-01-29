from .base_layer_infer import BaseLayerInfer


class PostLayerInfer(BaseLayerInfer):
    """ """

    def __init__(self, network_config):
        super().__init__()
        self.network_config_ = network_config
        return
