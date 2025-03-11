from .base_layer_infer import BaseLayerInfer


class PostLayerInfer(BaseLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__()
        self.network_config_ = network_config
        self.mode = mode
        return
