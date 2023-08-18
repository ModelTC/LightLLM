from .base_layer_infer import BaseLayerInfer


class PreLayerInfer(BaseLayerInfer):
    """
    """
    def __init__(self, tp_rank, world_size, network_config, mode):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.network_config_ = network_config
        self.mode = mode
        return
