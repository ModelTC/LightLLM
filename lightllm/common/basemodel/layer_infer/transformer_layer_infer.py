from .base_layer_infer import BaseLayerInfer


class TransformerLayerInfer(BaseLayerInfer):
    """ """

    def release_attention_resources(self):
        for name in (
            "attention",
            "recurrent",
            "state_bindings",
            "full_att_state",
            "full_att_view",
            "write_full_att_kv",
        ):
            self.__dict__.pop(name, None)

    def __init__(self, layer_num, network_config):
        super().__init__()
        self.layer_num_ = layer_num
        self.network_config_ = network_config
        return
