from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek3_2.layer_weights.nsa_indexer_layer_weight import NSAIndexerWeight


class Deepseek3_2TransformerLayerWeight(Deepseek2TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        self.index_topk = network_config["index_topk"]
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        self.indexer_layer_weight = NSAIndexerWeight(
            layer_num=layer_num,
            data_type=data_type,
            network_config=network_config,
            mode=mode,
            quant_cfg=quant_cfg
        )
        return
