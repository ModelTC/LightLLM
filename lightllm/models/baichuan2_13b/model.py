from lightllm.models.baichuan13b.layer_weights.transformer_layer_weight import BaiChuan13bTransformerLayerWeight
from lightllm.models.baichuan2_7b.layer_weights.pre_and_post_layer_weight import Baichuan2_7bPreAndPostLayerWeight
from lightllm.models.baichuan13b.layer_infer.transformer_layer_infer import Baichuan13bTransformerLayerInfer
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.model import LlamaTpPartModel


class Baichuan2_13bTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = BaiChuan13bTransformerLayerWeight
    pre_and_post_weight_class = Baichuan2_7bPreAndPostLayerWeight

    # infer class
    transformer_layer_infer_class = Baichuan13bTransformerLayerInfer

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
    
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        return
    
    def _verify_params(self):
        assert self.load_way == "HF", "llama only support HF format to load Now!"
