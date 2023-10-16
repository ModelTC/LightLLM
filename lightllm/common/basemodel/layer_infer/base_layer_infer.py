from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight

class BaseLayerInfer:

    def __init__(self) -> None:
        pass

    @mark_cost_time("pre context forward")  # dont to remove this, will make performence down, did not know why
    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")