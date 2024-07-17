from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight


class BaseLayerInfer:
    def __init__(self) -> None:
        pass

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def splitfuse_forward(self, input_ids, infer_state: SplitFuseInferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")
