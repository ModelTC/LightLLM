from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo


class KimiLinearInferStateInfo(Deepseek2InferStateInfo):
    def init_some_extra_state(self, model):
        InferStateInfo.init_some_extra_state(self, model)
        self.b_conv_buffer_idx = self.b_req_idx
        self.b_buffer_idx = self.b_req_idx
        self.mla_output_gate = None
        self.attnres_state = None
