from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo


class Qwen35InferStateInfo(Qwen2VLInferStateInfo):
    def __init__(self):
        super().__init__()
        self.gate_value = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        from lightllm.common.basemodel.mtp_verify_extra_state import init_mtp_verify_extra_state

        init_mtp_verify_extra_state(self, model)
        return
