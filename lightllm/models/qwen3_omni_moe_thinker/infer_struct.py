from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo


class Qwen3OmniMOEInferStateInfo(Qwen3VLInferStateInfo):
    def __init__(self):
        super().__init__()

    def get_mrope_position(self, multimodal_params):
        return super().get_mrope_position(multimodal_params, is_qwen3_omini=True)
