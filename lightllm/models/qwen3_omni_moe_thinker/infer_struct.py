from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo


class Qwen3OmniMOEInferStateInfo(Qwen3VLInferStateInfo):
    def __init__(self):
        self.use_image_h = False
        super().__init__()
