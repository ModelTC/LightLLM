from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo


class Qwen3VLInferStateInfo(Qwen2VLInferStateInfo):
    def __init__(self):
        super().__init__()
        self.input_ids = None
        self.img_start_token_ids = None
        self.img_token_lens = None
        self.img_start_locs_in_cache = None
