from lightllm.models.qwen3_dflash.infer_struct import Qwen3DFlashInferStateInfo


class Qwen3DSparkInferStateInfo(Qwen3DFlashInferStateInfo):
    def __init__(self):
        super().__init__()
        self.confidence_logits = None
        self.draft_token_ids = None
