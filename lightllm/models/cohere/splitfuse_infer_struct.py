from lightllm.models.cohere.infer_struct import CohereInferStateInfo
from lightllm.models.llama.splitfuse_infer_struct import LlamaSplitFuseInferStateInfo


class CohereSplitFuseInferStateInfo(LlamaSplitFuseInferStateInfo):
    inner_infer_state_class = CohereInferStateInfo

    def __init__(self):
        super().__init__()
        self._attn_out = None
        self._ffn_out = None
