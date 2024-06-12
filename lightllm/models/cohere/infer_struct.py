from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class CohereInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self._attn_out = None
        self._ffn_out = None
