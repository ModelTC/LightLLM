from lightllm.common.basemodel import InferStateInfo
from lightllm.core.runner.batch_metadata import RotaryMetadata, RotaryTables


class LlamaInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None

    @classmethod
    def bind_metadata(cls, model):
        return RotaryMetadata((("", RotaryTables(model._cos_cached, model._sin_cached)),))
