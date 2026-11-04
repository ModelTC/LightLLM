from lightllm.common.basemodel import InferStateInfo
from lightllm.core.runner.batch_metadata import RotaryMetadata, RotaryTables


class Gemma3InferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos_global = None
        self.position_sin_global = None
        self.position_sin_local = None
        self.position_cos_local = None

    @classmethod
    def bind_metadata(cls, model):
        return RotaryMetadata(
            (
                ("", RotaryTables(model._cos_cached, model._sin_cached)),
                ("_local", RotaryTables(model._cos_cached_local, model._sin_cached_local)),
                ("_global", RotaryTables(model._cos_cached_global, model._sin_cached_global)),
            )
        )
