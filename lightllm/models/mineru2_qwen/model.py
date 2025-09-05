from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.registry import ModelRegistry


@ModelRegistry("mineru2_qwen", is_multimodal=True)
class Mineru2QwenForCausalLM(Qwen2TpPartModel):
    def __init__(self, kvargs):
        super().__init__(kvargs)
