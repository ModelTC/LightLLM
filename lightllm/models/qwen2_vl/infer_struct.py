import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.infer_struct import InferStateInfo


class Qwen2VLInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        InferStateInfo.init_some_extra_state(self, model, input_ids)
        if self.is_prefill:
            position_ids = self.position_ids
            self.position_sin = model._sin_cached[:, position_ids, :].unsqueeze(1)
            self.position_cos = model._cos_cached[:, position_ids, :].unsqueeze(1)
            position_ids = None
        else:
            position_ids = self.position_ids
            self.position_sin = model._sin_cached[:, position_ids, :].unsqueeze(1)
            self.position_cos = model._cos_cached[:, position_ids, :].unsqueeze(1)
        return
