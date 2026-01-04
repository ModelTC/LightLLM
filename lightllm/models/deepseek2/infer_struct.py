import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Deepseek2InferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.kv_starts = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if not self.is_prefill:
            self.kv_starts = self.b1_cu_kv_seq_len

        if self.is_prefill:
            self.b1_kv_start_loc = self.b1_cu_kv_seq_len
            self.max_value_in_b_seq_len = self.max_kv_seq_len
        return
