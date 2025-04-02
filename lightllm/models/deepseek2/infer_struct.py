import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Deepseek2InferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.kv_starts = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        # 只有 decode 阶段使用 ppl 的优化算子才会有这个管理变量
        if not self.is_prefill:
            self.kv_starts = torch.cat([self.b_start_loc, self.b_start_loc[-1:] + self.b_seq_len[-1:]], dim=0)
            self.total_token_num_tensor = torch.sum(self.b_seq_len)

        if self.is_prefill:
            self.b_kv_start_loc = self.b_seq_len.cumsum(dim=0) - self.b_seq_len
            self.max_value_in_b_seq_len = self.b_seq_len.max().item()

        return
