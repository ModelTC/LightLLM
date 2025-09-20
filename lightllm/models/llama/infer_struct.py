import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.common.basemodel.batch_objs import ModelInput


class LlamaInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None

    def init_some_extra_state(self, model, model_input: ModelInput):
        super().init_some_extra_state(model, model_input)
        if self.is_prefill:
            b_ready_cache_len_numpy = model_input.b_ready_cache_len_cpu.numpy()
            self.b_ready_cache_len_numpy = b_ready_cache_len_numpy

            self.max_seq_len = self.max_kv_seq_len
            self.q_max_seq_len = self.max_q_seq_len
            position_ids = self.position_ids
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        else:
            position_ids = self.position_ids
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
        return
