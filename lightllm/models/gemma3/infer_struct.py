import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager


class Gemma3InferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos_global = None
        self.position_sin_global = None
        self.position_sin_local = None
        self.position_cos_local = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if self.is_prefill:
            self.max_seq_len = self.max_kv_seq_len
            position_ids = self.position_ids
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)

            self.position_cos_local = torch.index_select(model._cos_cached_local, 0, position_ids).view(
                position_ids.shape[0], -1
            )
            self.position_sin_local = torch.index_select(model._sin_cached_local, 0, position_ids).view(
                position_ids.shape[0], -1
            )

            self.position_cos_global = torch.index_select(model._cos_cached_global, 0, position_ids).view(
                position_ids.shape[0], -1
            )
            self.position_sin_global = torch.index_select(model._sin_cached_global, 0, position_ids).view(
                position_ids.shape[0], -1
            )
        else:
            position_ids = self.position_ids
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)

            self.position_cos_local = torch.index_select(model._cos_cached_local, 0, position_ids).view(
                self.b_seq_len.shape[0], -1
            )
            self.position_sin_local = torch.index_select(model._sin_cached_local, 0, position_ids).view(
                self.b_seq_len.shape[0], -1
            )

            self.position_cos_global = torch.index_select(model._cos_cached_global, 0, position_ids).view(
                self.b_seq_len.shape[0], -1
            )
            self.position_sin_global = torch.index_select(model._sin_cached_global, 0, position_ids).view(
                self.b_seq_len.shape[0], -1
            )
        return
