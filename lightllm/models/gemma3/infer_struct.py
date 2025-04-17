import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Gemma3InferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos_global = None
        self.position_sin_global = None
        self.position_sin_local = None
        self.position_cos_local = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            self.max_seq_len = b_seq_len_numpy.max()
            b_ready_cache_len_numpy = self.b_ready_cache_len.cpu().numpy()
            position_ids = torch.from_numpy(
                np.concatenate(
                    [np.arange(b_ready_cache_len_numpy[i], b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))],
                    axis=0,
                )
            ).cuda()
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
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
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
