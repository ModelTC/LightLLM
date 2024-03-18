import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo


class StarcoderInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_ids = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            b_ready_cache_len_numpy = self.b_ready_cache_len.cpu().numpy()
            self.position_ids = torch.from_numpy(
                np.concatenate(
                    [np.arange(b_ready_cache_len_numpy[i], b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))],
                    axis=0,
                )
            ).cuda()
        else:
            self.position_ids = self.b_seq_len - 1
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.b_req_idx[0], 0].item()
        return
