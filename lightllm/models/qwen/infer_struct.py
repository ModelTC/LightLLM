import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo

class QwenInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None
        self.logn_values = None

    def init_some_extra_state(self, 
            model, 
            batch_size, 
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            b_loc : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            is_prefill):
        if is_prefill:
            b_seq_len_numpy = b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                            for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            if model.logn_tensor is not None:
                self.logn_values = torch.index_select(model.logn_tensor, 0, position_ids).view(-1)
            position_ids = None
        else:
            position_ids = b_seq_len - 1
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(b_seq_len.shape[0], -1)
            if model.logn_tensor is not None:
                self.logn_values = torch.index_select(model.logn_tensor, 0, position_ids).view(-1)
            position_ids = None
            self.other_kv_index = b_loc[0, max_len_in_batch - 1].item()
        return
