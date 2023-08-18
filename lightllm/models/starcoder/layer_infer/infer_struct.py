import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo

class StarcoderInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_ids = None
    
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
            self.position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                            for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
        else:
            self.position_ids = b_seq_len - 1
        return
