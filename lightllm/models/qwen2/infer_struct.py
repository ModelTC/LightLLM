import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager


class Qwen2InferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.sliding_window = None
        self.b_start_loc_window = None
        self.b_att_seq_len = None
        self.b_att_start_loc = None
        self.total_cache_num = None
        # self.window_postion = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        self.sliding_window = model.config["sliding_window"]
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(
                np.concatenate([np.arange(0, b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))], axis=0)
            ).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.b_req_idx[0], 0].item()

            self.b_att_seq_len = self.b_seq_len.clone()
            self.b_att_start_loc = self.b_start_loc.clone()
            self.b_start_loc_window = self.b_start_loc.clone()
            self.total_cache_num = 0
            for i in range(0, self.batch_size):
                if self.sliding_window < self.b_seq_len[i]:
                    self.b_start_loc_window[i] = self.b_seq_len[i] - self.sliding_window
                    self.b_att_seq_len[i] = self.sliding_window
                else:
                    self.b_start_loc_window[i] = 0
                    self.b_att_seq_len[i] = self.b_seq_len[i]
                self.b_att_start_loc[i] = self.total_cache_num
                self.total_cache_num += self.b_att_seq_len[i]
        return
