import torch
import numpy as np
from lightllm.models.mistral.infer_struct import MistralInferStateInfo
from lightllm.models.mistral.triton_kernel.init_att_sliding_window_info import init_att_window_info_fwd
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MixtralInferStateInfo(MistralInferStateInfo):
    def __init__(self):
        super().__init__()
        self.sliding_window = None
        self.b_att_seq_len = None
        self.b_att_start_loc = None
        self.total_cache_num = None
        self.experts_topk = None
        self.num_local_experts = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        # sliding_window is not used in Mixtral 8x7b, ignore it
        self.sliding_window = 4096 if model.config["sliding_window"] is None else model.config["sliding_window"]
        self.experts_topk = model.config["num_experts_per_tok"]
        self.num_local_experts = model.config["num_local_experts"]
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

            # [SYM] still reserve all kv cache
            self.b_att_seq_len = torch.zeros_like(self.b_seq_len)
            init_att_window_info_fwd(self.batch_size, self.b_seq_len, self.b_att_seq_len, self.sliding_window)
            self.b_att_start_loc = torch.cumsum(self.b_att_seq_len, 0) - self.b_att_seq_len
            self.total_cache_num = torch.sum(self.b_att_seq_len).item()
        return
