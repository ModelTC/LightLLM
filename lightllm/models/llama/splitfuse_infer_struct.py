import torch
import numpy as np
from lightllm.common.basemodel import SplitFuseInferStateInfo
from lightllm.common.req_manager import ReqManager
from .infer_struct import LlamaInferStateInfo


class LlamaSplitFuseInferStateInfo(SplitFuseInferStateInfo):

    inner_decode_infer_state_class = LlamaInferStateInfo

    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        position_ids = []
        if self.decode_req_num != 0:
            position_ids.append((self.decode_b_seq_len - 1).cpu().numpy())
        if self.prefill_req_num != 0:
            b_seq_len_numpy = self.prefill_b_seq_len.cpu().numpy()
            b_ready_cache_len_numpy = self.prefill_b_split_ready_cache_len.cpu().numpy()
            position_ids.extend(
                [np.arange(b_ready_cache_len_numpy[i], b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))]
            )

        position_ids = torch.from_numpy(np.concatenate(position_ids, axis=0)).cuda().view(-1)
        self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)

        if self.decode_req_num != 0:
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.decode_b_req_idx[0], 0].item()
        elif self.prefill_req_num != 0:
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.prefill_b_req_idx[0], 0].item()
        return

    def create_inner_decode_infer_status(self):
        infer_status = super().create_inner_decode_infer_status()
        infer_status.other_kv_index = self.other_kv_index
        return
