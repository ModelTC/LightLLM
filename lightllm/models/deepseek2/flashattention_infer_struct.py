import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.utils.dist_utils import get_current_device_id


class Deepseek2FlashAttentionStateInfo(Deepseek2InferStateInfo):
    _shared_page_table_buffer = None

    def __init__(self):
        super().__init__()

    @classmethod
    def get_page_table_buffer(cls, graph_max_batch_size: int, max_seq_len: int):
        if cls._shared_page_table_buffer is None:
            cls._shared_page_table_buffer = [
                torch.empty(graph_max_batch_size * max_seq_len, dtype=torch.int32).to(get_current_device_id()),
                torch.empty(graph_max_batch_size * max_seq_len, dtype=torch.int32).to(get_current_device_id()),
            ]
        return cls._shared_page_table_buffer

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        if self.is_prefill:
            self.cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(self.b_seq_len - self.b_ready_cache_len, dim=0, dtype=torch.int32), (1, 0)
            )
            self.cu_seqlens_k = torch.cat([self.b_start_loc, self.b_start_loc[-1:] + self.b_seq_len[-1:]], dim=0)
            self.has_prefix_kv = self.b_ready_cache_len_numpy.any()
            if self.has_prefix_kv:
                self.cu_seqlens_prefix_k = torch.nn.functional.pad(
                    torch.cumsum(self.b_ready_cache_len, dim=0, dtype=torch.int32), (1, 0)
                )
                self.prefix_k_max_len = self.b_ready_cache_len_numpy.max()
                self.prefix_total_token_num = self.b_ready_cache_len_numpy.sum()
        else:
            # Meta information of flashattention for decoding
            self.cu_seqlens_q = torch.arange(0, self.batch_size + 1, dtype=torch.int32, device=input_ids.device)
            self.cu_seqlens_k = torch.cat([self.b_start_loc, self.b_start_loc[-1:] + self.b_seq_len[-1:]], dim=0)
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            max_seq_len_k = b_seq_len_numpy.max()
            if self.batch_size <= model.graph_max_batch_size and self.max_len_in_batch <= model.graph_max_len_in_batch:
                page_buffer = Deepseek2FlashAttentionStateInfo.get_page_table_buffer(
                    model.graph_max_batch_size, model.graph_max_len_in_batch
                )
                self.page_table = page_buffer[self.microbatch_index][
                    : self.batch_size * model.graph_max_len_in_batch
                ].reshape(self.batch_size, model.graph_max_len_in_batch)
            else:
                self.page_table = torch.empty((self.batch_size, self.max_len_in_batch), dtype=torch.int32).to(
                    input_ids.device
                )

            self.page_table[:, :max_seq_len_k].copy_(
                model.req_manager.req_to_token_indexs[self.b_req_idx, :max_seq_len_k]
            )
            self.page_table[:, max_seq_len_k:].fill_(0)
        return
