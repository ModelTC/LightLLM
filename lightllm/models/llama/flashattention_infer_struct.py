import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.models.deepseek2.triton_kernel.repack_kv_index import repack_kv_index


class FlashAttentionStateInfo(LlamaInferStateInfo):
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
            self.cu_seqlens_q = self.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.b1_cu_kv_seq_len.int()
            self.page_table = torch.empty((self.batch_size, self.max_seq_len), dtype=torch.int32).to(input_ids.device)
            self.page_table.copy_(model.req_manager.req_to_token_indexs[self.b_req_idx, : self.max_seq_len])
        else:
            # Meta information of flashattention for decoding
            self.cu_seqlens_q = self.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.b1_cu_kv_seq_len.int()
            max_seq_len_k = self.max_kv_seq_len
            if self.batch_size <= model.graph_max_batch_size and self.max_len_in_batch <= model.graph_max_len_in_batch:
                page_buffer = FlashAttentionStateInfo.get_page_table_buffer(
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
