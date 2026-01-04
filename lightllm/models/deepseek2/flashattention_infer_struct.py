import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy


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

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        args_mtp_step = get_env_start_args().mtp_step
        if self.is_prefill:
            self.cu_seqlens_q = self.b1_cu_q_seq_len
            self.cu_seqlens_k = self.b1_cu_kv_seq_len
            self.has_prefix_kv = self.max_cache_len > 0
            if self.has_prefix_kv:
                self.cu_seqlens_prefix_k = torch.nn.functional.pad(
                    torch.cumsum(self.b_ready_cache_len, dim=0, dtype=torch.int32), (1, 0)
                )
                self.prefix_k_max_len = self.max_cache_len
                self.prefix_total_token_num = self.prefix_total_token_num
        else:
            # Meta information of flashattention for decoding
            self.cu_seqlens_q = self.b1_cu_q_seq_len
            self.cu_seqlens_k = self.b1_cu_kv_seq_len
            max_seq_len_k = self.max_kv_seq_len
            att_batch_size = self.batch_size // (args_mtp_step + 1)
            if self.batch_size <= model.graph_max_batch_size and self.max_len_in_batch <= model.graph_max_len_in_batch:
                page_buffer = Deepseek2FlashAttentionStateInfo.get_page_table_buffer(
                    model.graph_max_batch_size, model.graph_max_len_in_batch
                )
                self.page_table = page_buffer[self.microbatch_index][
                    : att_batch_size * model.graph_max_len_in_batch
                ].view(att_batch_size, model.graph_max_len_in_batch)
            else:
                self.page_table = torch.empty((att_batch_size, self.max_len_in_batch), dtype=torch.int32).to(
                    self.input_ids.device
                )
            page_table_copy(
                page_table=self.page_table[:, :max_seq_len_k],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.b_req_idx[args_mtp_step :: (args_mtp_step + 1)],
            )
            if args_mtp_step > 0:
                self.b_att_seq_len = self.b_seq_len[args_mtp_step :: (args_mtp_step + 1)].contiguous()
            else:
                self.b_att_seq_len = self.b_seq_len
        return
