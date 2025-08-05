import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args, get_page_size
from lightllm.models.deepseek2.triton_kernel.repack_kv_index import repack_kv_index


def cdiv(a, b):
    return (a + b - 1) // b


class LlamaFlashInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.flashinfer_extra_state = None
        self.page_size = get_page_size()

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        self.flashinfer_extra_state = model.flashinfer_extra_state

        import flashinfer

        if not self.is_prefill:
            if get_env_start_args().enable_flashinfer_decode:
                self.kv_last_page_len = torch.full((self.batch_size,), 1, dtype=torch.int32, device=input_ids.device)
                length = cdiv(self.flashinfer_extra_state.max_seq_length, self.page_size)
                if self.batch_size <= model.graph_max_batch_size:
                    self.kv_indices = self.flashinfer_extra_state.kv_indices_buffer[self.microbatch_index][
                        : self.batch_size * length
                    ]
                else:
                    self.kv_indices = torch.empty(
                        self.batch_size * length,
                        dtype=torch.int32,
                        device=input_ids.device,
                    )

                self.kv_starts = self.b1_cu_kv_seq_len.int()
                if "page_size_variable" in model.mode:
                    b_page_len = cdiv(self.b_seq_len, self.page_size)
                    self.kv_starts[1:] = b_page_len.cumsum(0)
                    self.kv_last_page_len = self.b_seq_len - (b_page_len - 1) * self.page_size
                    repack_kv_index(
                        self.req_manager.req_to_page_indexs,
                        self.b_req_idx,
                        b_page_len,
                        self.kv_starts[:-1],
                        cdiv(self.max_kv_seq_len, self.page_size),
                        self.kv_indices,
                    )
                else:
                    repack_kv_index(
                        self.req_manager.req_to_token_indexs,
                        self.b_req_idx,
                        self.b_seq_len,
                        self.b_start_loc,
                        self.max_kv_seq_len,
                        self.kv_indices,
                    )
                if self.decode_wrapper is None:
                    self.decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                        self.flashinfer_extra_state.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=True,
                        paged_kv_indptr_buffer=self.kv_starts,
                        paged_kv_indices_buffer=self.kv_indices,
                        paged_kv_last_page_len_buffer=self.kv_last_page_len,
                    )
                    self.decode_wrapper.plan(
                        self.kv_starts,
                        self.kv_indices,
                        self.kv_last_page_len,
                        self.flashinfer_extra_state.tp_q_head_num,
                        self.flashinfer_extra_state.tp_kv_head_num,
                        self.flashinfer_extra_state.head_dim,
                        self.page_size,
                        q_data_type=self.flashinfer_extra_state.q_data_type,
                        kv_data_type=self.flashinfer_extra_state.kv_data_type,
                        non_blocking=True,
                    )
        else:
            if get_env_start_args().enable_flashinfer_prefill:
                q_starts = self.b1_cu_q_seq_len.int()
                kv_starts = self.b1_cu_kv_seq_len.int()
                kv_last_page_len = torch.full((self.batch_size,), 1, dtype=torch.int32, device=input_ids.device)
                length = cdiv(self.flashinfer_extra_state.max_seq_length, self.page_size)
                kv_indices = torch.empty(
                    self.batch_size * length,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
                if "page_size_variable" in model.mode:
                    b_page_len = cdiv(self.b_seq_len, self.page_size)
                    kv_starts[1:] = b_page_len.cumsum(0)
                    kv_last_page_len = self.b_seq_len - (b_page_len - 1) * self.page_size
                    repack_kv_index(
                        self.req_manager.req_to_page_indexs,
                        self.b_req_idx,
                        b_page_len,
                        kv_starts[:-1],
                        cdiv(self.max_kv_seq_len, self.page_size),
                        kv_indices,
                    )
                else:
                    repack_kv_index(
                        self.req_manager.req_to_token_indexs,
                        self.b_req_idx,
                        self.b_seq_len,
                        kv_starts[:-1],
                        self.max_kv_seq_len,
                        kv_indices,
                    )
                self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                    self.flashinfer_extra_state.workspace_buffer,
                    qo_indptr_buf=q_starts,
                    paged_kv_indptr_buf=kv_starts,
                    paged_kv_indices_buf=kv_indices,
                    paged_kv_last_page_len_buf=kv_last_page_len,
                )
                self.prefill_wrapper.plan(
                    q_starts,
                    kv_starts,
                    kv_indices,
                    kv_last_page_len,
                    self.flashinfer_extra_state.tp_q_head_num,
                    self.flashinfer_extra_state.tp_kv_head_num,
                    self.flashinfer_extra_state.head_dim,
                    self.page_size,
                    causal=True,
                    pos_encoding_mode="NONE",
                    logits_soft_cap=0.0,
                    q_data_type=self.flashinfer_extra_state.q_data_type,
                    kv_data_type=self.flashinfer_extra_state.kv_data_type,
                )
        return

    def copy_for_cuda_graph(self, new_infer_state):
        super().copy_for_cuda_graph(new_infer_state)
        if get_env_start_args().enable_flashinfer_decode and not self.is_prefill:
            self.decode_wrapper.plan(
                new_infer_state.kv_starts,
                new_infer_state.kv_indices,
                new_infer_state.kv_last_page_len,
                new_infer_state.flashinfer_extra_state.tp_q_head_num,
                new_infer_state.flashinfer_extra_state.tp_kv_head_num,
                new_infer_state.flashinfer_extra_state.head_dim,
                self.page_size,
                q_data_type=new_infer_state.flashinfer_extra_state.q_data_type,
                kv_data_type=new_infer_state.flashinfer_extra_state.kv_data_type,
                non_blocking=True,
            )
        return
