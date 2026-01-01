import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.deepseek2.triton_kernel.repack_kv_index import repack_kv_index


class Deepseek2FlashInferStateInfo(Deepseek2InferStateInfo):
    def __init__(self):
        super().__init__()
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.flashinfer_extra_state = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.flashinfer_extra_state = model.flashinfer_extra_state

        import flashinfer

        if not self.is_prefill:
            if get_env_start_args().enable_flashinfer_decode:
                self.q_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32).to(self.input_ids.device)
                if self.batch_size <= model.graph_max_batch_size:
                    self.kv_indices = self.flashinfer_extra_state.kv_indices_buffer[self.microbatch_index][
                        : self.batch_size * self.flashinfer_extra_state.max_seq_length
                    ]
                else:
                    self.kv_indices = torch.empty(
                        self.batch_size * self.flashinfer_extra_state.max_seq_length,
                        dtype=torch.int32,
                        device=self.input_ids.device,
                    )
                repack_kv_index(
                    self.req_manager.req_to_token_indexs,
                    self.b_req_idx,
                    self.b_seq_len,
                    self.b_start_loc,
                    self.max_len_in_batch,
                    self.kv_indices,
                )
                if self.decode_wrapper is None:
                    self.decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                        self.flashinfer_extra_state.workspace_buffer,
                        use_cuda_graph=True,
                        qo_indptr=self.q_indptr,
                        kv_indices=self.kv_indices,
                        kv_indptr=self.kv_starts,
                        kv_len_arr=self.b_seq_len,
                    )
                    self.decode_wrapper.plan(
                        self.q_indptr,
                        self.kv_starts,
                        self.kv_indices,
                        self.b_seq_len,
                        self.flashinfer_extra_state.tp_q_head_num,
                        self.flashinfer_extra_state.kv_lora_rank,
                        self.flashinfer_extra_state.qk_rope_head_dim,
                        1,
                        False,  # causal
                        self.flashinfer_extra_state.softmax_scale,
                        self.flashinfer_extra_state.q_data_type,
                        self.flashinfer_extra_state.kv_data_type,
                    )
        else:
            if get_env_start_args().enable_flashinfer_prefill:
                q_starts = self.b1_cu_q_seq_len.int()
                kv_starts = self.b1_kv_start_loc.int()
                if self.prefill_wrapper is None:
                    self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                        self.flashinfer_extra_state.workspace_buffer, "NHD"
                    )
                self.prefill_wrapper.plan(
                    qo_indptr=q_starts,
                    kv_indptr=kv_starts,
                    num_qo_heads=self.flashinfer_extra_state.tp_q_head_num,
                    num_kv_heads=self.flashinfer_extra_state.tp_q_head_num,
                    head_dim_qk=self.flashinfer_extra_state.qk_nope_head_dim
                    + self.flashinfer_extra_state.qk_rope_head_dim,
                    head_dim_vo=self.flashinfer_extra_state.qk_nope_head_dim,
                    q_data_type=self.flashinfer_extra_state.q_data_type,
                    causal=True,
                    sm_scale=self.flashinfer_extra_state.softmax_scale,
                )
        return

    def copy_for_cuda_graph(self, new_infer_state):
        super().copy_for_cuda_graph(new_infer_state)
        if get_env_start_args().enable_flashinfer_decode and not self.is_prefill:
            self.decode_wrapper.plan(
                new_infer_state.q_indptr,
                new_infer_state.kv_starts,
                new_infer_state.kv_indices,
                new_infer_state.b_seq_len,
                new_infer_state.flashinfer_extra_state.tp_q_head_num,
                new_infer_state.flashinfer_extra_state.kv_lora_rank,
                new_infer_state.flashinfer_extra_state.qk_rope_head_dim,
                1,
                False,  # causal
                new_infer_state.flashinfer_extra_state.softmax_scale,
                new_infer_state.flashinfer_extra_state.q_data_type,
                new_infer_state.flashinfer_extra_state.kv_data_type,
            )
        return
