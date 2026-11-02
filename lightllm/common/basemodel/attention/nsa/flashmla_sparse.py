from ..sparse import BoundSparse, SparsePrefillInput, SparseDecodeInput

# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa_backend.py
# Uses sgl_kernel.flash_mla and sgl_kernel.flash_attn from the sglang kernel library.

import dataclasses
import weakref
import torch
from typing import Tuple, TYPE_CHECKING

from ..base_att import BaseAttBackend, BaseAttState
from lightllm.utils.dist_utils import get_current_device_id

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class NsaFlashMlaSparseAttBackend(BaseAttBackend):
    metadata_lanes = 2

    def __init__(self, model):
        super().__init__(model=model)
        self.ragged_mem_buffers = None

    def runtime_memory_demands(self):
        size = self.model.graph_max_batch_size * self.model.max_seq_length
        if not size:
            return ()
        return self.workspace_memory_demands(
            f"nsa_ragged_indices_{self.model.att_state_role}", size, lanes=self.metadata_lanes, dtype=torch.int32
        )

    def reserve_runtime_resources(self):
        size = self.model.graph_max_batch_size * self.model.max_seq_length
        if not size:
            self.ragged_mem_buffers = []
            return
        self.ragged_mem_buffers = [
            self.get_gpu_workspace_buffer(
                f"nsa_ragged_indices_{self.model.att_state_role}", size, dtype=torch.int32, lane_id=lane
            )
            for lane in range(self.metadata_lanes)
        ]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaSparsePrefillAttState":
        return NsaFlashMlaSparsePrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaSparseDecodeAttState":
        return NsaFlashMlaSparseDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class NsaFlashMlaSparsePrefillAttState(BaseAttState):
    """Prefill attention state for NSA using flash_mla_sparse_fwd."""

    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaSparseAttBackend = self.backend
        self.ragged_mem_index = torch.empty(
            self.infer_state.total_token_num,
            dtype=torch.int32,
            device=get_current_device_id(),
        )
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        self.ks, self.ke, self.lengths = gen_nsa_ks_ke(
            b_seq_len=self.infer_state.b_seq_len,
            b_q_seq_len=self.infer_state.b_q_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            req_to_token_index=self.infer_state.req_manager.req_to_token_indexs,
            q_token_num=self.infer_state.input_ids.shape[0],
            ragged_mem_index=self.ragged_mem_index,
            hold_req_idx=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        return

    def run(self, inputs: SparsePrefillInput, binding: BoundSparse):
        q, kv = inputs.q, binding.cache
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = binding.softmax_scale
        kv_lora_rank = binding.value_dim

        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)

        mla_out, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=topk_mem_indices,
            sm_scale=softmax_scale,
            d_v=kv_lora_rank,
        )
        return mla_out


@dataclasses.dataclass
class NsaFlashMlaSparseDecodeAttState(BaseAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    nsa_cache_seqlens: torch.Tensor = None
    nsa_cu_seqlens_k_new: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaSparseAttBackend = self.backend
        model = self.backend.model
        use_cuda_graph = (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        )

        if use_cuda_graph:
            self.ragged_mem_index = self.backend.ragged_mem_buffers[self.infer_state.microbatch_index]
        else:
            self.ragged_mem_index = torch.empty(
                self.infer_state.total_token_num,
                dtype=torch.int32,
                device=get_current_device_id(),
            )

        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        self.ks, self.ke, self.lengths = gen_nsa_ks_ke(
            b_seq_len=self.infer_state.b_seq_len,
            b_q_seq_len=self.infer_state.b_q_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            req_to_token_index=self.infer_state.req_manager.req_to_token_indexs,
            q_token_num=self.infer_state.b_seq_len.shape[0],
            ragged_mem_index=self.ragged_mem_index,
            hold_req_idx=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        self.nsa_cache_seqlens = torch.minimum(
            torch.full(size=(self.infer_state.batch_size,), fill_value=2048, dtype=torch.int32, device="cuda"),
            self.infer_state.b_seq_len,
        )
        padded_seq_lens = torch.zeros(size=(self.nsa_cache_seqlens.shape[0] + 1,), dtype=torch.int32, device="cuda")
        # 进行 cumsum 操作
        padded_seq_lens[1:].copy_(self.nsa_cache_seqlens, non_blocking=True)
        self.nsa_cu_seqlens_k_new = padded_seq_lens.cumsum(dim=0, dtype=torch.int32)

    def run(self, inputs: SparseDecodeInput, binding: BoundSparse):
        q, kv = (inputs.q_nope, inputs.q_rope), binding.cache
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = binding.softmax_scale
        kv_lora_rank = binding.value_dim
        qk_rope_head_dim = binding.rope_dim

        q_nope, q_rope = q

        # Extract k_rope and kv_nope from the KV buffer
        k_rope = kv[:, :, -qk_rope_head_dim:].view(-1, 1, 1, qk_rope_head_dim)
        kv_nope = kv[:, :, :-qk_rope_head_dim].view(-1, 1, 1, kv_lora_rank)

        o_tensor = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=topk_mem_indices,
            cache_seqlens=self.nsa_cache_seqlens,
            cu_seqlens_q=self.infer_state.b1_cu_q_seq_len,
            cu_seqlens_k_new=self.nsa_cu_seqlens_k_new,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=softmax_scale,
            causal=True,
        )
        return o_tensor
