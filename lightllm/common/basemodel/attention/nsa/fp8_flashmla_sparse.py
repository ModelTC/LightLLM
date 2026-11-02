from ..sparse import BoundSparse, SparsePrefillInput, SparseDecodeInput
import dataclasses
import weakref
import torch
from typing import TYPE_CHECKING, Tuple

from ..base_att import BaseAttState
from .flashmla_sparse import NsaFlashMlaSparseAttBackend
from lightllm.utils.dist_utils import get_current_device_id

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class NsaFlashMlaFp8SparseAttBackend(NsaFlashMlaSparseAttBackend):
    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8SparsePrefillAttState":
        return NsaFlashMlaFp8SparsePrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8SparseDecodeAttState":
        return NsaFlashMlaFp8SparseDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class NsaFlashMlaFp8SparsePrefillAttState(BaseAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8SparseAttBackend = self.backend
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
        q, packed_kv = inputs.q, binding.cache
        import flash_mla

        topk_indices = inputs.topk_indices
        softmax_scale = binding.softmax_scale
        kv_lora_rank = binding.value_dim
        topk_mem_indices = inputs.topk_mem_indices
        prefill_cache_kv = inputs.new_kv

        if self.infer_state.max_cache_len > 0:
            # 当前推理生成的token kv部分从 prefill_cache_kv 中获取，历史
            # 部分kv 从 packed_kv 中获取, 并进行反量化，这样可以避免 prefill_cache_kv
            # 部分的数据进行重复的反量化操作，提升整体的性能。
            from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                get_prefill_kv_cache_and_remap_indices_triton,
            )

            kv, topk_indices = get_prefill_kv_cache_and_remap_indices_triton(
                packed_kv=packed_kv,
                topk_mem_indices=topk_mem_indices,
                prefill_mem_index=self.infer_state.mem_index,
                prefill_cache_kv=prefill_cache_kv,
                prefill_dtype=prefill_cache_kv.dtype,
            )
        else:
            kv = prefill_cache_kv

        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)

        mla_out, _, _ = flash_mla.flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=topk_indices,
            sm_scale=softmax_scale,
            d_v=kv_lora_rank,
        )
        return mla_out


@dataclasses.dataclass
class NsaFlashMlaFp8SparseDecodeAttState(BaseAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    flashmla_sched_meta: object = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8SparseAttBackend = self.backend
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
        import flash_mla

        self.flashmla_sched_meta, _ = flash_mla.get_mla_metadata()
        return

    def run(self, inputs: SparseDecodeInput, binding: BoundSparse):
        q, packed_kv = (inputs.q_nope, inputs.q_rope), binding.cache
        import flash_mla

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = binding.softmax_scale
        kv_lora_rank = binding.value_dim

        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        assert topk_mem_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"

        q_nope, q_rope = q
        q_all = torch.cat([q_nope, q_rope], dim=-1).unsqueeze(1).contiguous()
        kv = torch.as_strided(
            packed_kv,
            size=(packed_kv.shape[0], 1, 1, packed_kv.shape[-1]),
            stride=(packed_kv.stride(0), packed_kv.shape[-1], packed_kv.shape[-1], packed_kv.stride(-1)),
        )

        o_tensor, _ = flash_mla.flash_mla_with_kvcache(
            q=q_all,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=kv_lora_rank,
            tile_scheduler_metadata=self.flashmla_sched_meta,
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return o_tensor[:, 0, :, :]  # [b, 1, h, d] -> [b, h, d]
