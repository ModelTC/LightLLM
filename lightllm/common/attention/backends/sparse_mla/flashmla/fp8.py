from lightllm.common.attention.backends.base import BaseAttentionBackend
from lightllm.common.attention.api.spec import AttentionFamily
import dataclasses
import torch
from typing import Tuple
from lightllm.utils.dist_utils import get_current_device_id

from lightllm.common.attention.backends.base import PreparedPlan
from lightllm.common.attention.api.backend import LayerBinding
from lightllm.common.attention.api.inputs import SparseMLAPrefillInputs, SparseMLADecodeInputs


@dataclasses.dataclass
class NsaFlashMlaFp8SparsePrefillPlan(PreparedPlan):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    q_kv_seq_len: torch.Tensor = None
    q_req_idx: torch.Tensor = None
    q_kv_start_loc: torch.Tensor = None


@dataclasses.dataclass
class NsaFlashMlaFp8SparseDecodePlan(PreparedPlan):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    flashmla_sched_meta: object = None
    capture_flashmla_sched_meta: object = None
    q_kv_seq_len: torch.Tensor = None
    q_req_idx: torch.Tensor = None


class NsaFlashMlaFp8SparseAttBackend(BaseAttentionBackend):
    provider = "flashmla_sparse"
    cache_formats = frozenset({"fp8kv_dsa"})
    family = AttentionFamily.SPARSE_MLA

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return NsaFlashMlaFp8SparsePrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return NsaFlashMlaFp8SparseDecodePlan(capacity, capture, workspace)

    def _update_prefill(self, plan):
        batch = plan.batch
        plan.ragged_mem_index = torch.empty(
            batch.total_token_num, dtype=torch.int32, device=get_current_device_id()
        )
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        plan.ks, plan.ke, plan.lengths = gen_nsa_ks_ke(
            b_seq_len=batch.b_seq_len,
            b_q_seq_len=batch.b_q_seq_len,
            b_req_idx=batch.b_req_idx,
            req_to_token_index=batch.req_to_token_indexs,
            q_token_num=batch.token_num,
            ragged_mem_index=plan.ragged_mem_index,
            hold_req_idx=batch.hold_req_id,
        )
        self._update_query_metadata(plan)
        return

    @staticmethod
    def _update_query_metadata(plan):
        batch = plan.batch
        if batch.position_ids is None:
            raise ValueError("sparse MLA prefill requires AttentionBatch.position_ids")
        plan.q_kv_seq_len = batch.position_ids + 1
        plan.q_req_idx = torch.repeat_interleave(
            batch.b_req_idx, batch.b_q_seq_len, output_size=batch.token_num
        )
        plan.q_kv_start_loc = torch.repeat_interleave(
            batch.b1_cu_kv_seq_len[:-1], batch.b_q_seq_len, output_size=batch.token_num
        )

    def _forward_prefill(self, layer: LayerBinding, plan, inputs: SparseMLAPrefillInputs, *, out=None) -> torch.Tensor:
        q = inputs.query
        k = layer.params.kv
        return self._prefill_nsa_prefill_att(layer, plan, q=q, packed_kv=k, inputs=inputs)

    def _prefill_nsa_prefill_att(self, layer, plan, q: torch.Tensor, packed_kv: torch.Tensor, inputs) -> torch.Tensor:
        import flash_mla

        topk_indices = inputs.topk_indices
        softmax_scale = layer.params.softmax_scale
        kv_lora_rank = layer.params.kv_lora_rank
        topk_mem_indices = inputs.topk_mem_indices
        prefill_cache_kv = inputs.prefill_cache_kv
        if plan.batch.max_cache_len > 0:
            from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                get_prefill_kv_cache_and_remap_indices_triton,
            )

            kv, topk_indices = get_prefill_kv_cache_and_remap_indices_triton(
                packed_kv=packed_kv,
                topk_mem_indices=topk_mem_indices,
                prefill_mem_index=plan.batch.mem_index,
                prefill_cache_kv=prefill_cache_kv,
                prefill_dtype=prefill_cache_kv.dtype,
            )
        else:
            kv = prefill_cache_kv
        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)
        mla_out, _, _ = flash_mla.flash_mla_sparse_fwd(
            q=q, kv=kv, indices=topk_indices, sm_scale=softmax_scale, d_v=kv_lora_rank
        )
        return mla_out

    def _update_decode(self, plan):
        plan.q_kv_seq_len = plan.batch.b_seq_len
        plan.q_req_idx = plan.batch.b_req_idx
        plan.ragged_mem_index = plan.workspace.get(
            "sparse_ragged", max(1, plan.capacity.batch_size * plan.capacity.max_kv_seq_len), torch.int32
        )
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        plan.ks, plan.ke, plan.lengths = gen_nsa_ks_ke(
            b_seq_len=plan.batch.b_seq_len,
            b_q_seq_len=plan.batch.b_q_seq_len,
            b_req_idx=plan.batch.b_req_idx,
            req_to_token_index=plan.batch.req_to_token_indexs,
            q_token_num=plan.batch.b_seq_len.shape[0],
            ragged_mem_index=plan.ragged_mem_index,
            hold_req_idx=plan.batch.hold_req_id,
        )
        return

    def _forward_decode(self, layer: LayerBinding, plan, inputs: SparseMLADecodeInputs, *, out=None) -> torch.Tensor:
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.params.kv
        return self._decode_nsa_decode_att(layer, plan, q=q, packed_kv=k, inputs=inputs)

    def _decode_nsa_decode_att(
        self, layer, plan, q: Tuple[torch.Tensor, torch.Tensor], packed_kv: torch.Tensor, inputs
    ) -> torch.Tensor:
        import flash_mla

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = layer.params.softmax_scale
        kv_lora_rank = layer.params.kv_lora_rank
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
            tile_scheduler_metadata=(
                plan.capture_flashmla_sched_meta
                if torch.cuda.is_current_stream_capturing()
                else plan.flashmla_sched_meta
            ),
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return o_tensor[:, 0, :, :]

    def _prepare_decode(self, plan):
        import flash_mla

        if plan.flashmla_sched_meta is None:
            plan.flashmla_sched_meta = flash_mla.get_mla_metadata()[0]
            plan.capture_flashmla_sched_meta = flash_mla.get_mla_metadata()[0]
