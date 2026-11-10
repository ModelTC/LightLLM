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
class NsaFlashMlaSparsePrefillPlan(PreparedPlan):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    # Query-row metadata is consumed by model-specific sparse indexers.  It is
    # kept on the plan so the indexer never reaches into InferStateInfo.
    q_kv_seq_len: torch.Tensor = None
    q_req_idx: torch.Tensor = None
    q_kv_start_loc: torch.Tensor = None


@dataclasses.dataclass
class NsaFlashMlaSparseDecodePlan(PreparedPlan):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    nsa_cache_seqlens: torch.Tensor = None
    nsa_cu_seqlens_k_new: torch.Tensor = None
    q_kv_seq_len: torch.Tensor = None
    q_req_idx: torch.Tensor = None


class NsaFlashMlaSparseAttBackend(BaseAttentionBackend):
    provider = "flashmla_sparse"
    dependency = "flash_mla"
    min_device_capability = (9, 0)
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    cache_formats = frozenset({"None"})
    family = AttentionFamily.SPARSE_MLA

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return NsaFlashMlaSparsePrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return NsaFlashMlaSparseDecodePlan(capacity, capture, workspace)

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
        """Materialize per-query request metadata on the execution plan.

        Prefill positions are supplied by the model's input preparation and
        therefore remain correct for prompt-cache and speculative layouts.
        ``output_size`` keeps the operation graph-capturable without a host
        read of the GPU-side query lengths.
        """
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
        return self._prefill_nsa_prefill_att(layer, plan, q=q, kv=k, inputs=inputs)

    def _prefill_nsa_prefill_att(self, layer, plan, q: torch.Tensor, kv: torch.Tensor, inputs) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = layer.params.softmax_scale
        kv_lora_rank = layer.params.kv_lora_rank
        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        mla_out, _, _ = flash_mla_sparse_fwd(
            q=q, kv=kv, indices=topk_mem_indices, sm_scale=softmax_scale, d_v=kv_lora_rank
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
        plan.nsa_cache_seqlens = torch.minimum(
            torch.full(size=(plan.batch.batch_size,), fill_value=2048, dtype=torch.int32, device="cuda"),
            plan.batch.b_seq_len,
        )
        padded_seq_lens = torch.zeros(size=(plan.nsa_cache_seqlens.shape[0] + 1,), dtype=torch.int32, device="cuda")
        padded_seq_lens[1:].copy_(plan.nsa_cache_seqlens, non_blocking=True)
        plan.nsa_cu_seqlens_k_new = padded_seq_lens.cumsum(dim=0, dtype=torch.int32)

    def _forward_decode(self, layer: LayerBinding, plan, inputs: SparseMLADecodeInputs, *, out=None) -> torch.Tensor:
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.params.kv
        return self._decode_nsa_decode_att(layer, plan, q=q, kv=k, inputs=inputs)

    def _decode_nsa_decode_att(
        self, layer, plan, q: Tuple[torch.Tensor, torch.Tensor], kv: torch.Tensor, inputs
    ) -> torch.Tensor:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = layer.params.softmax_scale
        kv_lora_rank = layer.params.kv_lora_rank
        qk_rope_head_dim = layer.params.qk_rope_head_dim
        q_nope, q_rope = q
        k_rope = kv[:, :, -qk_rope_head_dim:].view(-1, 1, 1, qk_rope_head_dim)
        kv_nope = kv[:, :, :-qk_rope_head_dim].view(-1, 1, 1, kv_lora_rank)
        o_tensor = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=topk_mem_indices,
            cache_seqlens=plan.nsa_cache_seqlens,
            cu_seqlens_q=plan.batch.b1_cu_q_seq_len,
            cu_seqlens_k_new=plan.nsa_cu_seqlens_k_new,
            max_seqlen_q=plan.batch.max_q_seq_len,
            softmax_scale=softmax_scale,
            causal=True,
        )
        return o_tensor
