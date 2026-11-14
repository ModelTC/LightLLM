from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses
from typing import Tuple

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import SparseMLADecodeInputs, SparseMLAPrefillInputs
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend
from lightllm.utils.dist_utils import get_current_device_id


@dataclasses.dataclass(slots=True)
class NsaFlashMlaSparsePrefillContext(AttentionContext):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class NsaFlashMlaSparseDecodeContext(AttentionContext):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    nsa_cache_seqlens: torch.Tensor = None
    nsa_cu_seqlens_k_new: torch.Tensor = None


class FlashMLASparseMLACommon:
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    provider = "flashmla_sparse"

    dependency = "flash_mla"

    min_device_capability = (9, 0)

    activation_dtypes = frozenset({torch.float16, torch.bfloat16})

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    family = AttentionFamily.SPARSE_MLA


class FlashMLASparseMLAPrefillBackend(
    FlashMLASparseMLACommon, AttentionBackend[NsaFlashMlaSparsePrefillContext, SparseMLAPrefillInputs, torch.Tensor]
):
    phase = AttentionPhase.PREFILL

    def _run_prefill(self, layer, context, q: torch.Tensor, kv: torch.Tensor, inputs) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = self.spec.softmax_scale
        kv_lora_rank = layer.resources.kv_lora_rank
        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        mla_out, _, _ = flash_mla_sparse_fwd(
            q=q, kv=kv, indices=topk_mem_indices, sm_scale=softmax_scale, d_v=kv_lora_rank
        )
        return mla_out

    def allocate_context(self, capacity, *, capture, workspace) -> NsaFlashMlaSparsePrefillContext:
        return NsaFlashMlaSparsePrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: NsaFlashMlaSparsePrefillContext):
        batch = context.batch
        context.ragged_mem_index = torch.empty(batch.total_token_num, dtype=torch.int32, device=get_current_device_id())
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        context.ks, context.ke, context.lengths = gen_nsa_ks_ke(
            b_seq_len=batch.b_seq_len,
            b_q_seq_len=batch.b_q_seq_len,
            b_req_idx=batch.b_req_idx,
            req_to_token_index=batch.req_to_token_indexs,
            q_token_num=batch.token_num,
            ragged_mem_index=context.ragged_mem_index,
            hold_req_idx=batch.hold_req_id,
        )
        return

    def forward(
        self,
        layer: LayerAttention,
        context: NsaFlashMlaSparsePrefillContext,
        inputs: SparseMLAPrefillInputs,
        *,
        out=None,
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.kv
        return self._run_prefill(layer, context, q=q, kv=k, inputs=inputs)


class FlashMLASparseMLADecodeBackend(
    FlashMLASparseMLACommon, AttentionBackend[NsaFlashMlaSparseDecodeContext, SparseMLADecodeInputs, torch.Tensor]
):
    phase = AttentionPhase.DECODE

    def _run_decode(
        self, layer, context, q: Tuple[torch.Tensor, torch.Tensor], kv: torch.Tensor, inputs
    ) -> torch.Tensor:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = self.spec.softmax_scale
        kv_lora_rank = layer.resources.kv_lora_rank
        qk_rope_head_dim = layer.resources.qk_rope_head_dim
        q_nope, q_rope = q
        k_rope = kv[:, :, -qk_rope_head_dim:].view(-1, 1, 1, qk_rope_head_dim)
        kv_nope = kv[:, :, :-qk_rope_head_dim].view(-1, 1, 1, kv_lora_rank)
        o_tensor = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=topk_mem_indices,
            cache_seqlens=context.nsa_cache_seqlens,
            cu_seqlens_q=context.batch.b1_cu_q_seq_len,
            cu_seqlens_k_new=context.nsa_cu_seqlens_k_new,
            max_seqlen_q=context.batch.max_q_seq_len,
            softmax_scale=softmax_scale,
            causal=True,
        )
        return o_tensor

    def allocate_context(self, capacity, *, capture, workspace) -> NsaFlashMlaSparseDecodeContext:
        return NsaFlashMlaSparseDecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: NsaFlashMlaSparseDecodeContext):
        context.ragged_mem_index = context.workspace.scratch(
            "sparse_ragged", max(1, context.capacity.batch_size * context.capacity.max_kv_seq_len), torch.int32
        )
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        context.ks, context.ke, context.lengths = gen_nsa_ks_ke(
            b_seq_len=context.batch.b_seq_len,
            b_q_seq_len=context.batch.b_q_seq_len,
            b_req_idx=context.batch.b_req_idx,
            req_to_token_index=context.batch.req_to_token_indexs,
            q_token_num=context.batch.b_seq_len.shape[0],
            ragged_mem_index=context.ragged_mem_index,
            hold_req_idx=context.batch.hold_req_id,
        )
        context.nsa_cache_seqlens = torch.minimum(
            torch.full(size=(context.batch.batch_size,), fill_value=2048, dtype=torch.int32, device="cuda"),
            context.batch.b_seq_len,
        )
        padded_seq_lens = torch.zeros(size=(context.nsa_cache_seqlens.shape[0] + 1,), dtype=torch.int32, device="cuda")
        padded_seq_lens[1:].copy_(context.nsa_cache_seqlens, non_blocking=True)
        context.nsa_cu_seqlens_k_new = padded_seq_lens.cumsum(dim=0, dtype=torch.int32)

    def forward(
        self, layer: LayerAttention, context: NsaFlashMlaSparseDecodeContext, inputs: SparseMLADecodeInputs, *, out=None
    ) -> torch.Tensor:
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.resources.kv
        return self._run_decode(layer, context, q=q, kv=k, inputs=inputs)
