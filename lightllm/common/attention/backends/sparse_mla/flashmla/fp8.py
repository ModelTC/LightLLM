from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses
from typing import Tuple

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import SparseMLADecodeInputs, SparseMLAPrefillInputs
from lightllm.common.attention.api.backend import AttentionContext
from lightllm.common.attention.backends.sparse_mla.flashmla.backend import (
    FlashMLASparseMLAPrefillBackend,
    FlashMLASparseMLADecodeBackend,
)


@dataclasses.dataclass(slots=True)
class NsaFlashMlaFp8SparsePrefillContext(AttentionContext):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class NsaFlashMlaFp8SparseDecodeContext(AttentionContext):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    flashmla_sched_meta: object = None
    capture_flashmla_sched_meta: object = None


class FlashMLASparseMLAFp8Common:
    family = AttentionFamily.SPARSE_MLA
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    cache_encodings = frozenset({CacheEncoding.FP8_SPARSE_MLA})


class FlashMLASparseMLAFp8PrefillBackend(FlashMLASparseMLAFp8Common, FlashMLASparseMLAPrefillBackend):
    phase = AttentionPhase.PREFILL

    def _run_prefill(self, layer, context, q: torch.Tensor, packed_kv: torch.Tensor, inputs) -> torch.Tensor:
        import flash_mla

        topk_indices = inputs.topk_indices
        softmax_scale = self.spec.softmax_scale
        kv_lora_rank = layer.resources.kv_lora_rank
        topk_mem_indices = inputs.topk_mem_indices
        prefill_cache_kv = inputs.prefill_cache_kv
        if context.batch.max_cache_len > 0:
            from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                get_prefill_kv_cache_and_remap_indices_triton,
            )

            kv, topk_indices = get_prefill_kv_cache_and_remap_indices_triton(
                packed_kv=packed_kv,
                topk_mem_indices=topk_mem_indices,
                prefill_mem_index=context.batch.mem_index,
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

    def allocate_context(self, capacity, *, capture, workspace) -> NsaFlashMlaFp8SparsePrefillContext:
        return NsaFlashMlaFp8SparsePrefillContext(capacity, capture, workspace)

    def forward(
        self,
        layer: LayerAttention,
        context: NsaFlashMlaFp8SparsePrefillContext,
        inputs: SparseMLAPrefillInputs,
        *,
        out=None,
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.kv
        return self._run_prefill(layer, context, q=q, packed_kv=k, inputs=inputs)


class FlashMLASparseMLAFp8DecodeBackend(FlashMLASparseMLAFp8Common, FlashMLASparseMLADecodeBackend):
    phase = AttentionPhase.DECODE

    def _run_decode(
        self, layer, context, q: Tuple[torch.Tensor, torch.Tensor], packed_kv: torch.Tensor, inputs
    ) -> torch.Tensor:
        import flash_mla

        topk_mem_indices = inputs.topk_mem_indices
        softmax_scale = self.spec.softmax_scale
        kv_lora_rank = layer.resources.kv_lora_rank
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
                context.capture_flashmla_sched_meta
                if torch.cuda.is_current_stream_capturing()
                else context.flashmla_sched_meta
            ),
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return o_tensor[:, 0, :, :]

    def allocate_context(self, capacity, *, capture, workspace) -> NsaFlashMlaFp8SparseDecodeContext:
        return NsaFlashMlaFp8SparseDecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: NsaFlashMlaFp8SparseDecodeContext):
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
        return

    def forward(
        self,
        layer: LayerAttention,
        context: NsaFlashMlaFp8SparseDecodeContext,
        inputs: SparseMLADecodeInputs,
        *,
        out=None,
    ) -> torch.Tensor:
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.resources.kv
        return self._run_decode(layer, context, q=q, packed_kv=k, inputs=inputs)

    def plan_host(self, context: NsaFlashMlaFp8SparseDecodeContext):
        import flash_mla

        if context.flashmla_sched_meta is None:
            context.flashmla_sched_meta = flash_mla.get_mla_metadata()[0]
            context.capture_flashmla_sched_meta = flash_mla.get_mla_metadata()[0]
