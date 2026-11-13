import dataclasses

import torch

from lightllm.common.attention.api import AttentionPhase
from lightllm.common.attention.api.execution import BackendResources, WorkspaceRequirement
from lightllm.common.attention.api.inputs import SparseMLAPrefillInputs
from lightllm.common.attention.backends.sparse_mla.flashmla.fp8 import (
    NsaFlashMlaFp8SparseAttBackend,
    NsaFlashMlaFp8SparseDecodeContext,
)

from .sparse_mla import Glm5NextSparsePrefillContext


@dataclasses.dataclass(slots=True)
class Fp8Glm5NextSparsePrefillContext(Glm5NextSparsePrefillContext):
    smallq_workspace: object = None
    generic_workspace: object = None
    generic_binding: object = None


@dataclasses.dataclass(slots=True)
class Fp8Glm5NextSparseDecodeContext(NsaFlashMlaFp8SparseDecodeContext):
    decode_topk_bound: int = None


class Fp8Glm5NextSparseAttBackend(NsaFlashMlaFp8SparseAttBackend):
    provider = "flashmla_sparse"
    cache_formats = frozenset({"fp8kv_dsa"})

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self.prefill_graph_enabled = model.args.enable_prefill_cudagraph
        self.prefill_token_capacity = model.batch_max_tokens or 0
        self.decode_only = model.args.run_mode == "decode"

    def resource_requirements(self, capacity, *, capture):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import KV_GATHER_CHUNK_SIZE

        if capacity.phase is not AttentionPhase.PREFILL:
            return super().resource_requirements(capacity, capture=capture)
        tokens = max(1, self.prefill_token_capacity, capacity.token_num)
        kv_tokens = tokens if self.decode_only else capacity.batch_size * capacity.max_kv_seq_len + tokens
        gather_tokens = min(KV_GATHER_CHUNK_SIZE, kv_tokens)
        topk = (self.network_config["index_topk"] + 3 + 127) // 128 * 128
        return BackendResources(
            workspace=(
                WorkspaceRequirement("glm_gathered_kv", gather_tokens * 512, torch.bfloat16),
                WorkspaceRequirement("glm_chunk_indices", tokens * topk, torch.int32),
                WorkspaceRequirement("glm_chunk_lengths", tokens, torch.int32),
                WorkspaceRequirement("glm_cu_q", tokens + 1, torch.int32),
            )
        )

    def get_smallq_workspace(self, context, q):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            allocate_direct_smallq_workspace,
            allocate_selected_prefill_workspace,
        )

        context.smallq_workspace = context.workspace.get_object(
            "glm_smallq",
            lambda: {
                "direct": allocate_direct_smallq_workspace(q[:1]),
                "selected": allocate_selected_prefill_workspace(q, batch_capacity=8),
            },
            expected_type=dict,
        )
        if q.shape[0] == 1:
            return context.smallq_workspace["direct"]
        batch = q.shape[0]
        selected = context.smallq_workspace["selected"]
        slots = selected["compact"].shape[1]
        return {
            "compact": selected["compact"][:batch],
            "lengths": selected["lengths"][:batch],
            "page_table": selected["page_table"][:batch],
            "cu_q": selected["cu_q"][: batch + 1],
            "cu_k": selected["cu_k"][: batch + 1],
            "selected_kv": selected["selected_kv"][: batch * slots],
        }

    def get_generic_prefill_workspace(self, context, q, packed_kv, topk_indices):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import KV_GATHER_CHUNK_SIZE

        rows, columns = topk_indices.shape
        workspace = context.workspace
        tokens = max(self.prefill_token_capacity, rows)
        binding = (rows, columns, topk_indices.dtype, packed_kv.shape[0])
        if context.generic_workspace is None or context.generic_binding != binding:
            row_table = workspace.get("glm_prefill_rows", packed_kv.shape[0], torch.int32)[: packed_kv.shape[0]]
            row_table.fill_(-1)
            cu_q = workspace.get("glm_cu_q", tokens + 1, torch.int32)[: rows + 1]
            torch.arange(rows + 1, device=q.device, dtype=torch.int32, out=cu_q)
            context.generic_workspace = {
                "prefill_row_table": row_table,
                "chunk_indices": workspace.get("glm_chunk_indices", tokens * columns, topk_indices.dtype)[
                    : rows * columns
                ].view(rows, columns),
                "chunk_lengths": workspace.get("glm_chunk_lengths", tokens, torch.int32)[:rows],
                "cu_seqlens_q": cu_q,
                "row_table_built": False,
            }
            context.generic_binding = binding
        gather_tokens = min(KV_GATHER_CHUNK_SIZE, context.ragged_mem_index.numel())
        if self.decode_only:
            gather_tokens = min(gather_tokens, max(1, self.prefill_token_capacity, rows))
        context.generic_workspace["gathered_kv"] = workspace.get(
            "glm_gathered_kv", max(1, gather_tokens) * 512, q.dtype
        )[: gather_tokens * 512].view(gather_tokens, 1, 512)
        context.generic_workspace["prefill_row_table"].fill_(-1)
        context.generic_workspace["row_table_built"] = False
        return context.generic_workspace

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return Fp8Glm5NextSparsePrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return Fp8Glm5NextSparseDecodeContext(
            capacity,
            capture,
            workspace,
            decode_topk_bound=2051
            if self.network_config["index_topk"] == 2048 and self.network_config["index_kpool"] == 4
            else None,
        )

    def _run_prefill(self, layer, context, q, kv, inputs):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import context_attention_fwd_fp8

        if q.shape[1] != 64:
            return context_attention_fwd_fp8(
                q,
                kv,
                context.ragged_mem_index,
                context.batch.mem_index,
                inputs.prefill_cache_kv,
                context.ks,
                inputs.topk_indices,
                layer.spec.softmax_scale,
                generic_workspace=self.get_generic_prefill_workspace(context, q, kv, inputs.topk_indices),
            )
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            can_use_direct_smallq,
            can_use_selected_prefill,
        )

        direct_workspace = (
            self.get_smallq_workspace(context, q)
            if can_use_direct_smallq(
                q, kv, context.ragged_mem_index, context.batch.mem_index, inputs.prefill_cache_kv, inputs.topk_indices
            )
            else None
        )
        selected_eligible = direct_workspace is None and can_use_selected_prefill(
            q, kv, context.ragged_mem_index, context.batch.mem_index, inputs.prefill_cache_kv, inputs.topk_indices
        )
        workspace = self.get_smallq_workspace(context, q) if selected_eligible else None
        generic_workspace = None
        if direct_workspace is None and (not selected_eligible):
            generic_workspace = self.get_generic_prefill_workspace(context, q, kv, inputs.topk_indices)
        return context_attention_fwd_fp8(
            q,
            kv,
            context.ragged_mem_index,
            context.batch.mem_index,
            inputs.prefill_cache_kv,
            context.ks,
            inputs.topk_indices,
            layer.spec.softmax_scale,
            workspace,
            direct_workspace,
            generic_workspace,
        )

    def _run_decode(self, layer, context, q, packed_kv, inputs):
        import flash_mla

        from lightllm.models.glm5_next.triton_kernel import fp8_sparse_decode
        from lightllm.models.glm5_next.triton_kernel.pad_query_heads import pad_query_heads

        topk_mem_indices = inputs.topk_mem_indices
        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        assert topk_mem_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"
        q_nope, _ = q
        num_heads = q_nope.shape[1]
        if fp8_sparse_decode.can_use_fp8_sparse_direct_decode(
            q_nope,
            packed_kv,
            topk_mem_indices,
            layer.params.kv_lora_rank,
            context.decode_topk_bound,
            is_cuda_graph_capture=context.capture,
        ):
            return fp8_sparse_decode.fp8_sparse_direct_decode(
                q_nope, packed_kv, topk_mem_indices, layer.spec.softmax_scale
            )
        padded_q = pad_query_heads(q_nope, is_cuda_graph_capture=context.capture)
        kv_cache = torch.as_strided(
            packed_kv,
            size=(packed_kv.shape[0], 1, 1, packed_kv.shape[-1]),
            stride=(packed_kv.stride(0), packed_kv.shape[-1], packed_kv.shape[-1], packed_kv.stride(-1)),
        )
        output, _ = flash_mla.flash_mla_with_kvcache(
            q=padded_q,
            k_cache=kv_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=layer.params.kv_lora_rank,
            tile_scheduler_metadata=(
                context.capture_flashmla_sched_meta
                if torch.cuda.is_current_stream_capturing()
                else context.flashmla_sched_meta
            ),
            softmax_scale=layer.spec.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return output[:, 0, :num_heads, :]

    def _forward_prefill(self, layer, context, inputs: SparseMLAPrefillInputs, *, out=None):
        return self._run_prefill(layer, context, inputs.query, layer.params.kv, inputs)
