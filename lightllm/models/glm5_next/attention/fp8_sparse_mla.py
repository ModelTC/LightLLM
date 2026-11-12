import dataclasses

import torch

from lightllm.common.attention.api.inputs import SparseMLAPrefillInputs
from lightllm.common.attention.backends.sparse_mla.flashmla.fp8 import (
    NsaFlashMlaFp8SparseAttBackend,
    NsaFlashMlaFp8SparseDecodePlan,
)

from .sparse_mla import Glm5NextSparsePrefillPlan


@dataclasses.dataclass
class Fp8Glm5NextSparsePrefillPlan(Glm5NextSparsePrefillPlan):
    smallq_workspace: object = None
    generic_workspace: object = None
    generic_binding: object = None


@dataclasses.dataclass
class Fp8Glm5NextSparseDecodePlan(NsaFlashMlaFp8SparseDecodePlan):
    decode_topk_bound: int = None


class Fp8Glm5NextSparseAttBackend(NsaFlashMlaFp8SparseAttBackend):
    provider = "flashmla_sparse"
    cache_formats = frozenset({"fp8kv_dsa"})

    def __init__(self, model):
        super().__init__(model)
        self.prefill_graph_enabled = model.args.enable_prefill_cudagraph
        self.prefill_token_capacity = model.batch_max_tokens or 0

    def get_smallq_workspace(self, plan, q):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            allocate_direct_smallq_workspace,
            allocate_selected_prefill_workspace,
        )

        slot = plan.workspace.objects
        if "glm_smallq" not in slot:
            slot["glm_smallq"] = {
                "direct": allocate_direct_smallq_workspace(q[:1]),
                "selected": allocate_selected_prefill_workspace(q, batch_capacity=8),
            }
        plan.smallq_workspace = slot["glm_smallq"]
        if q.shape[0] == 1:
            return plan.smallq_workspace["direct"]
        batch = q.shape[0]
        selected = plan.smallq_workspace["selected"]
        slots = selected["compact"].shape[1]
        return {
            "compact": selected["compact"][:batch],
            "lengths": selected["lengths"][:batch],
            "page_table": selected["page_table"][:batch],
            "cu_q": selected["cu_q"][: batch + 1],
            "cu_k": selected["cu_k"][: batch + 1],
            "selected_kv": selected["selected_kv"][: batch * slots],
        }

    def get_generic_prefill_workspace(self, plan, q, packed_kv, topk_indices):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            allocate_generic_prefill_workspace,
        )

        rows, columns = topk_indices.shape
        buffers = plan.workspace.objects.get("glm_generic")
        if (
            buffers is None
            or buffers["chunk_indices"].shape[0] < rows
            or buffers["chunk_indices"].shape[1] != columns
            or buffers["chunk_indices"].dtype != topk_indices.dtype
            or buffers["prefill_row_table"].numel() < packed_kv.shape[0]
        ):
            buffers = allocate_generic_prefill_workspace(
                q, packed_kv, topk_indices, query_capacity=max(self.prefill_token_capacity, rows)
            )
            plan.workspace.objects["glm_generic"] = buffers
        binding = (columns, topk_indices.dtype, packed_kv.shape[0], id(buffers))
        if plan.generic_workspace is None or plan.generic_binding != binding:
            buffers["prefill_row_table"].fill_(-1)
            plan.generic_workspace = {
                "prefill_row_table": buffers["prefill_row_table"][: packed_kv.shape[0]],
                "chunk_indices": buffers["chunk_indices"][:rows],
                "chunk_lengths": buffers["chunk_lengths"][:rows],
                "cu_seqlens_q": buffers["cu_seqlens_q"][: rows + 1],
                "row_table_built": False,
            }
            plan.generic_binding = binding
        return plan.generic_workspace

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return Fp8Glm5NextSparsePrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return Fp8Glm5NextSparseDecodePlan(
            capacity,
            capture,
            workspace,
            decode_topk_bound=2051
            if self.network_config["index_topk"] == 2048 and self.network_config["index_kpool"] == 4
            else None,
        )

    def _prefill_nsa_prefill_att(self, layer, plan, q, kv, inputs):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import context_attention_fwd_fp8

        if q.shape[1] != 64:
            return context_attention_fwd_fp8(
                q,
                kv,
                plan.ragged_mem_index,
                plan.batch.mem_index,
                inputs.prefill_cache_kv,
                plan.ks,
                inputs.topk_indices,
                layer.params.softmax_scale,
            )
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            can_use_direct_smallq,
            can_use_selected_prefill,
        )

        direct_workspace = (
            self.get_smallq_workspace(plan, q)
            if can_use_direct_smallq(
                q, kv, plan.ragged_mem_index, plan.batch.mem_index, inputs.prefill_cache_kv, inputs.topk_indices
            )
            else None
        )
        selected_eligible = direct_workspace is None and can_use_selected_prefill(
            q, kv, plan.ragged_mem_index, plan.batch.mem_index, inputs.prefill_cache_kv, inputs.topk_indices
        )
        workspace = self.get_smallq_workspace(plan, q) if selected_eligible else None
        generic_workspace = None
        if direct_workspace is None and (not selected_eligible):
            graph_guard = plan.capture or self.prefill_graph_enabled
            if not graph_guard and (not torch.cuda.is_current_stream_capturing()):
                generic_workspace = self.get_generic_prefill_workspace(plan, q, kv, inputs.topk_indices)
        return context_attention_fwd_fp8(
            q,
            kv,
            plan.ragged_mem_index,
            plan.batch.mem_index,
            inputs.prefill_cache_kv,
            plan.ks,
            inputs.topk_indices,
            layer.params.softmax_scale,
            workspace,
            direct_workspace,
            generic_workspace,
        )

    def _decode_nsa_decode_att(self, layer, plan, q, packed_kv, inputs):
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
            plan.decode_topk_bound,
            is_cuda_graph_capture=plan.capture,
        ):
            return fp8_sparse_decode.fp8_sparse_direct_decode(
                q_nope, packed_kv, topk_mem_indices, layer.params.softmax_scale
            )
        padded_q = pad_query_heads(q_nope, is_cuda_graph_capture=plan.capture)
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
                plan.capture_flashmla_sched_meta
                if torch.cuda.is_current_stream_capturing()
                else plan.flashmla_sched_meta
            ),
            softmax_scale=layer.params.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return output[:, 0, :num_heads, :]

    def _forward_prefill(self, layer, plan, inputs: SparseMLAPrefillInputs, *, out=None):
        return self._prefill_nsa_prefill_att(layer, plan, inputs.query, layer.params.kv, inputs)
