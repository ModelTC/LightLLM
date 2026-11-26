from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton

from lightllm.common.attention.backend import AttentionBackend, AttentionBuildResources, TensorAllocator
from lightllm.common.basemodel.infer_struct import InferStateInfo
from .. import set_flashinfer_envs
from . import MLADecodeLayerConfig, MLADecodeLayerInputs, MLAPrefillLayerConfig, MLAPrefillLayerInputs

if TYPE_CHECKING:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

# FlashInfer MLA's workspace has a fixed size, separate from the GQA one (--flashinfer_workspace_size).
_WORKSPACE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class FlashInferMLADecodeAttState:
    """A planned paged MLA wrapper with one query row per request; page index p addresses cache slots
    [p * page_size, (p + 1) * page_size)."""

    wrapper: "BatchMLAPagedAttentionWrapper"
    page_size: int


def _workspace(resources: AttentionBuildResources) -> torch.Tensor:
    # Allocated on first use, so prefill-only and decode-only nodes create it only when needed.
    return resources.get_workspace("flashinfer_mla", _WORKSPACE_BYTES, torch.int8)


class FlashInferMLAPrefillBackend(
    AttentionBackend[MLAPrefillLayerConfig, "BatchPrefillWithRaggedKVCacheWrapper", MLAPrefillLayerInputs]
):
    """The attention state is a causal ragged-KV wrapper planned over the step's expanded keys and values."""

    name = "flashinfer_mla_prefill"

    @classmethod
    def _check_supported(cls, layer_config: MLAPrefillLayerConfig) -> None:
        if not layer_config.causal:
            raise NotImplementedError("FlashInfer MLA prefill supports only causal attention")

    @classmethod
    def group_key(cls, layer_config: MLAPrefillLayerConfig):
        # FlashInfer plans with every field but causal, which the backend requires.
        return layer_config

    def setup(self, resources: AttentionBuildResources) -> None:
        set_flashinfer_envs()
        self.resources = resources

    def forward(
        self,
        att_state: "BatchPrefillWithRaggedKVCacheWrapper",
        layer_inputs: MLAPrefillLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [query_tokens, num_heads, v_head_dim]; keys are expanded into alloc_func scratch."""
        layer_config = self.layer_config
        q, k_nope = layer_inputs.q, layer_inputs.k_nope
        nope_dim, heads = layer_config.qk_nope_head_dim, layer_config.num_heads
        k = alloc_func((k_nope.shape[0], heads, q.shape[-1]), dtype=q.dtype, device=q.device)
        k[:, :, :nope_dim].copy_(k_nope)
        k[:, :, nope_dim:].copy_(layer_inputs.k_rope.expand(-1, heads, -1))
        att_state.run(q, k, layer_inputs.v, out=out)
        return out

    def build(self, infer_state: InferStateInfo) -> "BatchPrefillWithRaggedKVCacheWrapper":
        """Plan a fresh wrapper for the step's query and kv offsets."""
        import flashinfer

        c = self.layer_config
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(_workspace(self.resources), "NHD")
        wrapper.plan(
            qo_indptr=infer_state.b1_cu_q_seq_len.int(),
            kv_indptr=infer_state.b1_cu_kv_seq_len.int(),
            num_qo_heads=c.num_heads,
            num_kv_heads=c.num_heads,
            head_dim_qk=c.qk_nope_head_dim + c.qk_rope_head_dim,
            head_dim_vo=c.v_head_dim,
            q_data_type=c.dtype,
            causal=True,
            sm_scale=c.softmax_scale,
        )
        return wrapper


class FlashInferMLADecodeBackend(
    AttentionBackend[MLADecodeLayerConfig, FlashInferMLADecodeAttState, MLADecodeLayerInputs]
):
    name = "flashinfer_mla_decode"

    @classmethod
    def _check_supported(cls, layer_config: MLADecodeLayerConfig) -> None:
        # Each query row reads its own request prefix, which is causal across a request's rows.
        if not layer_config.causal:
            raise NotImplementedError("FlashInfer MLA decode supports only causal attention")

    @classmethod
    def group_key(cls, layer_config: MLADecodeLayerConfig):
        # FlashInfer plans with every field but causal, which the backend requires.
        return layer_config

    def setup(self, resources: AttentionBuildResources) -> None:
        """Keep one page-index buffer per microbatch for batches of up to the CUDA graph batch size."""
        set_flashinfer_envs()
        self.resources = resources
        self.page_size = resources.page_size
        self.max_pages_per_request = triton.cdiv(resources.max_seq_len, self.page_size)
        self.max_batch_size = resources.max_cuda_graph_batch_size
        numel = self.max_batch_size * self.max_pages_per_request
        self._kv_indices = [torch.empty(numel, dtype=torch.int32, device=resources.device) for _ in range(2)]
        self._graph_wrappers: dict[tuple[int, int], "BatchMLAPagedAttentionWrapper"] = {}

    def forward(
        self,
        att_state: FlashInferMLADecodeAttState,
        layer_inputs: MLADecodeLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's shape; the planned workspace holds the scratch."""
        layer_config = self.layer_config
        kv, rank, page_size = layer_inputs.kv_cache, layer_config.kv_lora_rank, att_state.page_size
        att_state.wrapper.run(
            layer_inputs.q_nope,
            layer_inputs.q_rope,
            kv[:, :, :rank].view(-1, page_size, 1, rank),
            kv[:, :, rank:].view(-1, page_size, 1, layer_config.qk_rope_head_dim),
            out=out,
            return_lse=False,
        )
        return out

    def build(self, infer_state: InferStateInfo) -> FlashInferMLADecodeAttState:
        """Plan the step's wrapper; every query row is its own request.

        A CUDA graph step keeps one wrapper and its index buffers per (microbatch, batch size).
        The first build plans it on the host; replays rewrite the buffers in place and refill the
        wrapper's plan tables on the GPU.
        """
        import flashinfer
        from lightllm.common.basemodel.triton_kernel.flashinfer_mla_plan import fill_mla_decode_plan_for_cuda_graph
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index

        c, page_size = self.layer_config, self.page_size
        device = infer_state.b_seq_len.device
        # A partial final page still counts as one page.
        b_page_len = (infer_state.b_seq_len + (page_size - 1)) // page_size
        kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        page_ids = infer_state.batch_size * self.max_pages_per_request
        if infer_state.batch_size <= self.max_batch_size:
            kv_indices = self._kv_indices[infer_state.microbatch_index][:page_ids]
        else:
            # Captured wrappers must read the fixed per-microbatch buffer.
            assert not infer_state.uses_cuda_graph, "CUDA graph batches must fit the page-index buffers"
            kv_indices = torch.empty(page_ids, dtype=torch.int32, device=device)
        q_indptr = torch.arange(infer_state.batch_size + 1, dtype=torch.int32, device=device)
        key = (infer_state.microbatch_index, infer_state.batch_size)
        kv_starts = self._keep_for_graph(infer_state, "kv_starts", kv_starts)
        if infer_state.uses_cuda_graph:
            # The row offsets are constant, so the captured tensor needs no rewrite.
            q_indptr = self._graph_tensors.setdefault((*key, "q_indptr"), q_indptr)
        repack_kv_index(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_token_len=infer_state.b_seq_len,
            b_page_start_loc=kv_starts[:-1],
            max_token_len=infer_state.max_kv_seq_len,
            out_page_indices=kv_indices,
            page_size=page_size,
        )
        wrapper = self._graph_wrappers.get(key) if infer_state.uses_cuda_graph else None
        if wrapper is not None:
            # Prefer the GPU-generated split plan for long decode; exact non-split for short decode.
            fill_mla_decode_plan_for_cuda_graph(
                decode_wrapper=wrapper,
                kv_page_indptr=kv_starts,
                kv_lens=infer_state.b_seq_len,
                batch_size=infer_state.batch_size,
                num_heads=c.num_heads,
                max_kv_len=infer_state.max_kv_seq_len,
            )
            return FlashInferMLADecodeAttState(wrapper=wrapper, page_size=page_size)
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            _workspace(self.resources),
            use_cuda_graph=True,
            qo_indptr=q_indptr,
            kv_indices=kv_indices,
            kv_indptr=kv_starts,
            kv_len_arr=infer_state.b_seq_len,
        )
        wrapper.plan(
            q_indptr,
            kv_starts,
            kv_indices,
            infer_state.b_seq_len,
            c.num_heads,
            c.kv_lora_rank,
            c.qk_rope_head_dim,
            page_size,
            # Each row is its own request, so causality across a request's rows comes from the lengths.
            False,
            c.softmax_scale,
            c.dtype,
            c.dtype,
        )
        if infer_state.uses_cuda_graph:
            self._graph_wrappers[key] = wrapper
        return FlashInferMLADecodeAttState(wrapper=wrapper, page_size=page_size)
