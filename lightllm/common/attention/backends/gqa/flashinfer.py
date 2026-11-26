from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import torch
import triton

from lightllm.common.attention.backend import AttentionBackend, AttentionBuildResources, TensorAllocator
from lightllm.common.basemodel.infer_struct import InferStateInfo
from .. import set_flashinfer_envs
from . import GQALayerConfig, GQALayerInputs

if TYPE_CHECKING:
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


@dataclass(frozen=True)
class FlashInferGQAAttState:
    """A wrapper planned for the step over the paged cache: NHD, packed and causal in prefill, tensor-core
    with one query per batch row in decode. Page index p addresses cache slots [p * page_size, (p + 1) * page_size).
    """

    wrapper: Union["BatchPrefillWithPagedKVCacheWrapper", "BatchDecodeWithPagedKVCacheWrapper"]
    page_size: int


def _page_metadata(batch: InferStateInfo, page_size: int, kv_indices: torch.Tensor):
    """Return per-request page offsets and last-page lengths, writing page ids into kv_indices."""
    from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
    from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index

    # A partial final page still counts as one page.
    b_page_len = (batch.b_seq_len + (page_size - 1)) // page_size
    kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
    kv_last_page_len = batch.b_seq_len - (b_page_len - 1) * page_size
    repack_kv_index(
        req_to_token_indexs=batch.req_to_token_indexs,
        b_req_idx=batch.b_req_idx,
        b_token_len=batch.b_seq_len,
        b_page_start_loc=kv_starts[:-1],
        max_token_len=batch.max_kv_seq_len,
        out_page_indices=kv_indices,
        page_size=page_size,
    )
    return kv_starts, kv_last_page_len


def _check_gqa_config(layer_config: GQALayerConfig) -> None:
    if not (
        layer_config.causal
        and layer_config.sliding_window == (-1, -1)
        and layer_config.has_default_scale
        and not layer_config.has_sinks
        and layer_config.kv_quant in (None, "fp8_per_tensor")
        and not layer_config.has_alibi
        and not layer_config.bidirectional_image_spans
    ):
        raise NotImplementedError(
            "FlashInfer GQA supports only causal attention without sliding window or sinks, "
            "with softmax_scale 1/sqrt(head_dim) and kv_quant None or 'fp8_per_tensor'"
        )


class FlashInferGQAPrefillBackend(AttentionBackend[GQALayerConfig, FlashInferGQAAttState, GQALayerInputs]):
    name = "flashinfer_gqa_prefill"

    @classmethod
    def _check_supported(cls, layer_config: GQALayerConfig) -> None:
        _check_gqa_config(layer_config)

    @classmethod
    def group_key(cls, layer_config: GQALayerConfig):
        # The wrappers are planned for these fields.
        c = layer_config
        return c.num_heads, c.num_kv_heads, c.head_dim, c.dtype, c.kv_quant

    def setup(self, resources: AttentionBuildResources) -> None:
        if resources.workspace_bytes <= 0:
            raise ValueError("the FlashInfer workspace size (--flashinfer_workspace_size) must be positive")
        set_flashinfer_envs()
        c, fp8 = self.layer_config, self.layer_config.kv_quant == "fp8_per_tensor"
        # TODO: use the cache's page size once FP8 FlashInfer supports multi-token pages.
        self.page_size = 1 if fp8 else resources.page_size
        self.max_pages_per_request = triton.cdiv(resources.max_seq_len, self.page_size)
        self._plan_args = (c.num_heads, c.num_kv_heads, c.head_dim, self.page_size)
        self._dtypes = dict(q_data_type=c.dtype, kv_data_type=torch.float8_e4m3fn if fp8 else c.dtype)
        # Allocated on first use, so prefill-only and decode-only nodes create it only when needed.
        self._workspace = lambda: resources.get_workspace("flashinfer_fp", resources.workspace_bytes, torch.int8)

    def forward(
        self,
        att_state: FlashInferGQAAttState,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape, dtype and device.

        Q/out are contiguous FP16/BF16 tensors on the device of the cache and att_state. Cache slots
        form complete pages; the slot stride may include padding (including K/V slices of shared
        storage). Q/K/V are read-only; out must not overlap them or the attention state's buffers. Scratch is
        the planned wrapper's workspace, so forward does not call alloc_func. Decode has one query per
        batch row.
        """
        layer_config = self.layer_config
        assert layer_inputs.q.dtype == layer_config.dtype, "q does not match the layer_config"
        assert (layer_inputs.kv_scales is not None) == (
            layer_config.kv_quant is not None
        ), "kv_scales must be given exactly when the layer_config has kv_quant"
        k, v = layer_inputs.k_cache, layer_inputs.v_cache
        k = k.view(-1, att_state.page_size, k.shape[1], k.shape[2])
        v = v.view(-1, att_state.page_size, v.shape[1], v.shape[2])
        if layer_config.kv_quant is None:
            att_state.wrapper.run(q=layer_inputs.q, paged_kv_cache=(k, v), out=out)
            return out
        att_state.wrapper.run(
            q=layer_inputs.q,
            paged_kv_cache=(k.view(torch.float8_e4m3fn), v.view(torch.float8_e4m3fn)),
            k_scale=layer_inputs.kv_scales.k_scale,
            v_scale=layer_inputs.kv_scales.v_scale,
            out=out,
        )
        return out

    def build(self, infer_state: InferStateInfo) -> FlashInferGQAAttState:
        """Build and plan a prefill wrapper for the step's packed queries and paged cache."""
        import flashinfer

        q_starts = infer_state.b1_cu_q_seq_len.int()
        kv_indices = torch.empty(
            infer_state.batch_size * self.max_pages_per_request, dtype=torch.int32, device=infer_state.b_seq_len.device
        )
        kv_starts, kv_last_page_len = _page_metadata(infer_state, self.page_size, kv_indices)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace(),
            qo_indptr_buf=q_starts,
            paged_kv_indptr_buf=kv_starts,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=kv_last_page_len,
        )
        wrapper.plan(
            q_starts,
            kv_starts,
            kv_indices,
            kv_last_page_len,
            *self._plan_args,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            **self._dtypes,
        )
        return FlashInferGQAAttState(wrapper=wrapper, page_size=self.page_size)


class FlashInferGQADecodeBackend(FlashInferGQAPrefillBackend):
    name = "flashinfer_gqa_decode"

    def setup(self, resources: AttentionBuildResources) -> None:
        """Keep one page-index buffer per microbatch for batches of up to the CUDA graph batch size."""
        super().setup(resources)
        self.max_batch_size = resources.max_cuda_graph_batch_size
        numel = self.max_batch_size * self.max_pages_per_request
        self._kv_indices = [torch.empty(numel, dtype=torch.int32, device=resources.device) for _ in range(2)]
        self._graph_wrappers: dict[tuple[int, int], tuple] = {}

    def build(self, infer_state: InferStateInfo) -> FlashInferGQAAttState:
        """Build the attention state, planning FlashInfer's decode wrapper for this step.

        Each row is planned as its own request, so speculative decode rows need no regrouping. A
        CUDA graph step keeps one wrapper per (microbatch, batch size) whose index buffers the graph
        captured. Its replay steps rewrite those buffers in place and re-plan it with fast_decode_plan,
        which schedules every request as if it had the step's longest length.
        """
        import flashinfer
        from flashinfer.decode import fast_decode_plan

        page_ids = infer_state.batch_size * self.max_pages_per_request
        if infer_state.batch_size <= self.max_batch_size:
            kv_indices = self._kv_indices[infer_state.microbatch_index][:page_ids]
        else:
            # Captured wrappers must read the fixed per-microbatch buffer.
            assert not infer_state.uses_cuda_graph, "CUDA graph batches must fit the page-index buffers"
            kv_indices = torch.empty(page_ids, dtype=torch.int32, device=infer_state.b_seq_len.device)
        kv_starts, kv_last_page_len = _page_metadata(infer_state, self.page_size, kv_indices)

        key = (infer_state.microbatch_index, infer_state.batch_size)
        captured = self._graph_wrappers.get(key) if infer_state.uses_cuda_graph else None
        if captured is not None:
            wrapper, captured_starts, captured_last_page_len = captured
            captured_starts.copy_(kv_starts)
            captured_last_page_len.copy_(kv_last_page_len)
            pages_per_request = triton.cdiv(infer_state.max_kv_seq_len, self.page_size)
            uniform_kv_indptr_cpu = (
                torch.arange(infer_state.batch_size + 1, dtype=torch.int32, device="cpu") * pages_per_request
            )
            fast_decode_plan(
                wrapper,
                captured_starts,
                kv_indices,
                captured_last_page_len,
                *self._plan_args,
                **self._dtypes,
                non_blocking=True,
                global_override_indptr_cpu=uniform_kv_indptr_cpu,
            )
            return FlashInferGQAAttState(wrapper=wrapper, page_size=self.page_size)

        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace(),
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=kv_starts,
            paged_kv_indices_buffer=kv_indices,
            paged_kv_last_page_len_buffer=kv_last_page_len,
        )
        wrapper.plan(
            kv_starts,
            kv_indices,
            kv_last_page_len,
            *self._plan_args,
            **self._dtypes,
            non_blocking=True,
        )
        if infer_state.uses_cuda_graph:
            self._graph_wrappers[key] = (wrapper, kv_starts, kv_last_page_len)
        return FlashInferGQAAttState(wrapper=wrapper, page_size=self.page_size)
