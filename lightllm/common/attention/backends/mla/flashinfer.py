from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import triton

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBatch,
    AttentionImpl,
    AttentionStepBuilder,
    TensorAllocator,
)
from .. import keep_for_cuda_graph
from . import MLADecodeLayerConfig, MLADecodeLayerInputs, MLAPrefillLayerConfig, MLAPrefillLayerInputs

if TYPE_CHECKING:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper


@dataclass(frozen=True)
class FlashInferMLABuilderConfig:
    """Per-rank MLA configuration FlashInfer plans its wrappers for.

    FlashInfer fixes head counts, dimensions, softmax_scale and dtypes when it plans a step, so a
    step builder serves the layers whose layer configs match these values. Page index p addresses
    cache slots [p * page_size, (p + 1) * page_size).
    """

    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    page_size: int
    dtype: torch.dtype
    softmax_scale: float


@dataclass(frozen=True)
class FlashInferMLAPrefillStepArgs:
    """A planned causal ragged-KV wrapper over the step's expanded keys and values."""

    wrapper: "BatchPrefillWithRaggedKVCacheWrapper"
    config: FlashInferMLABuilderConfig


@dataclass(frozen=True)
class FlashInferMLADecodeStepArgs:
    """A planned paged MLA wrapper with one query row per request."""

    wrapper: "BatchMLAPagedAttentionWrapper"
    config: FlashInferMLABuilderConfig


class FlashInferMLAPrefillStepBuilder(AttentionStepBuilder[FlashInferMLAPrefillStepArgs]):
    def __init__(self, config: FlashInferMLABuilderConfig, *, workspace: Callable[[], torch.Tensor]) -> None:
        """workspace returns the caller-owned FlashInfer workspace."""
        self.config = config
        self.workspace = workspace

    def build(self, batch: AttentionBatch) -> FlashInferMLAPrefillStepArgs:
        """Plan a fresh wrapper for the step's query and kv offsets."""
        import flashinfer

        config = self.config
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(self.workspace(), "NHD")
        wrapper.plan(
            qo_indptr=batch.cu_q_seq_len.int(),
            kv_indptr=batch.cu_kv_seq_len.int(),
            num_qo_heads=config.num_heads,
            num_kv_heads=config.num_heads,
            head_dim_qk=config.qk_nope_head_dim + config.qk_rope_head_dim,
            head_dim_vo=config.v_head_dim,
            q_data_type=config.dtype,
            causal=True,
            sm_scale=config.softmax_scale,
        )
        return FlashInferMLAPrefillStepArgs(wrapper=wrapper, config=config)


class FlashInferMLADecodeStepBuilder(AttentionStepBuilder[FlashInferMLADecodeStepArgs]):
    def __init__(
        self,
        config: FlashInferMLABuilderConfig,
        *,
        workspace: Callable[[], torch.Tensor],
        max_seq_len: int,
        max_batch_size: int,
        device: torch.device,
    ) -> None:
        """workspace returns the caller-owned FlashInfer workspace; max_seq_len bounds request lengths.

        Keeps one page-index buffer per microbatch for batches of up to max_batch_size requests.
        """
        self.config = config
        self.workspace = workspace
        self.max_pages_per_request = triton.cdiv(max_seq_len, config.page_size)
        self.max_batch_size = max_batch_size
        self._kv_indices = [
            torch.empty(max_batch_size * self.max_pages_per_request, dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._cuda_graph_wrappers: dict[tuple[int, int], "BatchMLAPagedAttentionWrapper"] = {}
        self._cuda_graph_tensors: dict[tuple, torch.Tensor] = {}

    def build(self, batch: AttentionBatch) -> FlashInferMLADecodeStepArgs:
        """Plan the step's wrapper; every query row is its own request.

        A CUDA graph step keeps one wrapper and its index buffers per (microbatch, batch size).
        The first build plans it on the host; replays rewrite the buffers in place and refill the
        wrapper's plan tables on the GPU.
        """
        import flashinfer
        from lightllm.common.basemodel.triton_kernel.flashinfer_mla_plan import fill_mla_decode_plan_for_cuda_graph
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index

        config, page_size = self.config, self.config.page_size
        device = batch.b_seq_len.device
        # A partial final page still counts as one page.
        b_page_len = (batch.b_seq_len + (page_size - 1)) // page_size
        kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        page_ids = batch.batch_size * self.max_pages_per_request
        if batch.batch_size <= self.max_batch_size:
            kv_indices = self._kv_indices[batch.microbatch_index][:page_ids]
        else:
            # Captured wrappers must read the fixed per-microbatch buffer.
            assert not batch.uses_cuda_graph, "CUDA graph batches must fit the page-index buffers"
            kv_indices = torch.empty(page_ids, dtype=torch.int32, device=device)
        q_indptr = torch.arange(batch.batch_size + 1, dtype=torch.int32, device=device)
        key = (batch.microbatch_index, batch.batch_size)
        if batch.uses_cuda_graph:
            store = self._cuda_graph_tensors
            kv_starts = keep_for_cuda_graph(store, (*key, "kv_starts"), kv_starts)
            q_indptr = store.setdefault((*key, "q_indptr"), q_indptr)
        repack_kv_index(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_token_len=batch.b_seq_len,
            b_page_start_loc=kv_starts[:-1],
            max_token_len=batch.max_kv_seq_len,
            out_page_indices=kv_indices,
            page_size=page_size,
        )
        wrapper = self._cuda_graph_wrappers.get(key) if batch.uses_cuda_graph else None
        if wrapper is not None:
            # Prefer the GPU-generated split plan for long decode; exact non-split for short decode.
            fill_mla_decode_plan_for_cuda_graph(
                decode_wrapper=wrapper,
                kv_page_indptr=kv_starts,
                kv_lens=batch.b_seq_len,
                batch_size=batch.batch_size,
                num_heads=config.num_heads,
                max_kv_len=batch.max_kv_seq_len,
            )
            return FlashInferMLADecodeStepArgs(wrapper=wrapper, config=config)
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.workspace(),
            use_cuda_graph=True,
            qo_indptr=q_indptr,
            kv_indices=kv_indices,
            kv_indptr=kv_starts,
            kv_len_arr=batch.b_seq_len,
        )
        wrapper.plan(
            q_indptr,
            kv_starts,
            kv_indices,
            batch.b_seq_len,
            config.num_heads,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            page_size,
            # Each row is its own request, so causality across a request's rows comes from the lengths.
            False,
            config.softmax_scale,
            config.dtype,
            config.dtype,
        )
        if batch.uses_cuda_graph:
            self._cuda_graph_wrappers[key] = wrapper
        return FlashInferMLADecodeStepArgs(wrapper=wrapper, config=config)


def _check_config(layer_config, config: FlashInferMLABuilderConfig) -> None:
    assert layer_config.num_heads == config.num_heads and layer_config.dtype == config.dtype
    assert layer_config.qk_rope_head_dim == config.qk_rope_head_dim
    assert layer_config.softmax_scale == config.softmax_scale, "step args were planned with a different softmax_scale"


class FlashInferMLAPrefillImpl(
    AttentionImpl[MLAPrefillLayerConfig, FlashInferMLAPrefillStepArgs, MLAPrefillLayerInputs]
):
    def __init__(self, layer_config: MLAPrefillLayerConfig) -> None:
        super().__init__(layer_config)
        if not layer_config.causal:
            raise NotImplementedError("FlashInfer MLA prefill supports only causal attention")

    def forward(
        self,
        step_args: FlashInferMLAPrefillStepArgs,
        layer_inputs: MLAPrefillLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [query_tokens, num_heads, v_head_dim]; keys are expanded into alloc_func scratch."""
        layer_config, config = self.layer_config, step_args.config
        _check_config(layer_config, config)
        assert (layer_config.qk_nope_head_dim, layer_config.v_head_dim) == (config.qk_nope_head_dim, config.v_head_dim)
        q, k_nope = layer_inputs.q, layer_inputs.k_nope
        nope_dim, heads = layer_config.qk_nope_head_dim, layer_config.num_heads
        k = alloc_func((k_nope.shape[0], heads, q.shape[-1]), dtype=q.dtype, device=q.device)
        k[:, :, :nope_dim].copy_(k_nope)
        k[:, :, nope_dim:].copy_(layer_inputs.k_rope.expand(-1, heads, -1))
        step_args.wrapper.run(q, k, layer_inputs.v, out=out)
        return out


class FlashInferMLADecodeImpl(AttentionImpl[MLADecodeLayerConfig, FlashInferMLADecodeStepArgs, MLADecodeLayerInputs]):
    def __init__(self, layer_config: MLADecodeLayerConfig) -> None:
        super().__init__(layer_config)
        # Each query row reads its own request prefix, which is causal across a request's rows.
        if not layer_config.causal:
            raise NotImplementedError("FlashInfer MLA decode supports only causal attention")

    def forward(
        self,
        step_args: FlashInferMLADecodeStepArgs,
        layer_inputs: MLADecodeLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's shape; the planned workspace holds the scratch."""
        layer_config, config = self.layer_config, step_args.config
        _check_config(layer_config, config)
        assert layer_config.kv_lora_rank == config.kv_lora_rank
        kv, rank, page_size = layer_inputs.kv_cache, layer_config.kv_lora_rank, config.page_size
        step_args.wrapper.run(
            layer_inputs.q_nope,
            layer_inputs.q_rope,
            kv[:, :, :rank].view(-1, page_size, 1, rank),
            kv[:, :, rank:].view(-1, page_size, 1, layer_config.qk_rope_head_dim),
            out=out,
            return_lse=False,
        )
        return out


class FlashInferMLAPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "flashinfer_mla_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[FlashInferMLAPrefillStepBuilder]:
        return FlashInferMLAPrefillStepBuilder

    @staticmethod
    def get_impl_cls() -> type[FlashInferMLAPrefillImpl]:
        return FlashInferMLAPrefillImpl


class FlashInferMLADecodeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "flashinfer_mla_decode"

    @staticmethod
    def get_step_builder_cls() -> type[FlashInferMLADecodeStepBuilder]:
        return FlashInferMLADecodeStepBuilder

    @staticmethod
    def get_impl_cls() -> type[FlashInferMLADecodeImpl]:
        return FlashInferMLADecodeImpl
