from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional

import torch
import triton

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionBuildResources,
    TensorAllocator,
)
from ..flashinfer_env import set_flashinfer_envs
from . import GQALayerConfig, GQALayerInputs

if TYPE_CHECKING:
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


@dataclass(frozen=True)
class FlashInferGQABuilderConfig:
    """Per-rank configuration a FlashInfer step builder builds its wrappers for.

    Page index p addresses cache slots [p * page_size, (p + 1) * page_size).
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype


@dataclass(frozen=True)
class FlashInferGQAPrefillAttState:
    """A planned NHD wrapper for packed, causal prefill over the paged cache."""

    wrapper: "BatchPrefillWithPagedKVCacheWrapper"
    config: FlashInferGQABuilderConfig


@dataclass(frozen=True)
class FlashInferGQADecodeAttState:
    """A planned NHD tensor-core wrapper for one query per batch row."""

    wrapper: "BatchDecodeWithPagedKVCacheWrapper"
    config: FlashInferGQABuilderConfig


def _page_metadata(batch: AttentionMetadata, page_size: int, kv_indices: torch.Tensor):
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


class FlashInferGQAPrefillBuildContext:
    def __init__(
        self, config: FlashInferGQABuilderConfig, *, workspace: Callable[[], torch.Tensor], max_seq_len: int
    ) -> None:
        """workspace returns the caller-owned FlashInfer workspace; max_seq_len bounds request lengths."""
        self.config = config
        self.workspace = workspace
        self.max_pages_per_request = triton.cdiv(max_seq_len, config.page_size)


class FlashInferGQADecodeBuildContext:
    def __init__(
        self,
        config: FlashInferGQABuilderConfig,
        *,
        workspace: Callable[[], torch.Tensor],
        max_seq_len: int,
        max_batch_size: int,
        device: torch.device,
    ) -> None:
        """Keep one page-index buffer per microbatch for batches of up to max_batch_size requests."""
        self.config = config
        self.workspace = workspace
        self.max_pages_per_request = triton.cdiv(max_seq_len, config.page_size)
        self.max_batch_size = max_batch_size
        self._kv_indices = [
            torch.empty(max_batch_size * self.max_pages_per_request, dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._cuda_graph_wrappers: dict[tuple[int, int], tuple] = {}

    def _fast_replan(self, wrapper, kv_starts, kv_indices, kv_last_page_len, batch: AttentionMetadata) -> None:
        from flashinfer.decode import fast_decode_plan

        config = self.config
        uniform_kv_indptr_cpu = torch.arange(batch.batch_size + 1, dtype=torch.int32, device="cpu") * triton.cdiv(
            batch.max_kv_seq_len, config.page_size
        )
        fast_decode_plan(
            wrapper,
            indptr=kv_starts,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            page_size=config.page_size,
            q_data_type=config.q_dtype,
            kv_data_type=config.kv_dtype,
            non_blocking=True,
            global_override_indptr_cpu=uniform_kv_indptr_cpu,
        )


def _check_supported(layer_config: GQALayerConfig) -> None:
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


def _run(
    layer_config: GQALayerConfig,
    wrapper,
    config: FlashInferGQABuilderConfig,
    layer_inputs: GQALayerInputs,
    out: torch.Tensor,
):
    assert (config.num_heads, config.num_kv_heads, config.head_dim) == (
        layer_config.num_heads,
        layer_config.num_kv_heads,
        layer_config.head_dim,
    ), "attention state were built for a different GQA layer configuration"
    assert layer_inputs.q.dtype == layer_config.dtype == config.q_dtype, "q does not match the layer_config"
    assert (layer_inputs.kv_scales is not None) == (
        layer_config.kv_quant is not None
    ), "kv_scales must be given exactly when the layer_config has kv_quant"
    k, v = layer_inputs.k_cache, layer_inputs.v_cache
    k = k.view(-1, config.page_size, k.shape[1], k.shape[2])
    v = v.view(-1, config.page_size, v.shape[1], v.shape[2])
    if layer_config.kv_quant is None:
        wrapper.run(q=layer_inputs.q, paged_kv_cache=(k, v), out=out)
        return out
    assert config.kv_dtype == torch.float8_e4m3fn, "attention state were planned for a different KV cache dtype"
    wrapper.run(
        q=layer_inputs.q,
        paged_kv_cache=(k.view(torch.float8_e4m3fn), v.view(torch.float8_e4m3fn)),
        k_scale=layer_inputs.kv_scales.k_scale,
        v_scale=layer_inputs.kv_scales.v_scale,
        out=out,
    )
    return out


class FlashInferGQAPrefillBackend(AttentionBackend[GQALayerConfig, FlashInferGQAPrefillAttState, GQALayerInputs]):
    def __init__(self, layer_config: GQALayerConfig) -> None:
        super().__init__(layer_config)
        _check_supported(layer_config)

    def forward(
        self,
        att_state: FlashInferGQAPrefillAttState,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape, dtype and device.

        Q/out are contiguous FP16/BF16 tensors on the device of the cache and att_state. Cache slots
        form complete pages; the slot stride may include padding (including K/V slices of shared
        storage). Q/K/V are read-only; out must not overlap them or the attention state' buffers. Scratch is
        the planned wrapper's workspace, so forward does not call alloc_func.
        """
        return _run(self.layer_config, att_state.wrapper, att_state.config, layer_inputs, out)

    @staticmethod
    def get_name() -> str:
        return "flashinfer_gqa_prefill"

    build_context_type = FlashInferGQAPrefillBuildContext

    @classmethod
    def build(
        cls, metadata: AttentionMetadata, context: FlashInferGQAPrefillBuildContext
    ) -> FlashInferGQAPrefillAttState:
        """Build and plan a prefill wrapper for the step's packed queries and paged cache."""
        import flashinfer

        config = context.config
        q_starts = metadata.cu_q_seq_len.int()
        kv_indices = torch.empty(
            metadata.batch_size * context.max_pages_per_request, dtype=torch.int32, device=metadata.b_seq_len.device
        )
        kv_starts, kv_last_page_len = _page_metadata(metadata, config.page_size, kv_indices)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            context.workspace(),
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
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            config.page_size,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            q_data_type=config.q_dtype,
            kv_data_type=config.kv_dtype,
        )
        return FlashInferGQAPrefillAttState(wrapper=wrapper, config=config)

    @classmethod
    def create_build_context(
        cls, build_key: "_BuilderKey", ctx: AttentionBuildResources
    ) -> FlashInferGQAPrefillBuildContext:
        set_flashinfer_envs()
        return FlashInferGQAPrefillBuildContext(
            flashinfer_builder_config(build_key, ctx), workspace=_workspace(ctx), max_seq_len=ctx.max_seq_len
        )

    @classmethod
    def build_context_key(cls, layer_config: GQALayerConfig):
        return _builder_key(layer_config)


class FlashInferGQADecodeBackend(AttentionBackend[GQALayerConfig, FlashInferGQADecodeAttState, GQALayerInputs]):
    def __init__(self, layer_config: GQALayerConfig) -> None:
        super().__init__(layer_config)
        _check_supported(layer_config)

    def forward(
        self,
        att_state: FlashInferGQADecodeAttState,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape, one query per batch row; layout rules match prefill."""
        return _run(self.layer_config, att_state.wrapper, att_state.config, layer_inputs, out)

    @staticmethod
    def get_name() -> str:
        return "flashinfer_gqa_decode"

    build_context_type = FlashInferGQADecodeBuildContext

    @classmethod
    def build(
        cls, metadata: AttentionMetadata, context: FlashInferGQADecodeBuildContext
    ) -> FlashInferGQADecodeAttState:
        """Build the attention state, planning FlashInfer's decode wrapper for this step.

        Each metadata row is planned as its own request, so speculative decode rows need no
        regrouping. A CUDA graph step keeps one wrapper per (microbatch, metadata size) whose
        index buffers the graph captured. Its replay steps rewrite those buffers in place and
        re-plan it with fast_decode_plan, which schedules every request as if it had the step's
        longest length.
        """
        import flashinfer

        config = context.config
        page_ids = metadata.batch_size * context.max_pages_per_request
        if metadata.batch_size <= context.max_batch_size:
            kv_indices = context._kv_indices[metadata.microbatch_index][:page_ids]
        else:
            # Captured wrappers must read the fixed per-microbatch buffer.
            assert not metadata.uses_cuda_graph, "CUDA graph batches must fit the page-index buffers"
            kv_indices = torch.empty(page_ids, dtype=torch.int32, device=metadata.b_seq_len.device)
        kv_starts, kv_last_page_len = _page_metadata(metadata, config.page_size, kv_indices)

        key = (metadata.microbatch_index, metadata.batch_size)
        captured = context._cuda_graph_wrappers.get(key) if metadata.uses_cuda_graph else None
        if captured is not None:
            wrapper, captured_starts, captured_last_page_len = captured
            captured_starts.copy_(kv_starts)
            captured_last_page_len.copy_(kv_last_page_len)
            context._fast_replan(wrapper, captured_starts, kv_indices, captured_last_page_len, metadata)
            return FlashInferGQADecodeAttState(wrapper=wrapper, config=config)

        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            context.workspace(),
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
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            config.page_size,
            q_data_type=config.q_dtype,
            kv_data_type=config.kv_dtype,
            non_blocking=True,
        )
        if metadata.uses_cuda_graph:
            context._cuda_graph_wrappers[key] = (wrapper, kv_starts, kv_last_page_len)
        return FlashInferGQADecodeAttState(wrapper=wrapper, config=config)

    @classmethod
    def create_build_context(
        cls, build_key: "_BuilderKey", ctx: AttentionBuildResources
    ) -> FlashInferGQADecodeBuildContext:
        set_flashinfer_envs()
        return FlashInferGQADecodeBuildContext(
            flashinfer_builder_config(build_key, ctx),
            workspace=_workspace(ctx),
            max_seq_len=ctx.max_seq_len,
            max_batch_size=ctx.max_cuda_graph_batch_size,
            device=ctx.device,
        )

    @classmethod
    def build_context_key(cls, layer_config: GQALayerConfig):
        return _builder_key(layer_config)


class _BuilderKey(NamedTuple):
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    kv_quant: Optional[str]


def flashinfer_builder_config(build_key: _BuilderKey, ctx: AttentionBuildResources) -> FlashInferGQABuilderConfig:
    """The wrapper configuration for layers with this layer config."""
    fp8 = build_key.kv_quant == "fp8_per_tensor"
    return FlashInferGQABuilderConfig(
        num_heads=build_key.num_heads,
        num_kv_heads=build_key.num_kv_heads,
        head_dim=build_key.head_dim,
        # TODO: use the cache's page size once FP8 FlashInfer supports multi-token pages.
        page_size=1 if fp8 else ctx.page_size,
        q_dtype=build_key.dtype,
        kv_dtype=torch.float8_e4m3fn if fp8 else build_key.dtype,
    )


def _workspace(ctx: AttentionBuildResources) -> Callable[[], torch.Tensor]:
    if ctx.workspace_bytes <= 0:
        raise ValueError("the FlashInfer workspace size (--flashinfer_workspace_size) must be positive")
    # Allocated on first use, so prefill-only and decode-only nodes create it only when needed.
    return lambda: ctx.get_workspace("flashinfer_fp", ctx.workspace_bytes, torch.int8)


def _builder_key(layer_config: GQALayerConfig):
    # The wrappers are planned for these fields.
    c = layer_config
    return _BuilderKey(c.num_heads, c.num_kv_heads, c.head_dim, c.dtype, c.kv_quant)
