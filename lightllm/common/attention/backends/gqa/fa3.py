from dataclasses import dataclass
from typing import Callable, Optional

import torch
import triton

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBuildResources,
    TensorAllocator,
)
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.utils.log_utils import init_logger
from .. import keep_for_cuda_graph
from . import GQALayerConfig, GQALayerInputs

logger = init_logger(__name__)


@dataclass(frozen=True)
class FA3GQAPrefillAttState:
    """Paged-cache metadata for packed prefill queries, one page-table row per request.

    page_table[i, j] is the physical page holding request i's tokens
    [j * page_size, (j + 1) * page_size). cu_seqlens_q/cu_seqlens_k are the int32 cumulative
    query and kv offsets that place each request's new keys in the paged cache.
    """

    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    page_size: int
    # Per query token: exclusive end of its image span, 0 for text.
    b_image_token_end: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class FA3GQADecodeAttState:
    """Paged-cache metadata for decode, one page-table row per attention request.

    With speculative decoding an attention request groups the query rows of one logical
    request; cu_seqlens_q gives the int32 query offsets and cache_seqlens each request's kv
    length. max_seqlen_k is this step's actual maximum, only a kernel-config hint.
    """

    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    page_size: int


class FA3GQAPrefillBuildContext:
    def __init__(self, *, page_size: int) -> None:
        self.page_size = page_size


class FA3GQADecodeBuildContext:
    def __init__(self, *, page_size: int, page_table_view: Callable[[int, int, int], torch.Tensor]) -> None:
        """page_table_view(rows, max_kv_len, microbatch_index) returns a [rows, pages] view of
        caller-owned page-table storage, the same storage for the same arguments."""
        self.page_size = page_size
        self.page_table_view = page_table_view
        self._cuda_graph_tensors: dict[tuple, torch.Tensor] = {}


def build_fa3_decode_state(batch: InferStateInfo, context: FA3GQADecodeBuildContext) -> FA3GQADecodeAttState:
    """Group query rows into attention requests and fill their page-table rows.

    A CUDA graph step uses a page table as wide as the graph capacity and keeps every
    derived tensor at a fixed address per (microbatch, batch size), rewriting them in place
    on replay steps.
    """
    from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy

    cu_seqlens_q, cu_seqlens_k, cache_seqlens, att_req_idx = _group_rows(batch)
    width = batch.cuda_graph_max_kv_seq_len if batch.uses_cuda_graph else batch.max_kv_seq_len
    page_table = context.page_table_view(att_req_idx.shape[0], width, batch.microbatch_index)
    page_table_copy(
        page_table=page_table[:, : triton.cdiv(batch.max_kv_seq_len, context.page_size)],
        req_to_token_indexs=batch.req_to_token_indexs,
        b_req_idx=att_req_idx,
        page_size=context.page_size,
    )
    if batch.uses_cuda_graph:
        key = (batch.microbatch_index, batch.batch_size)
        store = context._cuda_graph_tensors
        cu_seqlens_q = keep_for_cuda_graph(store, (*key, "cu_seqlens_q"), cu_seqlens_q)
        cu_seqlens_k = keep_for_cuda_graph(store, (*key, "cu_seqlens_k"), cu_seqlens_k)
        cache_seqlens = keep_for_cuda_graph(store, (*key, "cache_seqlens"), cache_seqlens)
    return FA3GQADecodeAttState(
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=batch.decode_rows_per_request,
        max_seqlen_k=batch.max_kv_seq_len,
        page_size=context.page_size,
    )


def _group_rows(batch: InferStateInfo):
    """Return int32 query/kv offsets, kv lengths and request slots per attention request."""
    from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor

    rows = batch.decode_rows_per_request
    if batch.dynamic_decode_rows:
        from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_spec_fa3_decode_params
        from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

        markers = build_mtp_shared_group_markers(batch.b_req_idx, hold_req_id=batch.pad_req_idx)
        q_lens, kv_lens, att_req_idx, cache_seqlens = build_dynamic_spec_fa3_decode_params(
            b_req_idx=batch.b_req_idx,
            b_seq_len=batch.b_seq_len,
            b_mark_mtp_shared_group=markers,
            att_batch_size=batch.batch_size,
            hold_req_id=batch.pad_req_idx,
        )
    elif rows > 1:
        assert batch.batch_size % rows == 0, (
            "FA3 fixed-layout decode requires batch_size to be divisible by the rows per request, "
            f"got batch_size={batch.batch_size}, rows={rows}"
        )
        # The last row of each request sees its longest prefix.
        q_lens = torch.full((batch.batch_size // rows,), rows, dtype=torch.int32, device=batch.b_seq_len.device)
        kv_lens = batch.b_seq_len[rows - 1 :: rows]
        att_req_idx = batch.b_req_idx[rows - 1 :: rows]
        cache_seqlens = kv_lens.contiguous()
    else:
        return batch.b1_cu_q_seq_len.int(), batch.b1_cu_kv_seq_len.int(), batch.b_seq_len, batch.b_req_idx
    cu_seqlens_q, cu_seqlens_k = gen_cumsum_pad0_tensor(q_lens, kv_lens)
    return cu_seqlens_q.int(), cu_seqlens_k.int(), cache_seqlens, att_req_idx


def _check_gqa_config(layer_config: GQALayerConfig) -> None:
    if layer_config.dtype not in (torch.float16, torch.bfloat16) or layer_config.head_dim > 256:
        raise NotImplementedError(
            f"FA3 GQA supports FP16/BF16 with head_dim <= 256, got {layer_config.dtype}, {layer_config.head_dim}"
        )
    if layer_config.has_alibi:
        raise NotImplementedError("FA3 GQA does not support ALiBi")
    if layer_config.bidirectional_image_spans and (
        layer_config.kv_quant is not None or layer_config.sliding_window != (-1, -1) or layer_config.has_sinks
    ):
        raise NotImplementedError("FA3 GQA supports bidirectional image spans without KV quantization, window or sinks")
    if layer_config.kv_quant not in (None, "fp8_per_head"):
        raise NotImplementedError(f"FA3 GQA supports kv_quant None or 'fp8_per_head', got {layer_config.kv_quant!r}")
    if layer_config.kv_quant is not None and (
        layer_config.sliding_window != (-1, -1) or layer_config.has_sinks or not layer_config.has_default_scale
    ):
        raise NotImplementedError("FA3 GQA with an FP8 cache supports only the default scale without window or sinks")
    if layer_config.kv_quant is not None and layer_config.dtype != torch.bfloat16:
        # FA3 writes FP8 attention output only as BF16.
        raise NotImplementedError(f"FA3 GQA with an FP8 cache needs a BF16 model, got {layer_config.dtype}")


def _paged(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    return cache.view(-1, page_size, cache.shape[1], cache.shape[2])


def _check_inputs(layer_config: GQALayerConfig, layer_inputs: GQALayerInputs) -> None:
    assert (
        layer_inputs.q.shape[1:] == (layer_config.num_heads, layer_config.head_dim)
        and layer_inputs.q.dtype == layer_config.dtype
    )
    assert layer_inputs.k_cache.shape[1:] == (layer_config.num_kv_heads, layer_config.head_dim)
    assert (
        layer_inputs.sinks is not None
    ) == layer_config.has_sinks, "sinks must be given exactly when the layer_config has sinks"
    assert (layer_inputs.kv_scales is not None) == (
        layer_config.kv_quant is not None
    ), "kv_scales must be given exactly when the layer_config has kv_quant"
    if layer_config.kv_quant is not None:
        assert (layer_inputs.kv_scales.q_scale is not None) == layer_config.has_q_scale


def _fp8_operands(layer_config: GQALayerConfig, layer_inputs: GQALayerInputs, att_state, page_size: int):
    """Quantize q per KV head and return FA3's FP8 q, paged caches and [requests, kv_heads] descales."""
    from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
        q_per_head_fp8_quant,
        q_per_head_static_fp8_quant,
    )

    kv_heads, scales = layer_config.num_kv_heads, layer_inputs.kv_scales
    requests = att_state.cache_seqlens.shape[0]
    q = layer_inputs.q.reshape(layer_inputs.q.shape[0], kv_heads, -1)
    if scales.q_scale is None:
        # Without a static scale, quantize each request's queries with their own per-head scale.
        q, q_scale = q_per_head_fp8_quant(
            q=q,
            seq_lens=att_state.cu_seqlens_q[1:] - att_state.cu_seqlens_q[:-1],
            b1_start_loc=att_state.cu_seqlens_q,
        )
    else:
        q = q_per_head_static_fp8_quant(q=q, scales=scales.q_scale)
        q_scale = scales.q_scale.view(1, kv_heads).expand(requests, kv_heads)

    def paged(cache):
        return _paged(cache, page_size).view(torch.float8_e4m3fn)

    return dict(
        q=q.reshape(-1, layer_config.num_heads, layer_config.head_dim),
        k_cache=paged(layer_inputs.k_cache),
        v_cache=paged(layer_inputs.v_cache),
        q_descale=q_scale.view(requests, kv_heads),
        k_descale=scales.k_scale.view(1, kv_heads).expand(requests, kv_heads),
        v_descale=scales.v_scale.view(1, kv_heads).expand(requests, kv_heads),
    )


class FA3GQAPrefillBackend(AttentionBackend[GQALayerConfig, FA3GQAPrefillAttState, GQALayerInputs]):
    """Supports any scale, causal or non-causal attention, one- or two-sided windows and sinks.

    A per-head FP8 cache (kv_quant='fp8_per_head') supports the default scale without window or
    sinks; q is quantized per KV head with the static q scale, or dynamically per request.
    """

    def __init__(self, layer_config: GQALayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.bidirectional_image_spans:
            from lightllm.utils.neo_fa3_utils import HAS_FLASH_ATTN_INTERFACE, NEO_FA3_INSTALL_HINT

            if not HAS_FLASH_ATTN_INTERFACE:
                raise NotImplementedError(
                    "FA3 bidirectional image spans need FA3 with image_token_end support.\n" + NEO_FA3_INSTALL_HINT
                )

    @classmethod
    def _check_supported(cls, layer_config: GQALayerConfig) -> None:
        _check_gqa_config(layer_config)

    def forward(
        self,
        att_state: FA3GQAPrefillAttState,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape, dtype and device.

        Cache slots form complete pages and the current queries' KV is already in the cache.
        Q/K/V are read-only; out must not overlap them. FA3 manages its own scratch, so forward
        does not call alloc_func.
        """
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        layer_config = self.layer_config
        _check_inputs(layer_config, layer_inputs)
        if layer_config.bidirectional_image_spans:
            from lightllm.utils.neo_fa3_utils import flash_attn_with_kvcache_neo

            assert att_state.b_image_token_end is not None, "bidirectional image spans need b_image_token_end"
            # The Neo FA3 build returns its own output, so copy it into out.
            out.copy_(
                flash_attn_with_kvcache_neo(
                    q=layer_inputs.q,
                    k_cache=_paged(layer_inputs.k_cache, att_state.page_size),
                    v_cache=_paged(layer_inputs.v_cache, att_state.page_size),
                    page_table=att_state.page_table,
                    cache_seqlens=att_state.cache_seqlens,
                    cu_seqlens_q=att_state.cu_seqlens_q,
                    cu_seqlens_k_new=att_state.cu_seqlens_k,
                    max_seqlen_q=att_state.max_seqlen_q,
                    softmax_scale=layer_config.softmax_scale,
                    causal=layer_config.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    k_descale=None,
                    v_descale=None,
                    return_softmax_lse=False,
                    image_token_end=att_state.b_image_token_end,
                )
            )
            return out
        if layer_config.kv_quant is not None:
            flash_attn_with_kvcache(
                **_fp8_operands(layer_config, layer_inputs, att_state, att_state.page_size),
                page_table=att_state.page_table,
                cache_seqlens=att_state.cache_seqlens,
                cu_seqlens_q=att_state.cu_seqlens_q,
                cu_seqlens_k_new=att_state.cu_seqlens_k,
                max_seqlen_q=att_state.max_seqlen_q,
                causal=layer_config.causal,
                window_size=(-1, -1),
                softcap=0.0,
                return_softmax_lse=False,
                out=out,
            )
            return out
        flash_attn_with_kvcache(
            q=layer_inputs.q,
            k_cache=_paged(layer_inputs.k_cache, att_state.page_size),
            v_cache=_paged(layer_inputs.v_cache, att_state.page_size),
            page_table=att_state.page_table,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k_new=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            softmax_scale=layer_config.softmax_scale,
            causal=layer_config.causal,
            window_size=layer_config.sliding_window,
            softcap=0.0,
            return_softmax_lse=False,
            sinks=layer_inputs.sinks,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_gqa_prefill"

    build_context_type = FA3GQAPrefillBuildContext

    @classmethod
    def build(cls, metadata: InferStateInfo, context: FA3GQAPrefillBuildContext) -> FA3GQAPrefillAttState:
        """Build a fresh page table for the step's requests."""
        from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy

        page_table = torch.empty(
            (metadata.batch_size, triton.cdiv(metadata.max_kv_seq_len, context.page_size)),
            dtype=torch.int32,
            device=metadata.b_seq_len.device,
        )
        page_table_copy(
            page_table=page_table,
            req_to_token_indexs=metadata.req_to_token_indexs,
            b_req_idx=metadata.b_req_idx,
            page_size=context.page_size,
        )
        return FA3GQAPrefillAttState(
            page_table=page_table,
            cache_seqlens=metadata.b_seq_len,
            cu_seqlens_q=metadata.b1_cu_q_seq_len.int(),
            cu_seqlens_k=metadata.b1_cu_kv_seq_len.int(),
            max_seqlen_q=metadata.max_q_seq_len,
            page_size=context.page_size,
            b_image_token_end=metadata.b_image_token_end,
        )

    @classmethod
    def create_build_context(cls, build_key: Optional[str], ctx: AttentionBuildResources) -> FA3GQAPrefillBuildContext:
        return FA3GQAPrefillBuildContext(page_size=fa3_page_size(build_key, ctx))

    @classmethod
    def build_context_key(cls, layer_config: GQALayerConfig):
        return layer_config.kv_quant

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return fa3_requires_cpu_cache_sync()


class FA3GQADecodeBackend(AttentionBackend[GQALayerConfig, FA3GQADecodeAttState, GQALayerInputs]):
    @classmethod
    def _check_supported(cls, layer_config: GQALayerConfig) -> None:
        """Supports any scale, causal or non-causal attention, one- or two-sided windows and sinks.

        A per-head FP8 cache (kv_quant='fp8_per_head') supports the default scale without window or
        sinks; q is quantized per KV head with the static q scale, or dynamically per request.
        """
        _check_gqa_config(layer_config)
        if layer_config.bidirectional_image_spans:
            raise NotImplementedError("bidirectional image spans apply to prefill only")

    def forward(
        self,
        att_state: FA3GQADecodeAttState,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape; layout rules match prefill.

        q has one row per batch row; the attention state group them into attention requests.
        """
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache, flash_attn_with_kvcache_autotune

        layer_config = self.layer_config
        _check_inputs(layer_config, layer_inputs)
        if layer_config.kv_quant is not None:
            flash_attn_with_kvcache(
                **_fp8_operands(layer_config, layer_inputs, att_state, att_state.page_size),
                page_table=att_state.page_table,
                cache_seqlens=att_state.cache_seqlens,
                cu_seqlens_q=att_state.cu_seqlens_q,
                cu_seqlens_k_new=att_state.cu_seqlens_k,
                max_seqlen_q=att_state.max_seqlen_q,
                causal=layer_config.causal,
                window_size=(-1, -1),
                softcap=0.0,
                return_softmax_lse=False,
                out=out,
            )
            return out
        flash_attn_with_kvcache_autotune(
            q=layer_inputs.q,
            k_cache=_paged(layer_inputs.k_cache, att_state.page_size),
            v_cache=_paged(layer_inputs.v_cache, att_state.page_size),
            page_table=att_state.page_table,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            # The KV is already in the cache, so no new K/V segments are appended.
            cu_seqlens_k_new=None,
            max_seqlen_q=att_state.max_seqlen_q,
            max_seqlen_k=att_state.max_seqlen_k,
            softmax_scale=layer_config.softmax_scale,
            causal=layer_config.causal,
            window_size=layer_config.sliding_window,
            softcap=0.0,
            return_softmax_lse=False,
            sinks=layer_inputs.sinks,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_gqa_decode"

    build_context_type = FA3GQADecodeBuildContext

    @classmethod
    def build(cls, metadata: InferStateInfo, context: FA3GQADecodeBuildContext) -> FA3GQADecodeAttState:
        return build_fa3_decode_state(metadata, context)

    @classmethod
    def create_build_context(cls, build_key: Optional[str], ctx: AttentionBuildResources) -> FA3GQADecodeBuildContext:
        page_size = fa3_page_size(build_key, ctx)
        return FA3GQADecodeBuildContext(page_size=page_size, page_table_view=FA3PageTables(ctx, page_size).view)

    @classmethod
    def build_context_key(cls, layer_config: GQALayerConfig):
        return layer_config.kv_quant

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return fa3_requires_cpu_cache_sync()


def fa3_page_size(kv_quant: Optional[str], ctx: AttentionBuildResources) -> int:
    """The page size FA3 runs with for this layer config: the cache's, except 1 for an FP8 cache."""
    if kv_quant is None:
        return ctx.page_size
    # TODO: use the cache's page size once FP8 FA3 supports multi-token pages.
    logger.warning(f"FP8 FA3 temporarily uses page size 1 with model page_size={ctx.page_size}")
    return 1


class FA3PageTables:
    """Fixed page-table storage for FA3 decode, one buffer per microbatch, allocated on first use.

    Each buffer holds ctx.max_decode_rows rows of pages for ctx.max_seq_len tokens and is shared
    by name with every FA3 decode step builder of the process.
    """

    def __init__(self, ctx: AttentionBuildResources, page_size: int) -> None:
        self.ctx = ctx
        self.page_size = page_size
        self._buffers: Optional[list[torch.Tensor]] = None

    def view(self, rows: int, max_kv_len: int, microbatch_index: int) -> torch.Tensor:
        """Return a contiguous [rows, pages] view of microbatch_index's buffer."""
        ctx = self.ctx
        if self._buffers is None:
            # Allocated lazily, so a prefill-only node never allocates decode-only storage.
            numel = ctx.max_decode_rows * triton.cdiv(ctx.max_seq_len, self.page_size)
            self._buffers = [
                ctx.get_workspace(f"fa3_page_table_{index}", numel, torch.int32)
                for index in range(ctx.microbatch_count)
            ]
        if rows > ctx.max_decode_rows:
            raise RuntimeError(f"FA3 attention batch size {rows} exceeds page-table capacity {ctx.max_decode_rows}")
        if max_kv_len > ctx.max_seq_len:
            raise RuntimeError(f"FA3 max KV sequence length {max_kv_len} exceeds page-table capacity {ctx.max_seq_len}")
        max_pages = triton.cdiv(max_kv_len, self.page_size)
        return self._buffers[microbatch_index][: rows * max_pages].view(rows, max_pages)


def fa3_requires_cpu_cache_sync() -> bool:
    # 设为 1 / TRUE / ON 时，FA3 与 CPU cache 搬运在 GPU 上互相等待。
    return enable_env_vars("LIGHTLLM_FA3_CPU_CACHE_SYNC")
