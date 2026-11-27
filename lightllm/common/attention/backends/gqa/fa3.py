"""FA3 GQA over the paged cache, unquantized or with a per-head FP8 cache.

Decode groups the rows of one speculative request into one attention request; FA3 MLA decode reuses
that grouping through build_fa3_decode_metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.utils.log_utils import init_logger
from . import GQALayerConfig, GQALayerInputs

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo

logger = init_logger(__name__)


@dataclass(frozen=True)
class FA3GQAPrefillMetadata:
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
class FA3DecodeMetadata:
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


# 设为 1 / TRUE / ON 时，FA3 与 CPU cache 搬运在 GPU 上互相等待。
FA3_CPU_CACHE_SYNC = enable_env_vars("LIGHTLLM_FA3_CPU_CACHE_SYNC")


def build_fa3_decode_metadata(backend: AttentionBackend, infer_state: InferStateInfo) -> FA3DecodeMetadata:
    """Group query rows into attention requests and fill their page-table rows.

    backend's init_runtime set page_size. The page table is a view of one fixed buffer per
    microbatch, shared by name with every FA3 decode backend of the manager and allocated on first
    use, so a prefill-only node never allocates it. A CUDA graph step uses a page table as wide as the
    graph capacity and keeps every derived tensor at a fixed address per (microbatch, batch size),
    rewriting them in place on replay steps.
    """
    from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy

    page_size = backend.page_size
    cu_seqlens_q, cu_seqlens_k, cache_seqlens, att_req_idx = _group_rows(infer_state)
    rows = att_req_idx.shape[0]
    width = infer_state.cuda_graph_max_kv_seq_len if infer_state.uses_cuda_graph else infer_state.max_kv_seq_len
    if rows > infer_state.max_decode_rows:
        raise RuntimeError(f"FA3 attention batch size {rows} exceeds page-table capacity {infer_state.max_decode_rows}")
    if width > infer_state.model_max_seq_len:
        raise RuntimeError(
            f"FA3 max KV sequence length {width} exceeds page-table capacity {infer_state.model_max_seq_len}"
        )

    numel = infer_state.max_decode_rows * triton.cdiv(infer_state.model_max_seq_len, page_size)
    buffer = infer_state.attention_backend_manager.get_workspace(
        f"fa3_page_table_{infer_state.microbatch_index}", numel, torch.int32
    )
    max_pages = triton.cdiv(width, page_size)
    page_table = buffer[: rows * max_pages].view(rows, max_pages)
    page_table_copy(
        page_table=page_table[:, : triton.cdiv(infer_state.max_kv_seq_len, page_size)],
        req_to_token_indexs=infer_state.req_to_token_indexs,
        b_req_idx=att_req_idx,
        page_size=page_size,
    )
    return FA3DecodeMetadata(
        page_table=page_table,
        cache_seqlens=backend.update_graph_tensor(infer_state, "cache_seqlens", cache_seqlens),
        cu_seqlens_q=backend.update_graph_tensor(infer_state, "cu_seqlens_q", cu_seqlens_q),
        cu_seqlens_k=backend.update_graph_tensor(infer_state, "cu_seqlens_k", cu_seqlens_k),
        max_seqlen_q=infer_state.max_decode_rows_per_request,
        max_seqlen_k=infer_state.max_kv_seq_len,
        page_size=page_size,
    )


def _group_rows(infer_state: InferStateInfo):
    """Return int32 query/kv offsets, kv lengths and request slots per attention request."""
    from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor

    rows, batch_size = infer_state.max_decode_rows_per_request, infer_state.batch_size
    if infer_state.is_dynamic_verify:
        from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_spec_fa3_decode_params
        from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

        markers = build_mtp_shared_group_markers(infer_state.b_req_idx, hold_req_id=infer_state.pad_req_idx)
        q_lens, kv_lens, att_req_idx, cache_seqlens = build_dynamic_spec_fa3_decode_params(
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            b_mark_mtp_shared_group=markers,
            att_batch_size=batch_size,
            hold_req_id=infer_state.pad_req_idx,
        )
    elif rows > 1:
        assert batch_size % rows == 0, (
            "FA3 fixed-layout decode requires batch_size to be divisible by the rows per request, "
            f"got batch_size={batch_size}, rows={rows}"
        )
        # The last row of each request sees its longest prefix.
        q_lens = torch.full((batch_size // rows,), rows, dtype=torch.int32, device=infer_state.b_seq_len.device)
        kv_lens = infer_state.b_seq_len[rows - 1 :: rows]
        att_req_idx = infer_state.b_req_idx[rows - 1 :: rows]
        cache_seqlens = kv_lens.contiguous()
    else:
        cu_q, cu_kv = infer_state.b1_cu_q_seq_len, infer_state.b1_cu_kv_seq_len
        return cu_q.int(), cu_kv.int(), infer_state.b_seq_len, infer_state.b_req_idx
    cu_seqlens_q, cu_seqlens_k = gen_cumsum_pad0_tensor(q_lens, kv_lens)
    return cu_seqlens_q.int(), cu_seqlens_k.int(), cache_seqlens, att_req_idx


def _check_gqa_config(config: GQALayerConfig) -> None:
    if config.dtype not in (torch.float16, torch.bfloat16) or config.head_dim > 256:
        raise NotImplementedError(
            f"FA3 GQA supports FP16/BF16 with head_dim <= 256, got {config.dtype}, {config.head_dim}"
        )
    if config.has_alibi:
        raise NotImplementedError("FA3 GQA does not support ALiBi")
    if config.kv_quant not in (None, "fp8_per_head"):
        raise NotImplementedError(f"FA3 GQA supports kv_quant None or 'fp8_per_head', got {config.kv_quant!r}")
    plain = config.sliding_window == (-1, -1) and not config.has_sinks
    if config.bidirectional_image_spans and (config.kv_quant is not None or not plain):
        raise NotImplementedError("FA3 GQA supports bidirectional image spans without KV quantization, window or sinks")
    if config.kv_quant is not None and not (plain and config.has_default_scale):
        raise NotImplementedError("FA3 GQA with an FP8 cache supports only the default scale without window or sinks")
    if config.kv_quant is not None and config.dtype != torch.bfloat16:
        # FA3 writes FP8 attention output only as BF16.
        raise NotImplementedError(f"FA3 GQA with an FP8 cache needs a BF16 model, got {config.dtype}")


def _check_inputs(config: GQALayerConfig, layer_inputs: GQALayerInputs) -> None:
    q, scales = layer_inputs.q, layer_inputs.kv_scales
    assert q.shape[1:] == (config.num_heads, config.head_dim) and q.dtype == config.dtype
    assert layer_inputs.k_cache.shape[1:] == (config.num_kv_heads, config.head_dim)
    assert (
        layer_inputs.sink_logits is not None
    ) == config.has_sinks, "sinks must be given exactly when the layer has sinks"
    assert (scales is not None) == (config.kv_quant is not None), "kv_scales must be given exactly with kv_quant"
    if scales is not None:
        assert (scales.q_scale is not None) == config.has_q_scale


def _paged(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    return cache.view(-1, page_size, cache.shape[1], cache.shape[2])


def _fp8_attention(config: GQALayerConfig, layer_inputs: GQALayerInputs, att_metadata, out: torch.Tensor) -> None:
    """Quantize q per KV head and run FA3 over the FP8 cache with [requests, kv_heads] descales."""
    from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
        q_per_head_fp8_quant,
        q_per_head_static_fp8_quant,
    )
    from lightllm.utils.sgl_utils import flash_attn_with_kvcache

    kv_heads, scales = config.num_kv_heads, layer_inputs.kv_scales
    requests = att_metadata.cache_seqlens.shape[0]
    q = layer_inputs.q.reshape(layer_inputs.q.shape[0], kv_heads, -1)
    if scales.q_scale is None:
        # Without a static scale, quantize each request's queries with their own per-head scale.
        cu_seqlens_q = att_metadata.cu_seqlens_q
        q, q_scale = q_per_head_fp8_quant(q=q, seq_lens=cu_seqlens_q[1:] - cu_seqlens_q[:-1], b1_start_loc=cu_seqlens_q)
    else:
        q = q_per_head_static_fp8_quant(q=q, scales=scales.q_scale)
        q_scale = scales.q_scale.view(1, kv_heads).expand(requests, kv_heads)

    flash_attn_with_kvcache(
        q=q.reshape(-1, config.num_heads, config.head_dim),
        k_cache=_paged(layer_inputs.k_cache, att_metadata.page_size).view(torch.float8_e4m3fn),
        v_cache=_paged(layer_inputs.v_cache, att_metadata.page_size).view(torch.float8_e4m3fn),
        q_descale=q_scale.view(requests, kv_heads),
        k_descale=scales.k_scale.view(1, kv_heads).expand(requests, kv_heads),
        v_descale=scales.v_scale.view(1, kv_heads).expand(requests, kv_heads),
        page_table=att_metadata.page_table,
        cache_seqlens=att_metadata.cache_seqlens,
        cu_seqlens_q=att_metadata.cu_seqlens_q,
        cu_seqlens_k_new=att_metadata.cu_seqlens_k,
        max_seqlen_q=att_metadata.max_seqlen_q,
        causal=config.causal,
        window_size=(-1, -1),
        softcap=0.0,
        return_softmax_lse=False,
        out=out,
    )


class FA3GQAPrefillBackend(AttentionBackend[GQALayerConfig, FA3GQAPrefillMetadata, GQALayerInputs]):
    """Supports any scale, causal or non-causal attention, one- or two-sided windows and sinks.

    A per-head FP8 cache (kv_quant='fp8_per_head') supports the default scale without window or
    sinks; q is quantized per KV head with the static q scale, or dynamically per request.
    """

    name = "fa3_gqa_prefill"
    phase = "prefill"

    def __init__(self, layer_config: GQALayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.bidirectional_image_spans:
            from lightllm.utils.neo_fa3_utils import HAS_NEO_FA3, NEO_FA3_INSTALL_HINT

            if not HAS_NEO_FA3:
                raise NotImplementedError(
                    "FA3 bidirectional image spans need FA3 with image_token_end support.\n" + NEO_FA3_INSTALL_HINT
                )

    @classmethod
    def _check_supported(cls, layer_config: GQALayerConfig) -> None:
        _check_gqa_config(layer_config)

    @classmethod
    def metadata_config_key(cls, layer_config: GQALayerConfig):
        return layer_config.kv_quant

    def init_runtime(self, infer_state: InferStateInfo) -> None:
        self.page_size = infer_state.page_size
        if self.layer_config.kv_quant is not None:
            # TODO: use the cache's page size once FP8 FA3 supports multi-token pages.
            logger.warning(f"FP8 FA3 temporarily uses page size 1 with model page_size={infer_state.page_size}")
            self.page_size = 1

    def build_att_metadata(self, infer_state: InferStateInfo) -> FA3GQAPrefillMetadata:
        """Build a fresh page table for the step's requests."""
        from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy

        page_table = torch.empty(
            (infer_state.batch_size, triton.cdiv(infer_state.max_kv_seq_len, self.page_size)),
            dtype=torch.int32,
            device=infer_state.b_seq_len.device,
        )
        page_table_copy(
            page_table=page_table,
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            page_size=self.page_size,
        )
        return FA3GQAPrefillMetadata(
            page_table=page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.b1_cu_q_seq_len.int(),
            cu_seqlens_k=infer_state.b1_cu_kv_seq_len.int(),
            max_seqlen_q=infer_state.max_q_seq_len,
            page_size=self.page_size,
            b_image_token_end=infer_state.b_image_token_end,
        )

    def forward(
        self,
        att_metadata: FA3GQAPrefillMetadata,
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
        config = self.layer_config
        _check_inputs(config, layer_inputs)
        if config.kv_quant is not None:
            _fp8_attention(config, layer_inputs, att_metadata, out)
            return out

        paged_cache = dict(
            q=layer_inputs.q,
            k_cache=_paged(layer_inputs.k_cache, att_metadata.page_size),
            v_cache=_paged(layer_inputs.v_cache, att_metadata.page_size),
            page_table=att_metadata.page_table,
            cache_seqlens=att_metadata.cache_seqlens,
            cu_seqlens_q=att_metadata.cu_seqlens_q,
            cu_seqlens_k_new=att_metadata.cu_seqlens_k,
            max_seqlen_q=att_metadata.max_seqlen_q,
            softmax_scale=config.softmax_scale,
            causal=config.causal,
            softcap=0.0,
            return_softmax_lse=False,
        )
        if config.bidirectional_image_spans:
            from lightllm.utils.neo_fa3_utils import flash_attn_with_kvcache_neo

            assert att_metadata.b_image_token_end is not None, "bidirectional image spans need b_image_token_end"
            # The Neo FA3 build returns its own output, so copy it into out.
            result = flash_attn_with_kvcache_neo(
                **paged_cache,
                window_size=(-1, -1),
                k_descale=None,
                v_descale=None,
                image_token_end=att_metadata.b_image_token_end,
            )
            out.copy_(result)
        else:
            from lightllm.utils.sgl_utils import flash_attn_with_kvcache

            flash_attn_with_kvcache(
                **paged_cache, window_size=config.sliding_window, sinks=layer_inputs.sink_logits, out=out
            )
        return out


class FA3GQADecodeBackend(FA3GQAPrefillBackend):
    name = "fa3_gqa_decode"
    phase = "decode"

    @classmethod
    def _check_supported(cls, layer_config: GQALayerConfig) -> None:
        _check_gqa_config(layer_config)
        if layer_config.bidirectional_image_spans:
            raise NotImplementedError("bidirectional image spans apply to prefill only")

    def build_att_metadata(self, infer_state: InferStateInfo) -> FA3DecodeMetadata:
        return build_fa3_decode_metadata(self, infer_state)

    def forward(
        self,
        att_metadata: FA3DecodeMetadata,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape; layout rules match prefill.

        q has one row per batch row; the attention state groups them into attention requests.
        """
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache_autotune

        config = self.layer_config
        _check_inputs(config, layer_inputs)
        if config.kv_quant is not None:
            _fp8_attention(config, layer_inputs, att_metadata, out)
            return out

        flash_attn_with_kvcache_autotune(
            q=layer_inputs.q,
            k_cache=_paged(layer_inputs.k_cache, att_metadata.page_size),
            v_cache=_paged(layer_inputs.v_cache, att_metadata.page_size),
            page_table=att_metadata.page_table,
            cache_seqlens=att_metadata.cache_seqlens,
            cu_seqlens_q=att_metadata.cu_seqlens_q,
            # The KV is already in the cache, so no new K/V segments are appended.
            cu_seqlens_k_new=None,
            max_seqlen_q=att_metadata.max_seqlen_q,
            max_seqlen_k=att_metadata.max_seqlen_k,
            softmax_scale=config.softmax_scale,
            causal=config.causal,
            window_size=config.sliding_window,
            softcap=0.0,
            return_softmax_lse=False,
            sinks=layer_inputs.sink_logits,
            out=out,
        )
        return out
