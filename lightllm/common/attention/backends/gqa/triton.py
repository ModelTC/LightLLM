"""Triton GQA over a KV cache stored in the model dtype.

Prefill runs the plain, ALiBi or bidirectional image-span kernel; decode runs the MHA, GQA, ALiBi or
speculative shared-group kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from . import GQALayerConfig, GQALayerInputs

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


@dataclass(frozen=True)
class TritonGQAPrefillMetadata:
    """Packed prefill queries: request i's queries start at b_q_start_loc[i] and follow its
    b_ready_cache_len[i] cached tokens. The image-span bounds are per query token (see
    InferStateInfo.b_image_token_end) and only read by layers with bidirectional image spans."""

    b_req_idx: torch.Tensor
    b_q_start_loc: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int
    req_to_token_indexs: torch.Tensor
    b_image_token_end: Optional[torch.Tensor]
    b_image_token_start: Optional[torch.Tensor]


@dataclass(frozen=True)
class TritonGQADecodeMetadata:
    """Attention state for one query per request, in b_req_idx order.

    Index tensors follow InferStateInfo's layout and device.
    b_seq_len gives actual positive lengths, including the current token;
    all addressed token-map entries must identify populated cache slots.

    The two lengths describe the step, not a kernel. kv_seq_len_bound bounds every length
    the attention state may run with: the actual maximum for an eager step, the graph capacity for a
    CUDA graph step, whose replays reuse it. max_kv_seq_len is this step's actual maximum.
    It is valid only as a hint, since replays may see longer lengths: a hint of 33 and a
    bound of 512 remain correct when lengths later reach 300. total_token_bound likewise bounds
    the sum of lengths, and b_kv_start_loc[i] is that sum over the rows before row i.

    With speculative decoding, consecutive rows of one request form a group sharing its KV;
    b_mark_shared_group is 0 inside a group and holds the group size on its last row (padding
    rows form one-row groups). It is None when every request has one row.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    kv_seq_len_bound: int
    max_kv_seq_len: int
    total_token_bound: int = 0
    b_kv_start_loc: Optional[torch.Tensor] = None
    b_mark_shared_group: Optional[torch.Tensor] = None


def _check_gqa_config(config: GQALayerConfig) -> None:
    if config.head_dim not in (16, 32, 64, 128, 256, 512):
        raise NotImplementedError(f"Triton GQA does not support head_dim={config.head_dim}")
    # The Triton GQA kernels compute 1/sqrt(head_dim) themselves and only look left of the query.
    if not config.has_default_scale:
        raise NotImplementedError(
            f"Triton GQA supports only softmax_scale=1/sqrt(head_dim), got {config.softmax_scale}"
        )
    if config.sliding_window[1] > 0:
        raise NotImplementedError(
            f"Triton GQA supports only causal sliding windows (left, 0), got {config.sliding_window}"
        )
    if not config.causal or config.has_sinks:
        raise NotImplementedError("Triton GQA supports only causal attention without sinks")
    if config.kv_quant is not None:
        raise NotImplementedError(f"Triton GQA does not support kv_quant={config.kv_quant!r}")
    if config.has_alibi and config.bidirectional_image_spans:
        raise NotImplementedError("Triton GQA does not combine ALiBi with bidirectional image spans")
    if config.has_alibi and (config.num_heads != config.num_kv_heads or config.sliding_window != (-1, -1)):
        raise NotImplementedError("Triton GQA supports ALiBi only for MHA without sliding window")
    # The ALiBi kernels and the unwindowed image-span kernel only go up to head_dim 256.
    unwindowed_spans = config.bidirectional_image_spans and config.sliding_window == (-1, -1)
    if config.head_dim > 256 and (config.has_alibi or unwindowed_spans):
        raise NotImplementedError(
            f"Triton GQA supports head_dim <= 256 for ALiBi and unwindowed image spans, got {config.head_dim}"
        )


def _check_inputs(config: GQALayerConfig, layer_inputs: GQALayerInputs) -> None:
    q, k = layer_inputs.q, layer_inputs.k_cache
    assert q.shape[1:] == (config.num_heads, config.head_dim), "q does not match the layer_config"
    assert q.dtype == config.dtype, "q does not match the layer_config"
    assert k.shape[1:] == (config.num_kv_heads, config.head_dim), "KV cache does not match the layer_config"
    # A quantized cache needs a kv_quant layer config and the quantized backends.
    assert k.dtype == q.dtype and layer_inputs.v_cache.dtype == q.dtype, "Triton GQA needs a KV cache of q's dtype"
    assert (layer_inputs.alibi_slopes is not None) == config.has_alibi, "alibi_slopes must match has_alibi"


class TritonGQAPrefillBackend(AttentionBackend[GQALayerConfig, TritonGQAPrefillMetadata, GQALayerInputs]):
    name = "triton_gqa_prefill"
    phase = "prefill"

    @classmethod
    def _check_supported(cls, layer_config: GQALayerConfig) -> None:
        """Head dimensions 16, 32, 64, 128, 256 and 512 are supported, with softmax_scale 1/sqrt(head_dim)."""
        _check_gqa_config(layer_config)

    def build_att_metadata(self, infer_state: InferStateInfo) -> TritonGQAPrefillMetadata:
        return TritonGQAPrefillMetadata(
            b_req_idx=infer_state.b_req_idx,
            b_q_start_loc=infer_state.b_q_start_loc,
            b_seq_len=infer_state.b_seq_len,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            max_q_seq_len=infer_state.max_q_seq_len,
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_image_token_end=infer_state.b_image_token_end,
            b_image_token_start=infer_state.b_image_token_start,
        )

    def forward(
        self,
        att_metadata: TritonGQAPrefillMetadata,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out, with q's shape, dtype and device.

        q, K/V and out are matching FP16/BF16 tensors on one CUDA device, with
        unit head-dimension stride and non-overlapping token/head strides that
        may include padding. K/V may be disjoint slices of shared storage;
        arbitrary strided layouts are not supported. The caller must make q.device
        the current CUDA device. out must not alias layer_inputs or att_metadata tensors.
        Prefill needs no scratch and does not call alloc_func.
        """
        config = self.layer_config
        _check_inputs(config, layer_inputs)
        q, k, v = layer_inputs.q, layer_inputs.k_cache, layer_inputs.v_cache

        if config.bidirectional_image_spans:
            assert att_metadata.b_image_token_end is not None, "bidirectional image spans need b_image_token_end"
            args = (
                q,
                k,
                v,
                out,
                att_metadata.b_req_idx,
                att_metadata.b_q_start_loc,
                att_metadata.b_seq_len,
                att_metadata.b_ready_cache_len,
                att_metadata.max_q_seq_len,
                att_metadata.req_to_token_indexs,
                att_metadata.b_image_token_end,
            )
            if config.sliding_window == (-1, -1):
                from lightllm.common.attention.kernels.gqa.bf16.prefill.context_attention_fwd_neo import (
                    context_attention_fwd_neo,
                )

                context_attention_fwd_neo(*args)
            else:
                from lightllm.common.attention.kernels.gqa.bf16.prefill.context_attention_fwd_gemma4_mm import (
                    context_attention_fwd_gemma4_mm,
                )

                # A span may reach back past the window, so the kernel also needs each span's start.
                assert att_metadata.b_image_token_start is not None, "windowed image spans need b_image_token_start"
                context_attention_fwd_gemma4_mm(
                    *args, att_metadata.b_image_token_start, sliding_window=config.sliding_window
                )
        elif config.has_alibi:
            from lightllm.common.attention.kernels.gqa.alibi.context_flashattention_nopad import (
                context_attention_fwd as alibi_context_attention_fwd,
            )

            alibi_context_attention_fwd(
                q,
                k,
                v,
                out,
                att_metadata.b_req_idx,
                layer_inputs.alibi_slopes,
                att_metadata.b_q_start_loc,
                att_metadata.b_seq_len,
                att_metadata.b_ready_cache_len,
                att_metadata.max_q_seq_len,
                att_metadata.req_to_token_indexs,
            )
        else:
            from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
                context_attention_fwd,
            )

            context_attention_fwd(
                q=q,
                k=k,
                v=v,
                o=out,
                b_req_idx=att_metadata.b_req_idx,
                b_start_loc=att_metadata.b_q_start_loc,
                b_seq_len=att_metadata.b_seq_len,
                b_prompt_cache_len=att_metadata.b_ready_cache_len,
                max_input_len=att_metadata.max_q_seq_len,
                req_to_token_indexs=att_metadata.req_to_token_indexs,
                sliding_window=config.sliding_window,
            )
        return out


class TritonGQADecodeBackend(AttentionBackend[GQALayerConfig, TritonGQADecodeMetadata, GQALayerInputs]):
    name = "triton_gqa_decode"
    phase = "decode"

    @classmethod
    def _check_supported(cls, layer_config: GQALayerConfig) -> None:
        """Head dimensions 16, 32, 64, 128 and 256 are supported, plus 512 for GQA, with softmax_scale 1/sqrt(head_dim).

        GQA (num_heads > num_kv_heads) supports sliding windows (-1, -1) and (left, 0);
        MHA (num_heads == num_kv_heads) supports only (-1, -1).
        """
        _check_gqa_config(layer_config)
        if layer_config.bidirectional_image_spans:
            raise NotImplementedError("bidirectional image spans apply to prefill only")
        if layer_config.num_heads == layer_config.num_kv_heads:
            if layer_config.head_dim == 512:
                raise NotImplementedError("Triton MHA decode does not support head_dim=512")
            if layer_config.sliding_window != (-1, -1):
                raise NotImplementedError("sliding_window is not supported by the MHA decode kernel")

    def build_att_metadata(self, infer_state: InferStateInfo) -> TritonGQADecodeMetadata:
        """Kernels read the request tensors directly; a CUDA graph step bounds lengths by the graph
        capacity and keeps the speculative-decode groups per (microbatch, batch size)."""
        if infer_state.uses_cuda_graph:
            bound = infer_state.cuda_graph_max_kv_seq_len
            total_bound = bound * infer_state.batch_size
        else:
            bound, total_bound = infer_state.max_kv_seq_len, infer_state.total_token_num

        markers = None
        if infer_state.max_decode_rows_per_request > 1:
            from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

            markers = build_mtp_shared_group_markers(infer_state.b_req_idx, hold_req_id=infer_state.pad_req_idx)
            markers = self.update_graph_tensor(infer_state, "mtp_groups", markers)

        cu_kv = infer_state.b1_cu_kv_seq_len
        return TritonGQADecodeMetadata(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            kv_seq_len_bound=bound,
            max_kv_seq_len=infer_state.max_kv_seq_len,
            total_token_bound=total_bound,
            b_kv_start_loc=None if cu_kv is None else cu_kv[:-1],
            b_mark_shared_group=markers,
        )

    def forward(
        self,
        att_metadata: TritonGQADecodeMetadata,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out, with q's shape, dtype and device.

        q, K/V and out are matching FP16/BF16 tensors on one CUDA device, with
        unit head-dimension stride and non-overlapping token/head strides that
        may include padding. K/V may be disjoint slices of shared storage;
        arbitrary strided layouts are not supported. The caller must make q.device
        the current CUDA device. out must not alias layer_inputs or att_metadata tensors.
        K and V must have identical strides.
        Scratch uses alloc_func on q.device. A serving allocator with a cache
        scoped to one device must be used only on that device.
        """
        config = self.layer_config
        _check_inputs(config, layer_inputs)
        q, k, v = layer_inputs.q, layer_inputs.k_cache, layer_inputs.v_cache
        request_tensors = dict(
            req_to_token_indexs=att_metadata.req_to_token_indexs,
            b_req_idx=att_metadata.b_req_idx,
            b_seq_len=att_metadata.b_seq_len,
        )
        shares_kv_groups = (
            att_metadata.b_mark_shared_group is not None
            and config.sliding_window == (-1, -1)
            and config.head_dim <= 256
        )

        if config.has_alibi:
            from lightllm.common.attention.kernels.gqa.alibi.token_flashattention_nopad import (
                token_attention_fwd,
            )

            token_attention_fwd(
                q,
                k,
                v,
                out,
                layer_inputs.alibi_slopes,
                att_metadata.req_to_token_indexs,
                att_metadata.b_req_idx,
                att_metadata.b_kv_start_loc,
                att_metadata.b_seq_len,
                # Scratch and launch sizes must cover every length a replay may bring.
                att_metadata.kv_seq_len_bound,
                att_metadata.total_token_bound,
                alloc_tensor_func=alloc_func,
            )
        elif shares_kv_groups:
            from lightllm.common.attention.kernels.gqa.bf16.decode.mtp_diverse import (
                token_decode_attention_mtp_diverse_single_token,
            )

            # The rows of one request share its KV, so one pass over the KV serves the whole group.
            token_decode_attention_mtp_diverse_single_token(
                q=q,
                k=k,
                v=v,
                Req_to_tokens=att_metadata.req_to_token_indexs,
                B_req_idx=att_metadata.b_req_idx,
                b_seq_len=att_metadata.b_seq_len,
                b_mark_shared_group=att_metadata.b_mark_shared_group,
                # Only selects kernel configs; the kernel reads actual lengths on the GPU.
                max_kv_len=att_metadata.max_kv_seq_len,
                out=out,
                alloc_tensor_func=alloc_func,
            )
        elif config.num_heads == config.num_kv_heads:
            from lightllm.common.attention.kernels.gqa.bf16.decode.flash_decoding import (
                token_decode_attention_flash_decoding,
            )

            token_decode_attention_flash_decoding(
                q=q,
                cache_k=k,
                cache_v=v,
                **request_tensors,
                # Scratch size and launch grid must cover every length a replay may bring.
                max_kv_seq_len=att_metadata.kv_seq_len_bound,
                out=out,
                alloc_tensor_func=alloc_func,
            )
        else:
            from lightllm.common.attention.kernels.gqa.bf16.decode.gqa_flash_decoding import (
                gqa_token_decode_attention_flash_decoding,
            )

            gqa_token_decode_attention_flash_decoding(
                q=q,
                cache_k=k,
                cache_v=v,
                **request_tensors,
                # Only selects a kernel config; the GQA kernel reads actual lengths on the GPU.
                max_len_in_batch=att_metadata.max_kv_seq_len,
                out=out,
                alloc_tensor_func=alloc_func,
                sliding_window=config.sliding_window,
            )
        return out
