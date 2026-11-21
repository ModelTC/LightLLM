from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionStepBuilder,
    AttentionBatch,
    TensorAllocator,
)
from .. import keep_for_cuda_graph
from . import GQALayerConfig, GQALayerInputs


@dataclass(frozen=True)
class TritonGQAPrefillStepArgs:
    """Step args for packed queries after each request's cached prefix.

    req_to_token_indexs[request_id, position] is a physical KV cache slot,
    not a byte offset. It is a 2-D int32/int64 tensor with unit sequence stride.
    b_req_idx selects its rows; all b_* tensors are contiguous int32/int64
    vectors of batch_size elements on q.device, as is the token map.

    Request i has b_seq_len[i] total tokens including b_ready_cache_len[i]
    cached tokens. Its remaining queries occupy the packed q/out range
    starting at b_q_start_loc[i]. These ranges must be disjoint and cover q.
    max_q_seq_len bounds every b_seq_len[i] - b_ready_cache_len[i] for launch
    coverage; the caller supplies it without reading GPU lengths in forward.
    All addressed token-map entries must identify populated cache slots.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_q_start_loc: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int
    # Per query token: exclusive end of its image span, 0 for text, and the span's start (see AttentionBatch).
    b_image_token_end: Optional[torch.Tensor] = None
    b_image_token_start: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TritonGQADecodeStepArgs:
    """Step args for one query per request, in b_req_idx order.

    Index tensors follow TritonGQAPrefillStepArgs's layout/device contract.
    b_seq_len gives actual positive lengths, including the current token;
    all addressed token-map entries must identify populated cache slots.

    The two lengths describe the step, not a kernel. kv_seq_len_bound bounds every length
    the step args may run with: the actual maximum for an eager step, the graph capacity for a
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


class TritonGQAPrefillStepBuilder(AttentionStepBuilder[TritonGQAPrefillStepArgs]):
    def build(self, batch: AttentionBatch) -> TritonGQAPrefillStepArgs:
        """Kernels read the request tensors directly, so the step args only record them."""
        assert batch.b_q_start_loc is not None and batch.b_ready_cache_len is not None, "prefill batch fields required"
        return TritonGQAPrefillStepArgs(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_seq_len=batch.b_seq_len,
            b_q_start_loc=batch.b_q_start_loc,
            b_ready_cache_len=batch.b_ready_cache_len,
            max_q_seq_len=batch.max_q_seq_len,
            b_image_token_end=batch.b_image_token_end,
            b_image_token_start=batch.b_image_token_start,
        )


class TritonGQADecodeStepBuilder(AttentionStepBuilder[TritonGQADecodeStepArgs]):
    def __init__(self) -> None:
        self._cuda_graph_tensors: dict[tuple, torch.Tensor] = {}

    def build(self, batch: AttentionBatch) -> TritonGQADecodeStepArgs:
        """Kernels read the request tensors directly; a CUDA graph step bounds lengths by the graph
        capacity and keeps the speculative-decode groups per (microbatch, batch size)."""
        if batch.uses_cuda_graph:
            bound, total_bound = batch.cuda_graph_max_kv_seq_len, batch.cuda_graph_max_kv_seq_len * batch.batch_size
        else:
            bound, total_bound = batch.max_kv_seq_len, batch.total_token_num
        markers = None
        if batch.decode_rows_per_request > 1:
            from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

            markers = build_mtp_shared_group_markers(batch.b_req_idx, hold_req_id=batch.pad_req_idx)
            if batch.uses_cuda_graph:
                key = (batch.microbatch_index, batch.batch_size, "mtp_groups")
                markers = keep_for_cuda_graph(self._cuda_graph_tensors, key, markers)
        return TritonGQADecodeStepArgs(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_seq_len=batch.b_seq_len,
            kv_seq_len_bound=bound,
            max_kv_seq_len=batch.max_kv_seq_len,
            total_token_bound=total_bound,
            b_kv_start_loc=None if batch.cu_kv_seq_len is None else batch.cu_kv_seq_len[:-1],
            b_mark_shared_group=markers,
        )


def _check_supported(layer_config: GQALayerConfig) -> None:
    # The Triton GQA kernels compute 1/sqrt(head_dim) themselves and only look left of the query.
    if not layer_config.has_default_scale:
        raise NotImplementedError(
            f"Triton GQA supports only softmax_scale=1/sqrt(head_dim), got {layer_config.softmax_scale}"
        )
    if layer_config.sliding_window[1] > 0:
        raise NotImplementedError(
            f"Triton GQA supports only causal sliding windows (left, 0), got {layer_config.sliding_window}"
        )
    if not layer_config.causal or layer_config.has_sinks:
        raise NotImplementedError("Triton GQA supports only causal attention without sinks")
    if layer_config.kv_quant is not None:
        raise NotImplementedError(f"Triton GQA does not support kv_quant={layer_config.kv_quant!r}")
    if layer_config.bidirectional_image_spans and layer_config.has_alibi:
        raise NotImplementedError("Triton GQA does not combine ALiBi with bidirectional image spans")
    if layer_config.has_alibi and (
        layer_config.num_heads != layer_config.num_kv_heads or layer_config.sliding_window != (-1, -1)
    ):
        raise NotImplementedError("Triton GQA supports ALiBi only for MHA without sliding window")


def _check_inputs(layer_config: GQALayerConfig, layer_inputs: GQALayerInputs) -> None:
    q, k = layer_inputs.q, layer_inputs.k_cache
    assert (
        q.shape[1:] == (layer_config.num_heads, layer_config.head_dim) and q.dtype == layer_config.dtype
    ), "q does not match the layer_config"
    assert k.shape[1:] == (layer_config.num_kv_heads, layer_config.head_dim), "KV cache does not match the layer_config"
    assert (layer_inputs.alibi_slopes is not None) == layer_config.has_alibi, "alibi_slopes must match has_alibi"


class TritonGQAPrefillImpl(AttentionImpl[GQALayerConfig, TritonGQAPrefillStepArgs, GQALayerInputs]):
    def __init__(self, layer_config: GQALayerConfig) -> None:
        """Head dimensions 16, 32, 64, 128, 256 and 512 are supported, with softmax_scale 1/sqrt(head_dim)."""
        super().__init__(layer_config)
        _check_supported(layer_config)

    def forward(
        self,
        step_args: TritonGQAPrefillStepArgs,
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
        the current CUDA device. out must not alias layer_inputs or step_args tensors.
        Prefill needs no scratch and does not call alloc_func.
        """
        from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
            context_attention_fwd,
        )

        _check_inputs(self.layer_config, layer_inputs)
        if self.layer_config.bidirectional_image_spans:
            assert step_args.b_image_token_end is not None, "bidirectional image spans need b_image_token_end"
            args = (
                layer_inputs.q,
                layer_inputs.k_cache,
                layer_inputs.v_cache,
                out,
                step_args.b_req_idx,
                step_args.b_q_start_loc,
                step_args.b_seq_len,
                step_args.b_ready_cache_len,
                step_args.max_q_seq_len,
                step_args.req_to_token_indexs,
                step_args.b_image_token_end,
            )
            if self.layer_config.sliding_window == (-1, -1):
                from lightllm.common.attention.kernels.gqa.bf16.prefill.context_attention_fwd_neo import (
                    context_attention_fwd_neo,
                )

                context_attention_fwd_neo(*args)
            else:
                from lightllm.common.attention.kernels.gqa.bf16.prefill.context_attention_fwd_gemma4_mm import (
                    context_attention_fwd_gemma4_mm,
                )

                # A span may reach back past the window, so the kernel also needs each span's start.
                assert step_args.b_image_token_start is not None, "windowed image spans need b_image_token_start"
                context_attention_fwd_gemma4_mm(
                    *args, step_args.b_image_token_start, sliding_window=self.layer_config.sliding_window
                )
            return out
        if self.layer_config.has_alibi:
            from lightllm.common.basemodel.triton_kernel.alibi_att.context_flashattention_nopad import (
                context_attention_fwd as alibi_context_attention_fwd,
            )

            alibi_context_attention_fwd(
                layer_inputs.q,
                layer_inputs.k_cache,
                layer_inputs.v_cache,
                out,
                step_args.b_req_idx,
                layer_inputs.alibi_slopes,
                step_args.b_q_start_loc,
                step_args.b_seq_len,
                step_args.b_ready_cache_len,
                step_args.max_q_seq_len,
                step_args.req_to_token_indexs,
            )
            return out
        context_attention_fwd(
            q=layer_inputs.q,
            k=layer_inputs.k_cache,
            v=layer_inputs.v_cache,
            o=out,
            b_req_idx=step_args.b_req_idx,
            b_start_loc=step_args.b_q_start_loc,
            b_seq_len=step_args.b_seq_len,
            b_prompt_cache_len=step_args.b_ready_cache_len,
            max_input_len=step_args.max_q_seq_len,
            req_to_token_indexs=step_args.req_to_token_indexs,
            sliding_window=self.layer_config.sliding_window,
        )
        return out


class TritonGQADecodeImpl(AttentionImpl[GQALayerConfig, TritonGQADecodeStepArgs, GQALayerInputs]):
    def __init__(self, layer_config: GQALayerConfig) -> None:
        """Head dimensions 16, 32, 64, 128 and 256 are supported, plus 512 for GQA, with softmax_scale 1/sqrt(head_dim).

        GQA (num_heads > num_kv_heads) supports sliding windows (-1, -1) and (left, 0);
        MHA (num_heads == num_kv_heads) supports only (-1, -1).
        """
        super().__init__(layer_config)
        _check_supported(layer_config)
        if layer_config.bidirectional_image_spans:
            raise NotImplementedError("bidirectional image spans apply to prefill only")
        self.is_mha = layer_config.num_heads == layer_config.num_kv_heads
        if self.is_mha and layer_config.sliding_window != (-1, -1):
            raise NotImplementedError("sliding_window is not supported by the MHA decode kernel")

    def forward(
        self,
        step_args: TritonGQADecodeStepArgs,
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
        the current CUDA device. out must not alias layer_inputs or step_args tensors.
        K and V must have identical strides.
        Scratch uses alloc_func on q.device. A serving allocator with a cache
        scoped to one device must be used only on that device.
        """
        _check_inputs(self.layer_config, layer_inputs)
        if self.layer_config.has_alibi:
            from lightllm.common.basemodel.triton_kernel.alibi_att.token_flashattention_nopad import (
                token_attention_fwd,
            )

            token_attention_fwd(
                layer_inputs.q,
                layer_inputs.k_cache,
                layer_inputs.v_cache,
                out,
                layer_inputs.alibi_slopes,
                step_args.req_to_token_indexs,
                step_args.b_req_idx,
                step_args.b_kv_start_loc,
                step_args.b_seq_len,
                # Scratch and launch sizes must cover every length a replay may bring.
                step_args.kv_seq_len_bound,
                step_args.total_token_bound,
                alloc_tensor_func=alloc_func,
            )
            return out
        # The shared-group kernel has no sliding window; windowed layers decode each row on its own.
        if step_args.b_mark_shared_group is not None and self.layer_config.sliding_window == (-1, -1):
            from lightllm.common.attention.kernels.gqa.bf16.decode.mtp_diverse import (
                token_decode_attention_mtp_diverse_single_token,
            )

            # The rows of one request share its KV, so one pass over the KV serves the whole group.
            return token_decode_attention_mtp_diverse_single_token(
                q=layer_inputs.q,
                k=layer_inputs.k_cache,
                v=layer_inputs.v_cache,
                Req_to_tokens=step_args.req_to_token_indexs,
                B_req_idx=step_args.b_req_idx,
                b_seq_len=step_args.b_seq_len,
                b_mark_shared_group=step_args.b_mark_shared_group,
                # Only selects kernel configs; the kernel reads actual lengths on the GPU.
                max_kv_len=step_args.max_kv_seq_len,
                out=out,
                alloc_tensor_func=alloc_func,
            )
        if self.is_mha:
            from lightllm.common.attention.kernels.gqa.bf16.decode.flash_decoding import (
                token_decode_attention_flash_decoding,
            )

            return token_decode_attention_flash_decoding(
                q=layer_inputs.q,
                cache_k=layer_inputs.k_cache,
                cache_v=layer_inputs.v_cache,
                req_to_token_indexs=step_args.req_to_token_indexs,
                b_req_idx=step_args.b_req_idx,
                b_seq_len=step_args.b_seq_len,
                # Scratch size and launch grid must cover every length a replay may bring.
                max_kv_seq_len=step_args.kv_seq_len_bound,
                out=out,
                alloc_tensor_func=alloc_func,
            )
        from lightllm.common.attention.kernels.gqa.bf16.decode.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        return gqa_token_decode_attention_flash_decoding(
            q=layer_inputs.q,
            cache_k=layer_inputs.k_cache,
            cache_v=layer_inputs.v_cache,
            req_to_token_indexs=step_args.req_to_token_indexs,
            b_req_idx=step_args.b_req_idx,
            b_seq_len=step_args.b_seq_len,
            # Only selects a kernel config; the GQA kernel reads actual lengths on the GPU.
            max_len_in_batch=step_args.max_kv_seq_len,
            out=out,
            alloc_tensor_func=alloc_func,
            sliding_window=self.layer_config.sliding_window,
        )


class TritonGQAPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_gqa_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[TritonGQAPrefillStepBuilder]:
        return TritonGQAPrefillStepBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonGQAPrefillImpl]:
        return TritonGQAPrefillImpl


class TritonGQADecodeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_gqa_decode"

    @staticmethod
    def get_step_builder_cls() -> type[TritonGQADecodeStepBuilder]:
        return TritonGQADecodeStepBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonGQADecodeImpl]:
        return TritonGQADecodeImpl
