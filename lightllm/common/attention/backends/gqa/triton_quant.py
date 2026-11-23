"""Triton GQA over an int8/int4 group-quantized KV cache (GQALayerConfig.kv_quant "int8_group"/"int4_group")."""

from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBatch,
    AttentionImpl,
    AttentionStepBuilder,
    StepBuilderContext,
    TensorAllocator,
)
from .. import keep_for_cuda_graph
from . import GQALayerConfig, GQALayerInputs

_KINDS = ("int8_group", "int4_group")


@dataclass(frozen=True)
class TritonQuantGQAPrefillStepArgs:
    """Step args for packed prefill queries; request tensors follow TritonGQAPrefillStepArgs.

    Prefill dequantizes every request's KV into contiguous scratch first: request i's tokens land at
    b_kv_start_loc[i] of a [total_token_num, num_kv_heads, head_dim] buffer. max_kv_seq_len bounds
    each request's length.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_q_start_loc: torch.Tensor
    b_ready_cache_len: torch.Tensor
    b_kv_start_loc: torch.Tensor
    max_q_seq_len: int
    max_kv_seq_len: int
    total_token_num: int


@dataclass(frozen=True)
class TritonQuantGQADecodeStepArgs:
    """Step args for one query per batch row; lengths follow TritonGQADecodeStepArgs.

    With diverse sampling (int8 only), b_mark_shared_group marks the rows that start a group of
    rows sharing a prefix, and b_shared_seq_len gives each row's shared prefix length (0 for a
    one-row group); both are None otherwise.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    kv_seq_len_bound: int
    max_kv_seq_len: int
    b_shared_seq_len: Optional[torch.Tensor] = None
    b_mark_shared_group: Optional[torch.Tensor] = None


class TritonQuantGQAPrefillStepBuilder(AttentionStepBuilder[TritonQuantGQAPrefillStepArgs]):
    def build(self, batch: AttentionBatch) -> TritonQuantGQAPrefillStepArgs:
        assert batch.b_q_start_loc is not None and batch.b_ready_cache_len is not None, "prefill batch fields required"
        b_seq_len = batch.b_seq_len
        return TritonQuantGQAPrefillStepArgs(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_seq_len=b_seq_len,
            b_q_start_loc=batch.b_q_start_loc,
            b_ready_cache_len=batch.b_ready_cache_len,
            b_kv_start_loc=torch.cumsum(b_seq_len, dim=0, dtype=b_seq_len.dtype) - b_seq_len,
            max_q_seq_len=batch.max_q_seq_len,
            max_kv_seq_len=batch.max_kv_seq_len,
            total_token_num=batch.total_token_num,
        )


class TritonQuantGQADecodeStepBuilder(AttentionStepBuilder[TritonQuantGQADecodeStepArgs]):
    def __init__(self, *, diverse: bool = False) -> None:
        """diverse=True builds the shared-prefix groups for the int8 diverse-sampling decode kernel."""
        self.diverse = diverse
        self._cuda_graph_tensors: dict[tuple, torch.Tensor] = {}

    def build(self, batch: AttentionBatch) -> TritonQuantGQADecodeStepArgs:
        """A CUDA graph step bounds lengths by the graph capacity and keeps the group tensors per
        (microbatch, batch size), rewriting them in place on replay."""
        bound = batch.cuda_graph_max_kv_seq_len if batch.uses_cuda_graph else batch.max_kv_seq_len
        b_shared_seq_len = b_mark_shared_group = None
        if self.diverse:
            from lightllm.common.basemodel.triton_kernel.diverse_utils import build_diverse_shared_group_markers

            assert batch.b_shared_seq_len is not None and batch.b_shared_radix_node_id is not None
            b_mark_shared_group = build_diverse_shared_group_markers(
                b_shared_radix_node_id=batch.b_shared_radix_node_id
            )
            # A one-row group has no cross-request prefix sharing to accelerate.
            b_shared_seq_len = torch.where(b_mark_shared_group == 1, 0, batch.b_shared_seq_len)
            if batch.uses_cuda_graph:
                key, store = (batch.microbatch_index, batch.batch_size), self._cuda_graph_tensors
                b_mark_shared_group = keep_for_cuda_graph(store, (*key, "mark"), b_mark_shared_group)
                b_shared_seq_len = keep_for_cuda_graph(store, (*key, "shared_len"), b_shared_seq_len)
        return TritonQuantGQADecodeStepArgs(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_seq_len=batch.b_seq_len,
            kv_seq_len_bound=bound,
            max_kv_seq_len=batch.max_kv_seq_len,
            b_shared_seq_len=b_shared_seq_len,
            b_mark_shared_group=b_mark_shared_group,
        )


def _check_supported(layer_config: GQALayerConfig) -> None:
    if layer_config.kv_quant not in _KINDS:
        raise NotImplementedError(f"Triton quantized GQA supports kv_quant {_KINDS}, got {layer_config.kv_quant!r}")
    if not (
        layer_config.causal
        and layer_config.sliding_window == (-1, -1)
        and layer_config.has_default_scale
        and not layer_config.has_sinks
        and not layer_config.has_alibi
        and not layer_config.bidirectional_image_spans
    ):
        raise NotImplementedError(
            "Triton quantized GQA supports only causal attention with softmax_scale 1/sqrt(head_dim), "
            "without sliding window or sinks"
        )


def _cache_operands(layer_config: GQALayerConfig, layer_inputs: GQALayerInputs):
    q, k = layer_inputs.q, layer_inputs.k_cache
    assert q.shape[1:] == (layer_config.num_heads, layer_config.head_dim) and q.dtype == layer_config.dtype
    packed_dim = layer_config.head_dim // 2 if layer_config.kv_quant == "int4_group" else layer_config.head_dim
    assert k.shape[1:] == (layer_config.num_kv_heads, packed_dim), "KV cache does not match the layer_config"
    scales = layer_inputs.kv_scales
    assert scales is not None, "a quantized KV cache needs kv_scales"
    groups = layer_config.head_dim // layer_config.kv_quant_group_size
    assert scales.k_scale.shape[1:] == (layer_config.num_kv_heads, groups), "scales do not match the layer_config"
    return k, scales.k_scale, layer_inputs.v_cache, scales.v_scale


class TritonQuantGQAPrefillImpl(AttentionImpl[GQALayerConfig, TritonQuantGQAPrefillStepArgs, GQALayerInputs]):
    def __init__(self, layer_config: GQALayerConfig) -> None:
        super().__init__(layer_config)
        _check_supported(layer_config)

    def forward(
        self,
        step_args: TritonQuantGQAPrefillStepArgs,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Dequantize the batch's KV into alloc_func scratch, then run contiguous-KV prefill into out."""
        from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
            context_attention_fwd_contiguous_kv,
        )

        layer_config = self.layer_config
        k, k_scale, v, v_scale = _cache_operands(layer_config, layer_inputs)
        q = layer_inputs.q
        shape = (step_args.total_token_num, layer_config.num_kv_heads, layer_config.head_dim)
        k_dequant = alloc_func(shape, dtype=q.dtype, device=q.device)
        v_dequant = alloc_func(shape, dtype=q.dtype, device=q.device)
        if layer_config.kv_quant == "int8_group":
            from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import dequantize_int8kv as dequant
        else:
            from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import dequantize_int4kv as dequant
        dequant(
            k=k,
            k_scale=k_scale,
            v=v,
            v_scale=v_scale,
            req_to_token_indexs=step_args.req_to_token_indexs,
            b_seq_len=step_args.b_seq_len,
            b_req_idx=step_args.b_req_idx,
            b_kv_start_loc=step_args.b_kv_start_loc,
            k_out=k_dequant,
            v_out=v_dequant,
            max_len_in_batch=step_args.max_kv_seq_len,
            quant_group_size=layer_config.kv_quant_group_size,
        )
        context_attention_fwd_contiguous_kv(
            q=q,
            k=k_dequant,
            v=v_dequant,
            o=out,
            b_start_loc=step_args.b_q_start_loc,
            b_kv_start_loc=step_args.b_kv_start_loc,
            b_seq_len=step_args.b_seq_len,
            max_q_input_len=step_args.max_q_seq_len,
            b_prompt_cache_len=step_args.b_ready_cache_len,
        )
        return out


class TritonQuantGQADecodeImpl(AttentionImpl[GQALayerConfig, TritonQuantGQADecodeStepArgs, GQALayerInputs]):
    def __init__(self, layer_config: GQALayerConfig) -> None:
        super().__init__(layer_config)
        _check_supported(layer_config)

    def forward(
        self,
        step_args: TritonQuantGQADecodeStepArgs,
        layer_inputs: GQALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape; scratch uses alloc_func."""
        layer_config = self.layer_config
        k, k_scale, v, v_scale = _cache_operands(layer_config, layer_inputs)
        common = dict(
            q=layer_inputs.q,
            req_to_token_indexs=step_args.req_to_token_indexs,
            b_req_idx=step_args.b_req_idx,
            b_seq_len=step_args.b_seq_len,
            cache_k=k,
            cache_k_scale=k_scale,
            cache_v=v,
            cache_v_scale=v_scale,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        if step_args.b_mark_shared_group is not None:
            assert layer_config.kv_quant == "int8_group", "diverse decode supports only int8 caches"
            from lightllm.common.attention.kernels.gqa.int8kv.int8kv_flash_decoding_diverse import (
                token_decode_attention_flash_decoding,
            )

            return token_decode_attention_flash_decoding(
                **common,
                b_shared_seq_len=step_args.b_shared_seq_len,
                b_mark_shared_group=step_args.b_mark_shared_group,
                # Scratch size and launch grid must cover every length a replay may bring.
                max_kv_seq_len=step_args.kv_seq_len_bound,
            )
        if layer_config.kv_quant == "int8_group":
            from lightllm.common.attention.kernels.gqa.int8kv.int8kv_flash_decoding import (
                token_decode_attention_flash_decoding,
            )

            return token_decode_attention_flash_decoding(**common, max_len_in_batch=step_args.max_kv_seq_len)
        from lightllm.common.attention.kernels.gqa.int4kv.ppl_int4kv_flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        return token_decode_attention_flash_decoding(**common, max_kv_seq_len=step_args.max_kv_seq_len)


class TritonQuantGQAPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_quant_gqa_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[TritonQuantGQAPrefillStepBuilder]:
        return TritonQuantGQAPrefillStepBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonQuantGQAPrefillImpl]:
        return TritonQuantGQAPrefillImpl

    @classmethod
    def create_step_builder(
        cls, layer_config: GQALayerConfig, ctx: StepBuilderContext
    ) -> TritonQuantGQAPrefillStepBuilder:
        return TritonQuantGQAPrefillStepBuilder()


class TritonQuantGQADecodeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_quant_gqa_decode"

    @staticmethod
    def get_step_builder_cls() -> type[TritonQuantGQADecodeStepBuilder]:
        return TritonQuantGQADecodeStepBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonQuantGQADecodeImpl]:
        return TritonQuantGQADecodeImpl

    @classmethod
    def create_step_builder(
        cls, layer_config: GQALayerConfig, ctx: StepBuilderContext
    ) -> TritonQuantGQADecodeStepBuilder:
        # Only the int8 decode kernel has a diverse-sampling variant.
        return TritonQuantGQADecodeStepBuilder(diverse=ctx.diverse_decode and layer_config.kv_quant == "int8_group")

    @classmethod
    def step_builder_key(cls, layer_config: GQALayerConfig):
        return layer_config.kv_quant
