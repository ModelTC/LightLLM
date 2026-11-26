"""针对int8/int4组量化KV缓存的Triton GQA（GQAAttConfig.kv_quant "int8_group"/"int4_group"）。"""
from collections.abc import Callable

from lightllm.common.basemodel.infer_struct import InferStateInfo

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from . import GQAAttConfig, GQAAttInputs

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


@dataclass(frozen=True)
class TritonQuantGQAPrefillAttState:
    """为打包后的 prefill query 准备 attention 状态，请求张量来自 InferStateInfo。

    prefill 先将每个请求的 KV 反量化到连续的临时缓冲区。缓冲区形状为
    [total_token_num, num_kv_heads, head_dim]，请求 i 的 token 从 b_kv_start_loc[i]
    开始存放。max_kv_seq_len 是各请求长度的上界。
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
class TritonQuantGQADecodeAttState:
    """每批行一个查询的注意力状态；长度遵循TritonGQADecodeAttState。

    对于多样采样（仅int8），b_mark_shared_group标记开始共享前缀的行，b_shared_seq_len给出每行的共享前缀长度（单行组为0）；否则两者均为None。
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    kv_seq_len_bound: int
    max_kv_seq_len: int
    b_shared_seq_len: Optional[torch.Tensor] = None
    b_mark_shared_group: Optional[torch.Tensor] = None


class TritonQuantGQAPrefillBackend(PrefillAttBackend[GQAAttConfig, TritonQuantGQAPrefillAttState, GQAAttInputs]):
    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        TritonQuantGQAPrefillBackend._check_supported(att_config=att_config, head_dims=(64, 128, 256))

    def prefill(
        self,
        att_state: TritonQuantGQAPrefillAttState,
        layer_inputs: GQAAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """将批次的KV反量化到alloc_func临时缓冲区中，然后运行连续KV的prefill到out。"""
        from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
            context_attention_fwd_contiguous_kv,
        )

        att_config = self.att_config
        k, k_scale, v, v_scale = TritonQuantGQAPrefillBackend._cache_operands(
            att_config=att_config, layer_inputs=layer_inputs
        )
        q = layer_inputs.q
        shape = (att_state.total_token_num, att_config.num_kv_heads, att_config.head_dim)
        k_dequant = alloc_func(shape, dtype=q.dtype, device=q.device)
        v_dequant = alloc_func(shape, dtype=q.dtype, device=q.device)
        if att_config.kv_quant == "int8_group":
            from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import dequantize_int8kv as dequant
        else:
            from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import dequantize_int4kv as dequant
        dequant(
            k=k,
            k_scale=k_scale,
            v=v,
            v_scale=v_scale,
            req_to_token_indexs=att_state.req_to_token_indexs,
            b_seq_len=att_state.b_seq_len,
            b_req_idx=att_state.b_req_idx,
            b_kv_start_loc=att_state.b_kv_start_loc,
            k_out=k_dequant,
            v_out=v_dequant,
            max_len_in_batch=att_state.max_kv_seq_len,
            quant_group_size=att_config.kv_quant_group_size,
        )
        context_attention_fwd_contiguous_kv(
            q=q,
            k=k_dequant,
            v=v_dequant,
            o=out,
            b_start_loc=att_state.b_q_start_loc,
            b_kv_start_loc=att_state.b_kv_start_loc,
            b_seq_len=att_state.b_seq_len,
            max_q_input_len=att_state.max_q_seq_len,
            b_prompt_cache_len=att_state.b_ready_cache_len,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "triton_quant_gqa_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> TritonQuantGQAPrefillAttState:
        assert (
            infer_state.b_q_start_loc is not None and infer_state.b_ready_cache_len is not None
        ), "prefill metadata fields required"
        b_seq_len = infer_state.b_seq_len
        return TritonQuantGQAPrefillAttState(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=b_seq_len,
            b_q_start_loc=infer_state.b_q_start_loc,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            b_kv_start_loc=torch.cumsum(b_seq_len, dim=0, dtype=b_seq_len.dtype) - b_seq_len,
            max_q_seq_len=infer_state.max_q_seq_len,
            max_kv_seq_len=infer_state.max_kv_seq_len,
            total_token_num=infer_state.total_token_num,
        )

    @staticmethod
    def _check_supported(att_config: GQAAttConfig, head_dims: tuple[int, ...]) -> None:
        if att_config.kv_quant not in _KINDS:
            raise NotImplementedError(f"Triton quantized GQA supports kv_quant {_KINDS}, got {att_config.kv_quant!r}")
        if att_config.head_dim not in head_dims:
            raise NotImplementedError(
                f"Triton quantized GQA supports head dimensions {head_dims}, got {att_config.head_dim}"
            )
        if not (
            att_config.causal
            and att_config.sliding_window == (-1, -1)
            and att_config.has_default_scale
            and not att_config.has_sinks
            and not att_config.has_alibi
            and not att_config.bidirectional_image_spans
        ):
            raise NotImplementedError(
                "Triton quantized GQA supports only causal attention with softmax_scale 1/sqrt(head_dim), "
                "without sliding window or sinks"
            )

    @staticmethod
    def _cache_operands(att_config: GQAAttConfig, layer_inputs: GQAAttInputs):
        q, k = layer_inputs.q, layer_inputs.k_cache
        assert q.shape[1:] == (att_config.num_heads, att_config.head_dim) and q.dtype == att_config.dtype
        packed_dim = att_config.head_dim // 2 if att_config.kv_quant == "int4_group" else att_config.head_dim
        assert k.shape[1:] == (att_config.num_kv_heads, packed_dim), "KV cache does not match the att_config"
        scales = layer_inputs.kv_scales
        assert scales is not None, "a quantized KV cache needs kv_scales"
        groups = att_config.head_dim // att_config.kv_quant_group_size
        assert scales.k_scale.shape[1:] == (att_config.num_kv_heads, groups), "scales do not match the att_config"
        return k, scales.k_scale, layer_inputs.v_cache, scales.v_scale


class TritonQuantGQADecodeBackend(DecodeAttBackend[GQAAttConfig, TritonQuantGQADecodeAttState, GQAAttInputs]):
    @dataclass(frozen=True)
    class RuntimeConfig(AttRuntimeConfig):
        diverse_decode: bool = False

    @classmethod
    def create_runtime_config(cls, model: "TpPartBaseModel") -> RuntimeConfig:
        return replace(cls.RuntimeConfig.from_model(model=model), diverse_decode=model.args.diverse_mode)

    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: RuntimeConfig = RuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        TritonQuantGQAPrefillBackend._check_supported(att_config=att_config, head_dims=(16, 32, 64, 128, 256))

    def decode(
        self,
        att_state: TritonQuantGQADecodeAttState,
        layer_inputs: GQAAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，使其具有 q 的形状；scratch 使用 alloc_func 分配。"""
        att_config = self.att_config
        k, k_scale, v, v_scale = TritonQuantGQAPrefillBackend._cache_operands(
            att_config=att_config, layer_inputs=layer_inputs
        )
        if att_state.b_mark_shared_group is not None:
            assert att_config.kv_quant == "int8_group", "diverse decode supports only int8 caches"
            from lightllm.common.attention.kernels.gqa.int8kv.int8kv_flash_decoding_diverse import (
                token_decode_attention_flash_decoding,
            )

            return token_decode_attention_flash_decoding(
                q=layer_inputs.q,
                req_to_token_indexs=att_state.req_to_token_indexs,
                b_req_idx=att_state.b_req_idx,
                b_seq_len=att_state.b_seq_len,
                cache_k=k,
                cache_k_scale=k_scale,
                cache_v=v,
                cache_v_scale=v_scale,
                out=out,
                alloc_tensor_func=alloc_func,
                b_shared_seq_len=att_state.b_shared_seq_len,
                b_mark_shared_group=att_state.b_mark_shared_group,
                # 临时缓冲区大小和启动网格必须覆盖重放可能带来的每个长度。
                max_kv_seq_len=att_state.kv_seq_len_bound,
            )
        if att_config.kv_quant == "int8_group":
            from lightllm.common.attention.kernels.gqa.int8kv.int8kv_flash_decoding import (
                token_decode_attention_flash_decoding,
            )

            return token_decode_attention_flash_decoding(
                q=layer_inputs.q,
                req_to_token_indexs=att_state.req_to_token_indexs,
                b_req_idx=att_state.b_req_idx,
                b_seq_len=att_state.b_seq_len,
                cache_k=k,
                cache_k_scale=k_scale,
                cache_v=v,
                cache_v_scale=v_scale,
                out=out,
                alloc_tensor_func=alloc_func,
                max_len_in_batch=att_state.max_kv_seq_len,
            )
        from lightllm.common.attention.kernels.gqa.int4kv.ppl_int4kv_flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        return token_decode_attention_flash_decoding(
            q=layer_inputs.q,
            req_to_token_indexs=att_state.req_to_token_indexs,
            b_req_idx=att_state.b_req_idx,
            b_seq_len=att_state.b_seq_len,
            cache_k=k,
            cache_k_scale=k_scale,
            cache_v=v,
            cache_v_scale=v_scale,
            out=out,
            alloc_tensor_func=alloc_func,
            max_kv_seq_len=att_state.max_kv_seq_len,
        )

    @staticmethod
    def get_name() -> str:
        return "triton_quant_gqa_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> TritonQuantGQADecodeAttState:
        """图容量限制启动规模；动态分组每步生成，replay 前更新捕获状态。"""
        bound = (
            self.runtime_config.cuda_graph_max_kv_seq_len if infer_state.uses_cuda_graph else infer_state.max_kv_seq_len
        )
        if bound is None:
            raise ValueError(f"a {infer_state.graph_mode.value} step needs the CUDA graph kv length capacity")
        b_shared_seq_len = b_mark_shared_group = None
        if self.runtime_config.diverse_decode and self.att_config.kv_quant == "int8_group":
            from lightllm.common.basemodel.triton_kernel.diverse_utils import build_diverse_shared_group_markers

            assert infer_state.b_shared_seq_len is not None and infer_state.b_shared_radix_node_id is not None
            b_mark_shared_group = build_diverse_shared_group_markers(
                b_shared_radix_node_id=infer_state.b_shared_radix_node_id
            )
            # 单行组没有跨请求前缀共享来加速。
            b_shared_seq_len = infer_state.b_shared_seq_len.masked_fill(mask=b_mark_shared_group == 1, value=0)
        return TritonQuantGQADecodeAttState(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            kv_seq_len_bound=bound,
            max_kv_seq_len=infer_state.max_kv_seq_len,
            b_shared_seq_len=b_shared_seq_len,
            b_mark_shared_group=b_mark_shared_group,
        )


_KINDS = ("int8_group", "int4_group")
