from collections.abc import Callable
from lightllm.common.basemodel.infer_struct import InferStateInfo
from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from . import GQAAttConfig, GQAAttInputs


@dataclass(frozen=True)
class TritonGQADecodeAttState:
    """每个请求一个查询的注意力状态，按 b_req_idx 顺序排列。

    索引张量遵循 InferStateInfo 的布局和设备。
    b_seq_len 给出实际正长度，包括当前 token；所有地址的token 映射条目必须标识填充的缓存槽。

    这两个长度描述了步骤，而不是内核。kv_seq_len_bound 约束了注意力状态可能运行的每个长度：对于一个即时执行步骤的实际最大值，对于一个 CUDA Graph步骤的图容量，其重放步骤重用它。
    max_kv_seq_len 是该步骤的实际最大值。它仅作为提示有效，因为重放可能看到更长的长度：一个 33 的提示和一个 512 的约束在长度后来达到 300 时仍然正确。total_token_bound
    同样约束了长度之和，b_kv_start_loc[i] 是行 i 之前所有行的长度之和。

    使用投机解码时，一个请求的连续行形成一个共享其 KV 的组；b_mark_shared_group 在组内为 0，而在组的最后一行上保存组大小（填充行形成一个单行组）。当每个请求只有一个行时，它为 None。
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    kv_seq_len_bound: int
    max_kv_seq_len: int
    total_token_bound: int = 0
    b_kv_start_loc: Optional[torch.Tensor] = None
    b_mark_shared_group: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TritonGQAPrefillAttState:
    b_req_idx: torch.Tensor
    b_q_start_loc: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int
    req_to_token_indexs: torch.Tensor
    b_image_token_end: Optional[torch.Tensor]
    b_image_token_start: Optional[torch.Tensor]


class TritonGQAPrefillBackend(PrefillAttBackend[GQAAttConfig, TritonGQAPrefillAttState, GQAAttInputs]):
    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        """支持 16、32、64、128、256 和 512 的注意力头维度，softmax_scale 为 1/sqrt(head_dim)。"""
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        TritonGQAPrefillBackend._check_supported(att_config=att_config)

    def prefill(
        self,
        att_state: TritonGQAPrefillAttState,
        layer_inputs: GQAAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状、数据类型和设备与 q 一致。

        q、K/V 和 out 是在同一个 CUDA 设备上的匹配 FP16/BF16 张量，具有
        单位注意力头维度步长和不重叠的标记/注意力头步长，可能包括填充。K/V 可能是共享存储的不同切片；任意步长布局不受支持。调用者必须将 q.device 设置为当前 CUDA 设备。out 不能与
        layer_inputs 或 att_state 张量重叠。
        prefill 不需要临时缓冲区，也不调用 alloc_func。
        """
        from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
            context_attention_fwd,
        )

        assert (
            att_state.b_q_start_loc is not None and att_state.b_ready_cache_len is not None
        ), "prefill fields required"
        TritonGQAPrefillBackend._check_inputs(att_config=self.att_config, layer_inputs=layer_inputs)
        if self.att_config.bidirectional_image_spans:
            assert att_state.b_image_token_end is not None, "bidirectional image spans need b_image_token_end"
            if self.att_config.sliding_window == (-1, -1):
                from lightllm.common.attention.kernels.gqa.bf16.prefill.context_attention_fwd_neo import (
                    context_attention_fwd_neo,
                )

                context_attention_fwd_neo(
                    q=layer_inputs.q,
                    k=layer_inputs.k_cache,
                    v=layer_inputs.v_cache,
                    o=out,
                    b_req_idx=att_state.b_req_idx,
                    b_start_loc=att_state.b_q_start_loc,
                    b_seq_len=att_state.b_seq_len,
                    b_prompt_cache_len=att_state.b_ready_cache_len,
                    max_input_len=att_state.max_q_seq_len,
                    req_to_token_indexs=att_state.req_to_token_indexs,
                    b_image_token_end=att_state.b_image_token_end,
                )
            else:
                from lightllm.common.attention.kernels.gqa.bf16.prefill.context_attention_fwd_gemma4_mm import (
                    context_attention_fwd_gemma4_mm,
                )

                # 一个跨度可能回溯到窗口之外，因此内核还需要每个跨度的起始位置。
                assert att_state.b_image_token_start is not None, "windowed image spans need b_image_token_start"
                context_attention_fwd_gemma4_mm(
                    q=layer_inputs.q,
                    k=layer_inputs.k_cache,
                    v=layer_inputs.v_cache,
                    o=out,
                    b_req_idx=att_state.b_req_idx,
                    b_start_loc=att_state.b_q_start_loc,
                    b_seq_len=att_state.b_seq_len,
                    b_prompt_cache_len=att_state.b_ready_cache_len,
                    max_input_len=att_state.max_q_seq_len,
                    req_to_token_indexs=att_state.req_to_token_indexs,
                    b_image_token_end=att_state.b_image_token_end,
                    b_image_token_start=att_state.b_image_token_start,
                    sliding_window=self.att_config.sliding_window,
                )
            return out
        if self.att_config.has_alibi:
            from lightllm.common.basemodel.triton_kernel.alibi_att.context_flashattention_nopad import (
                context_attention_fwd as alibi_context_attention_fwd,
            )

            alibi_context_attention_fwd(
                layer_inputs.q,
                layer_inputs.k_cache,
                layer_inputs.v_cache,
                out,
                att_state.b_req_idx,
                layer_inputs.alibi_slopes,
                att_state.b_q_start_loc,
                att_state.b_seq_len,
                att_state.b_ready_cache_len,
                att_state.max_q_seq_len,
                att_state.req_to_token_indexs,
            )
            return out
        context_attention_fwd(
            q=layer_inputs.q,
            k=layer_inputs.k_cache,
            v=layer_inputs.v_cache,
            o=out,
            b_req_idx=att_state.b_req_idx,
            b_start_loc=att_state.b_q_start_loc,
            b_seq_len=att_state.b_seq_len,
            b_prompt_cache_len=att_state.b_ready_cache_len,
            max_input_len=att_state.max_q_seq_len,
            req_to_token_indexs=att_state.req_to_token_indexs,
            sliding_window=self.att_config.sliding_window,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "triton_gqa_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> TritonGQAPrefillAttState:
        return TritonGQAPrefillAttState(
            b_req_idx=infer_state.b_req_idx,
            b_q_start_loc=infer_state.b_q_start_loc,
            b_seq_len=infer_state.b_seq_len,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            max_q_seq_len=infer_state.max_q_seq_len,
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_image_token_end=infer_state.b_image_token_end,
            b_image_token_start=infer_state.b_image_token_start,
        )

    @staticmethod
    def _check_supported(att_config: GQAAttConfig) -> None:
        if att_config.head_dim not in (16, 32, 64, 128, 256, 512):
            raise NotImplementedError(f"Triton GQA does not support head_dim={att_config.head_dim}")
        # Triton GQA 内核自行计算 1/sqrt(head_dim)，并且仅查看查询左侧的内容。
        if not att_config.has_default_scale:
            raise NotImplementedError(
                f"Triton GQA supports only softmax_scale=1/sqrt(head_dim), got {att_config.softmax_scale}"
            )
        if att_config.sliding_window[1] > 0:
            raise NotImplementedError(
                f"Triton GQA supports only causal sliding windows (left, 0), got {att_config.sliding_window}"
            )
        if not att_config.causal or att_config.has_sinks:
            raise NotImplementedError("Triton GQA supports only causal attention without sinks")
        if att_config.kv_quant is not None:
            raise NotImplementedError(f"Triton GQA does not support kv_quant={att_config.kv_quant!r}")
        if att_config.bidirectional_image_spans and att_config.has_alibi:
            raise NotImplementedError("Triton GQA does not combine ALiBi with bidirectional image spans")
        if att_config.has_alibi and (
            att_config.num_heads != att_config.num_kv_heads or att_config.sliding_window != (-1, -1)
        ):
            raise NotImplementedError("Triton GQA supports ALiBi only for MHA without sliding window")
        # ALiBi 内核和无窗口图像跨度内核仅支持 head_dim 256 及以下。
        if att_config.head_dim > 256 and (
            att_config.has_alibi or (att_config.bidirectional_image_spans and att_config.sliding_window == (-1, -1))
        ):
            raise NotImplementedError(
                f"Triton GQA supports head_dim <= 256 for ALiBi and unwindowed image spans, got {att_config.head_dim}"
            )

    @staticmethod
    def _check_inputs(att_config: GQAAttConfig, layer_inputs: GQAAttInputs) -> None:
        q, k = layer_inputs.q, layer_inputs.k_cache
        assert (
            q.shape[1:] == (att_config.num_heads, att_config.head_dim) and q.dtype == att_config.dtype
        ), "q does not match the att_config"
        assert k.shape[1:] == (
            att_config.num_kv_heads,
            att_config.head_dim,
        ), "KV cache does not match the att_config"
        # 量化缓存需要在层配置中指定 kv_quant，并使用对应的量化实现。
        assert k.dtype == q.dtype and layer_inputs.v_cache.dtype == q.dtype, "Triton GQA needs a KV cache of q's dtype"
        assert (layer_inputs.alibi_slopes is not None) == att_config.has_alibi, "alibi_slopes must match has_alibi"


class TritonGQADecodeBackend(DecodeAttBackend[GQAAttConfig, TritonGQADecodeAttState, GQAAttInputs]):
    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        """支持的注意力头维度为 16、32、64、128 和 256，GQA 支持 512，softmax_scale 为 1/sqrt(head_dim)。

        GQA（num_heads > num_kv_heads）支持滑动窗口 (-1, -1) 和 (left, 0)；
        MHA（num_heads == num_kv_heads）仅支持 (-1, -1)。
        """
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        TritonGQAPrefillBackend._check_supported(att_config=att_config)
        if att_config.bidirectional_image_spans:
            raise NotImplementedError("bidirectional image spans apply to prefill only")
        self.is_mha = att_config.num_heads == att_config.num_kv_heads
        if self.is_mha and att_config.head_dim == 512:
            raise NotImplementedError("Triton MHA decode does not support head_dim=512")
        if self.is_mha and att_config.sliding_window != (-1, -1):
            raise NotImplementedError("sliding_window is not supported by the MHA decode kernel")

    def decode(
        self,
        att_state: TritonGQADecodeAttState,
        layer_inputs: GQAAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状、数据类型和设备与 q 一致。

        q、K/V 和 out 是在同一个 CUDA 设备上的匹配 FP16/BF16 张量，具有
        单位注意力头维度步长和不重叠的标记/注意力头步长，可能包括填充。K/V 可能是共享存储的不同切片；任意步长布局不受支持。调用者必须将 q.device 设置为当前 CUDA 设备。out 不能与
        layer_inputs 或 att_state 张量重叠。
        K 和 V 必须具有相同的步长。
        临时缓冲区使用 alloc_func 在 q.device 上分配。必须使用一个仅限于一个设备的缓存分配器，并且仅在该设备上使用。
        """
        TritonGQAPrefillBackend._check_inputs(att_config=self.att_config, layer_inputs=layer_inputs)
        if self.att_config.has_alibi:
            from lightllm.common.basemodel.triton_kernel.alibi_att.token_flashattention_nopad import (
                token_attention_fwd,
            )

            token_attention_fwd(
                layer_inputs.q,
                layer_inputs.k_cache,
                layer_inputs.v_cache,
                out,
                layer_inputs.alibi_slopes,
                att_state.req_to_token_indexs,
                att_state.b_req_idx,
                att_state.b_kv_start_loc,
                att_state.b_seq_len,
                # 临时缓冲区和启动大小必须覆盖重放可能带来的每个长度。
                att_state.kv_seq_len_bound,
                att_state.total_token_bound,
                alloc_tensor_func=alloc_func,
            )
            return out
        if (
            att_state.b_mark_shared_group is not None
            and self.att_config.sliding_window == (-1, -1)
            and self.att_config.head_dim <= 256
        ):
            from lightllm.common.attention.kernels.gqa.bf16.decode.mtp_diverse import (
                token_decode_attention_mtp_diverse_single_token,
            )

            # 一个请求的各行共享其KV，因此一次KV遍历可以服务整个组。
            return token_decode_attention_mtp_diverse_single_token(
                q=layer_inputs.q,
                k=layer_inputs.k_cache,
                v=layer_inputs.v_cache,
                Req_to_tokens=att_state.req_to_token_indexs,
                B_req_idx=att_state.b_req_idx,
                b_seq_len=att_state.b_seq_len,
                b_mark_shared_group=att_state.b_mark_shared_group,
                # 仅选择内核配置；内核在GPU上读取实际长度。
                max_kv_len=att_state.max_kv_seq_len,
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
                req_to_token_indexs=att_state.req_to_token_indexs,
                b_req_idx=att_state.b_req_idx,
                b_seq_len=att_state.b_seq_len,
                # 临时缓冲区大小和启动网格必须覆盖重放可能带来的每种长度。
                max_kv_seq_len=att_state.kv_seq_len_bound,
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
            req_to_token_indexs=att_state.req_to_token_indexs,
            b_req_idx=att_state.b_req_idx,
            b_seq_len=att_state.b_seq_len,
            # 仅选择一个内核配置；GQA内核在GPU上读取实际长度。
            max_len_in_batch=att_state.max_kv_seq_len,
            out=out,
            alloc_tensor_func=alloc_func,
            sliding_window=self.att_config.sliding_window,
        )

    @staticmethod
    def get_name() -> str:
        return "triton_gqa_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> TritonGQADecodeAttState:
        """内核直接读取请求张量；投机分组每步生成，replay 前更新捕获状态。"""
        if infer_state.uses_cuda_graph:
            bound = self.runtime_config.cuda_graph_max_kv_seq_len
            if bound is None:
                raise ValueError(f"a {infer_state.graph_mode.value} step needs the CUDA graph kv length capacity")
            total_bound = bound * infer_state.batch_size
        else:
            bound, total_bound = infer_state.max_kv_seq_len, infer_state.total_token_num
        markers = None
        if self.runtime_config.decode_rows_per_request > 1:
            from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

            markers = build_mtp_shared_group_markers(
                b_req_idx=infer_state.b_req_idx, hold_req_id=infer_state.pad_req_idx
            )
        return TritonGQADecodeAttState(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            kv_seq_len_bound=bound,
            max_kv_seq_len=infer_state.max_kv_seq_len,
            total_token_bound=total_bound,
            b_kv_start_loc=None if infer_state.b1_cu_kv_seq_len is None else infer_state.b1_cu_kv_seq_len[:-1],
            b_mark_shared_group=markers,
        )
