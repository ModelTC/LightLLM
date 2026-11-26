from collections.abc import Callable
from lightllm.common.basemodel.infer_struct import InferStateInfo
from dataclasses import dataclass

import torch

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from . import MLADecodeAttInputs, MLADecodeAttConfig, MLAPrefillAttInputs, MLAPrefillAttConfig


@dataclass(frozen=True)
class TritonMLAPrefillAttState:
    b_q_start_loc: torch.Tensor
    cu_kv_seq_len: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int


class TritonMLAPrefillBackend(PrefillAttBackend[MLAPrefillAttConfig, TritonMLAPrefillAttState, MLAPrefillAttInputs]):
    def __init__(
        self,
        att_config: MLAPrefillAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        # 256 维查询（GLM-4 MoE Lite：192 nope + 64 rope）运行连续 KV GQA prefill 内核
        # 在每个注意力头扩展的键上。
        self.expands_keys = att_config.qk_nope_head_dim + att_config.qk_rope_head_dim == 256
        if not att_config.causal:
            raise NotImplementedError("Triton MLA prefill supports only causal attention")

    def prefill(
        self,
        att_state: TritonMLAPrefillAttState,
        layer_inputs: MLAPrefillAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """将因果注意力写入 out [query_tokens, num_heads, v_head_dim] 并返回它。

        操作数共享数据类型/设备，并具有单位最后一个步长。out 必须与 layer_inputs 不重叠。
        """
        from lightllm.common.attention.kernels.mla.prefill.context_flashattention_nopad_with_v import (
            context_attention_fwd_with_v,
        )

        assert att_state.b_q_start_loc is not None and att_state.cu_kv_seq_len is not None, "prefill fields required"
        assert att_state.b_ready_cache_len is not None, "prefill fields required"
        att_config = self.att_config
        nope_dim = att_config.qk_nope_head_dim
        assert layer_inputs.q.shape[1:] == (
            att_config.num_heads,
            nope_dim + att_config.qk_rope_head_dim,
        ), "q does not match the att_config"
        assert layer_inputs.q.dtype == att_config.dtype, "q does not match the att_config"
        if self.expands_keys:
            return self._expanded_key_forward(att_state, layer_inputs, out, alloc_func)
        context_attention_fwd_with_v(
            q_nope=layer_inputs.q[:, :, :nope_dim],
            q_rope=layer_inputs.q[:, :, nope_dim:],
            k_nope=layer_inputs.k_nope,
            k_rope=layer_inputs.k_rope,
            v=layer_inputs.v,
            o=out,
            b_start_loc=att_state.b_q_start_loc,
            b_kv_start_loc=att_state.cu_kv_seq_len,
            b_seq_len=att_state.b_seq_len,
            b_prompt_cache_len=att_state.b_ready_cache_len,
            max_input_len=att_state.max_q_seq_len,
            softmax_scale=att_config.softmax_scale,
        )
        return out

    def create_att_state(self, infer_state: InferStateInfo) -> TritonMLAPrefillAttState:
        return TritonMLAPrefillAttState(
            b_q_start_loc=infer_state.b_q_start_loc,
            cu_kv_seq_len=infer_state.b1_cu_kv_seq_len,
            b_seq_len=infer_state.b_seq_len,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            max_q_seq_len=infer_state.max_q_seq_len,
        )

    @staticmethod
    def get_name() -> str:
        return "triton_mla_prefill"

    def _expanded_key_forward(
        self,
        att_state: TritonMLAPrefillAttState,
        layer_inputs: MLAPrefillAttInputs,
        out: torch.Tensor | None,
        alloc_func: Callable[..., torch.Tensor],
    ):
        from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
            context_attention_fwd_contiguous_kv,
        )

        att_config, q = self.att_config, layer_inputs.q
        nope_dim, heads = att_config.qk_nope_head_dim, att_config.num_heads
        k = alloc_func((layer_inputs.k_nope.shape[0], heads, q.shape[-1]), dtype=q.dtype, device=q.device)
        k[:, :, :nope_dim].copy_(layer_inputs.k_nope.expand(-1, heads, -1))
        k[:, :, nope_dim:].copy_(layer_inputs.k_rope.expand(-1, heads, -1))
        context_attention_fwd_contiguous_kv(
            q=q,
            k=k,
            v=layer_inputs.v,
            o=out,
            b_start_loc=att_state.b_q_start_loc,
            b_kv_start_loc=att_state.cu_kv_seq_len,
            b_seq_len=att_state.b_seq_len,
            max_q_input_len=att_state.max_q_seq_len,
            b_prompt_cache_len=att_state.b_ready_cache_len,
            softmax_scale=att_config.softmax_scale,
        )
        return out


@dataclass(frozen=True)
class TritonMLADecodeAttState:
    """在 decode 时，请求 i 读取物理槽 req_to_token_indexs[b_req_idx[i], :b_seq_len[i]]。

    分割调度是为记录的注意力头数量、潜在/旋转维度和数据类型构建的。
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    decode_att_block_seq: torch.Tensor
    mid_o_batch_start_index: torch.Tensor
    vsm_count: int
    run_config: dict[str, int]


class TritonMLADecodeBackend(DecodeAttBackend[MLADecodeAttConfig, TritonMLADecodeAttState, MLADecodeAttInputs]):
    def __init__(
        self,
        att_config: MLADecodeAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        # 每个查询行读取自己的请求前缀，该前缀在请求的行之间具有因果关系。
        if not att_config.causal:
            raise NotImplementedError("Triton MLA decode supports only causal attention")

    def decode(
        self,
        att_state: TritonMLADecodeAttState,
        layer_inputs: MLADecodeAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状为 q_nope 的 [requests, num_heads, kv_lora_rank]，数据类型和设备与 q_nope 相同。

        输入和缓存为只读。out 必须具有单位最后一个步长，并且不能与它们重叠。
        """
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        att_config = self.att_config
        q_nope, q_rope = layer_inputs.q_nope, layer_inputs.q_rope
        assert q_nope.shape[1:] == (att_config.num_heads, att_config.kv_lora_rank) and q_nope.dtype == att_config.dtype
        assert q_rope.shape[-1] == att_config.qk_rope_head_dim
        assert layer_inputs.kv_cache.shape[-1] == att_config.kv_lora_rank + att_config.qk_rope_head_dim
        gqa_token_decode_attention_flash_decoding(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_nope=layer_inputs.kv_cache[:, :, : att_config.kv_lora_rank],
            kv_rope=layer_inputs.kv_cache[:, :, att_config.kv_lora_rank :],
            req_to_token_indexs=att_state.req_to_token_indexs,
            b_req_idx=att_state.b_req_idx,
            b_seq_len=att_state.b_seq_len,
            softmax_scale=att_config.softmax_scale,
            decode_att_block_seq=att_state.decode_att_block_seq,
            mid_o_batch_start_index=att_state.mid_o_batch_start_index,
            vsm_count=att_state.vsm_count,
            run_config=att_state.run_config,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "triton_mla_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> TritonMLADecodeAttState:
        """每步计算分块张量，replay 前复制到捕获状态；图容量固定启动配置。"""
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import plan_mla_decode_schedule

        if infer_state.uses_cuda_graph:
            tuning_seq_len = self.runtime_config.cuda_graph_max_kv_seq_len
            if tuning_seq_len is None:
                raise ValueError(f"a {infer_state.graph_mode.value} step needs the CUDA graph kv length capacity")
        else:
            tuning_seq_len = infer_state.total_token_num // infer_state.batch_size
        block_seq, starts, vsm_count, run_config = plan_mla_decode_schedule(
            b_seq_len=infer_state.b_seq_len,
            q_head_num=self.att_config.num_heads,
            kv_lora_rank=self.att_config.kv_lora_rank,
            q_rope_dim=self.att_config.qk_rope_head_dim,
            dtype=self.att_config.dtype,
            tuning_seq_len=tuning_seq_len,
        )
        return TritonMLADecodeAttState(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            decode_att_block_seq=block_seq,
            mid_o_batch_start_index=starts,
            vsm_count=vsm_count,
            run_config=run_config,
        )
