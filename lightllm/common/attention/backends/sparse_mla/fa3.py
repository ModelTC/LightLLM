"""FA3 稀疏 MLA：顶部的 top-k 缓存槽作为每页一个 token 的页表。

解码覆盖 bf16 缓存，带或不带旋转通道；prefill覆盖 GLM-5 的打包 FP8 缓存，
它逐块反量化缓存。
"""
from collections.abc import Callable

from lightllm.common.basemodel.infer_struct import InferStateInfo

from dataclasses import dataclass

import torch

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from . import SparseMLADecodeAttInputs, SparseMLAAttConfig, SparseMLAPrefillAttInputs


@dataclass(frozen=True)
class FA3SparseMLADecodeAttState:
    """cache_seqlens 是每个请求读取的选定槽的数量；cu_seqlens_k 是其 int32 前缀和，cu_seqlens_q 是 int32 查询偏移量。causal 是内核的掩码模式，不会改变一行解码。"""

    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int


class FA3SparseMLADecodeBackend(
    DecodeAttBackend[SparseMLAAttConfig, FA3SparseMLADecodeAttState, SparseMLADecodeAttInputs]
):
    def __init__(
        self,
        att_config: SparseMLAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if att_config.kv_cache_format != "bf16":
            raise NotImplementedError("FA3 sparse MLA decode supports only the bf16 cache")

    def decode(
        self,
        att_state: FA3SparseMLADecodeAttState,
        layer_inputs: SparseMLADecodeAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """用 q_nope 的形状填充并返回 out。"""
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        att_config = self.att_config
        assert layer_inputs.q_nope.shape[1] == att_config.num_heads, "q does not match the att_config"
        kv, rank, rope = layer_inputs.kv_cache, att_config.kv_lora_rank, att_config.qk_rope_head_dim
        if rope == 0:
            # NoPE：only_qv 仅使用潜在查询和缓存进行注意力计算。
            flash_attn_with_kvcache(
                q=None,
                qv=layer_inputs.q_nope,
                k_cache=None,
                v_cache=kv.view(-1, 1, 1, rank),
                page_table=layer_inputs.topk_mem_indices,
                cache_seqlens=att_state.cache_seqlens,
                cu_seqlens_q=att_state.cu_seqlens_q,
                cu_seqlens_k_new=att_state.cu_seqlens_k,
                max_seqlen_q=att_state.max_seqlen_q,
                softmax_scale=att_config.softmax_scale,
                causal=att_config.causal,
                only_qv=True,
                out=out,
            )
            return out
        flash_attn_with_kvcache(
            q=layer_inputs.q_rope,
            k_cache=kv[:, :, rank:].view(-1, 1, 1, rope),
            v_cache=kv[:, :, :rank].view(-1, 1, 1, rank),
            qv=layer_inputs.q_nope,
            page_table=layer_inputs.topk_mem_indices,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k_new=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            softmax_scale=att_config.softmax_scale,
            causal=att_config.causal,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_sparse_mla_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> FA3SparseMLADecodeAttState:
        """每步生成索引器对应的长度，replay 前更新捕获状态。"""
        config = self.att_config
        if config.topk is None:
            raise ValueError("FA3 sparse MLA decode needs the layer config's topk")
        device, lengths, kpool = infer_state.b_seq_len.device, infer_state.b_seq_len, config.kpool
        if kpool == 1:
            cache_seqlens = torch.clamp(input=lengths, max=config.topk)
        else:
            cache_seqlens = torch.clamp(input=lengths // kpool * kpool, max=config.topk) + lengths % kpool
        padded = torch.zeros(size=(infer_state.batch_size + 1,), dtype=torch.int32, device=device)
        padded[1:].copy_(cache_seqlens, non_blocking=True)
        cu_seqlens_k = padded.cumsum(dim=0, dtype=torch.int32)
        return FA3SparseMLADecodeAttState(
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=infer_state.b1_cu_q_seq_len,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=infer_state.max_q_seq_len,
        )


@dataclass(frozen=True)
class FA3SparseMLAPrefillAttState:
    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    total_token_num: int
    cu_kv_seq_len: torch.Tensor
    cu_q_seq_len: torch.Tensor


class FA3SparseMLAPrefillBackend(
    PrefillAttBackend[SparseMLAAttConfig, FA3SparseMLAPrefillAttState, SparseMLAPrefillAttInputs]
):
    def __init__(
        self,
        att_config: SparseMLAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if att_config.kv_cache_format != "fp8_group64_e8m0" or att_config.qk_rope_head_dim != 0:
            raise NotImplementedError("FA3 sparse MLA prefill supports only the NoPE fp8_group64_e8m0 cache")

    @torch.no_grad()
    def prefill(
        self,
        att_state: FA3SparseMLAPrefillAttState,
        layer_inputs: SparseMLAPrefillAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回形状为 [query_tokens, num_heads, kv_lora_rank] 的 out。

        本步的 key 分块收集到工作区，其中包括 current_kv 和反量化后的历史缓存。
        各分块的 attention 部分结果以 FP32 合并到输出。
        """
        import triton
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        from lightllm.common.attention.kernels.sparse_mla.fp8_group64_prefill import (
            _gather_prefill_kv_kernel,
            _compact_topk_indices_kernel,
            _merge_att_output_kernel,
        )

        q, packed_kv = layer_inputs.q, layer_inputs.kv_cache
        prefill_cache_kv, topk_indices = layer_inputs.current_kv, layer_inputs.topk_indices
        assert prefill_cache_kv is not None and topk_indices is not None and layer_inputs.q_kv_start_loc is not None
        q_token_num, q_head_num, _ = q.shape
        buffers = layer_inputs.fp8_workspace
        if buffers is None:
            raise ValueError("FA3 sparse MLA prefill requires the reserved FP8 workspace")
        topk_size = topk_indices.shape[1]
        assert q_token_num <= buffers["chunk_cache_seqlens"].numel()
        assert topk_size <= buffers["chunk_topk_indices"].shape[1]
        gathered_kv = buffers["gathered_kv"]
        chunk_topk_indices = buffers["chunk_topk_indices"].view(-1)[: q_token_num * topk_size]
        chunk_topk_indices = chunk_topk_indices.view(q_token_num, topk_size)
        chunk_cache_seqlens = buffers["chunk_cache_seqlens"][:q_token_num]
        cu_seqlens_q = buffers["cu_seqlens_q"][: q_token_num + 1]
        req_to_token_indexs, b_req_idx = att_state.req_to_token_indexs, att_state.b_req_idx
        total_token_num = att_state.total_token_num
        gather_chunk_size = min(gathered_kv.shape[0], total_token_num)
        has_history = total_token_num > q_token_num
        if not has_history and prefill_cache_kv.stride(-1) != 1:
            prefill_cache_kv = prefill_cache_kv.contiguous()
        # 一个块直接写入输出；多个块先在 FP32 中合并其输出。
        single_chunk = gather_chunk_size >= total_token_num
        output = softmax_lse = None
        for chunk_start in range(0, total_token_num, gather_chunk_size):
            chunk_token_num = min(gather_chunk_size, total_token_num - chunk_start)
            if has_history:
                _gather_prefill_kv_kernel[(chunk_token_num // 2 + b_req_idx.numel(),)](
                    packed_kv,
                    prefill_cache_kv,
                    gathered_kv,
                    req_to_token_indexs,
                    b_req_idx,
                    att_state.cu_kv_seq_len,
                    att_state.cu_q_seq_len,
                    chunk_start,
                    chunk_token_num,
                    packed_kv.stride(0),
                    prefill_cache_kv.stride(0),
                    prefill_cache_kv.stride(2),
                    req_to_token_indexs_stride_b=req_to_token_indexs.stride(0),
                    req_to_token_indexs_stride_s=req_to_token_indexs.stride(1),
                    batch_size=b_req_idx.numel(),
                    BLOCK_ROWS=2,
                    num_warps=2,
                )
                chunk_kv = gathered_kv[:chunk_token_num].view(-1, 1, 1, 512)
            else:
                # 步骤的键已经形成了完整的请求布局。
                chunk_kv = prefill_cache_kv[chunk_start : chunk_start + chunk_token_num].view(-1, 1, 1, 512)
            _compact_topk_indices_kernel[(q_token_num,)](
                topk_indices,
                layer_inputs.q_kv_start_loc,
                chunk_topk_indices,
                chunk_cache_seqlens,
                chunk_start,
                chunk_start + chunk_token_num,
                topk_indices.stride(0),
                topk_indices.stride(1),
                TOPK=topk_size,
                BLOCK=triton.next_power_of_2(topk_size),
            )
            chunk_output, chunk_softmax_lse, *_ = flash_attn_with_kvcache(
                # only_qv 忽略 Q/K；这些视图避免分配占位张量。
                q=q[..., :64],
                k_cache=chunk_kv[..., :64],
                qv=q,
                v_cache=chunk_kv,
                page_table=chunk_topk_indices,
                cache_seqlens=chunk_cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                softmax_scale=self.att_config.softmax_scale,
                causal=False,
                only_qv=True,
                num_splits=1,
                return_softmax_lse=True,
                out=out if single_chunk else None,
            )
            if chunk_start == 0:
                output, softmax_lse = chunk_output, chunk_softmax_lse
            else:
                # 保持所有剩余块中的在线减少在 FP32 中进行。
                output = output.float()
                _merge_att_output_kernel[(q_token_num, triton.cdiv(x=q_head_num, y=4))](
                    output,
                    softmax_lse,
                    chunk_output,
                    chunk_softmax_lse,
                    chunk_softmax_lse.stride(1),
                    chunk_softmax_lse.stride(0),
                    q_head_num=q_head_num,
                    q_token_num=q_token_num,
                    BLOCK_HEADS=4,
                )
            del chunk_output, chunk_softmax_lse
        if not single_chunk:
            out.copy_(output)
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_sparse_mla_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> FA3SparseMLAPrefillAttState:
        return FA3SparseMLAPrefillAttState(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            total_token_num=infer_state.total_token_num,
            cu_kv_seq_len=infer_state.b1_cu_kv_seq_len,
            cu_q_seq_len=infer_state.b1_cu_q_seq_len,
        )
