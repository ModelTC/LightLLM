import dataclasses

import torch
import triton
import triton.language as tl

from .fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend, NsaFlashMlaFp8SparseDecodeAttState
from .glm5_next import Glm5NextSparsePrefillState


@triton.jit
def _glm53_direct_decode_kernel(
    q,
    packed_kv,
    topk,
    partial_o,
    partial_lse,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    packed_stride: tl.constexpr,
    topk_stride_b: tl.constexpr,
    topk_stride_d: tl.constexpr,
    packed_rows: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLITS: tl.constexpr,
):
    qid = tl.program_id(0)
    split = tl.program_id(1)
    h = tl.arange(0, 16)
    d = tl.arange(0, 512)
    h_mask = h < 8
    n = tl.arange(0, BLOCK_K)
    slot = split * BLOCK_K + n
    indices = tl.load(topk + qid * topk_stride_b + slot * topk_stride_d, slot < 2176, -1).to(tl.int64)
    valid = (indices >= 0) & (indices < packed_rows)
    packed = packed_kv + indices[:, None] * packed_stride
    q_values = tl.load(
        q + qid * q_stride_b + h[:, None] * q_stride_h + d[None, :] * q_stride_d,
        h_mask[:, None],
        0.0,
    ).to(tl.bfloat16)
    group = tl.arange(0, 8)
    raw_scales = tl.load(
        packed + 576 + group[None, :],
        valid[:, None] & (group[None, :] < 7),
        127,
    )
    scales = (raw_scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    scales = tl.broadcast_to(scales[:, :, None], (BLOCK_K, 8, 64)).reshape(BLOCK_K, 512)
    fp8 = (
        tl.load(
            packed.to(tl.pointer_type(tl.float8e4nv)) + d[None, :],
            valid[:, None] & (d[None, :] < 448),
            0.0,
        ).to(tl.float32)
        * scales
    )
    tail = tl.load(
        (packed + 448).to(tl.pointer_type(tl.bfloat16)) + d[None, :] - 448,
        valid[:, None] & (d[None, :] >= 448),
        0.0,
    ).to(tl.float32)
    kv = tl.where(d[None, :] < 448, fp8, tail).to(tl.bfloat16)
    score = tl.dot(q_values, tl.trans(kv), out_dtype=tl.float32) * softmax_scale
    score = tl.where(valid[None, :], score, -float("inf"))
    maximum = tl.max(score, axis=1)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    probability = tl.where(valid[None, :], tl.exp(score - safe_maximum[:, None]), 0.0)
    denominator = tl.sum(probability, axis=1)
    has_values = denominator > 0.0
    values = tl.dot(probability.to(tl.bfloat16), kv, out_dtype=tl.float32)
    base = (qid * SPLITS + split) * 8 + h[:, None]
    tl.store(
        partial_o + base * 512 + d[None, :],
        tl.where(has_values[:, None], values / denominator[:, None], 0.0),
        h_mask[:, None],
    )
    tl.store(
        partial_lse + (qid * SPLITS + split) * 8 + h,
        tl.where(has_values, maximum + tl.log(denominator), -float("inf")),
        h_mask,
    )


def _can_use_glm53_direct_decode(q_nope, packed_kv, topk, softmax_scale, kv_lora_rank, bound):
    return (
        bound == 2051
        and kv_lora_rank == 512
        and q_nope.ndim == 3
        and q_nope.shape[0] in (1, 2, 4, 8)
        and q_nope.shape[1:] == (8, 512)
        and q_nope.dtype == torch.bfloat16
        and q_nope.is_cuda
        and q_nope.stride(-1) == 1
        and packed_kv.ndim == 3
        and packed_kv.shape[1:] == (1, 584)
        and packed_kv.dtype == torch.uint8
        and packed_kv.is_cuda
        and packed_kv.device == q_nope.device
        and packed_kv.stride(-1) == 1
        and topk.ndim == 3
        and topk.shape == (q_nope.shape[0], 1, 2176)
        and topk.dtype == torch.int32
        and topk.is_cuda
        and topk.device == q_nope.device
        and topk.stride(-1) == 1
        and isinstance(softmax_scale, (float, int))
        and float(softmax_scale) > 0.0
    )


def _glm53_direct_decode(q_nope, packed_kv, topk, softmax_scale):
    from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
        _reduce_direct_smallq_kernel,
    )

    batch = q_nope.shape[0]
    block_k = 16 if batch == 1 else 64
    splits = 2112 // block_k
    partial_o = torch.empty((batch, splits, 8, 512), device=q_nope.device, dtype=torch.float32)
    partial_lse = torch.empty((batch, splits, 8), device=q_nope.device, dtype=torch.float32)
    out = torch.empty(q_nope.shape, device=q_nope.device, dtype=q_nope.dtype)
    _glm53_direct_decode_kernel[(batch, splits)](
        q_nope,
        packed_kv,
        topk,
        partial_o,
        partial_lse,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        packed_kv.stride(0),
        topk.stride(0),
        topk.stride(2),
        packed_kv.shape[0],
        float(softmax_scale),
        block_k,
        splits,
        num_warps=4 if batch == 1 else 8,
        num_stages=1,
    )
    _reduce_direct_smallq_kernel[(batch, 8, 4)](
        partial_o,
        partial_lse,
        out,
        8,
        splits,
        256 if batch == 1 else 64,
        num_warps=4,
    )
    return out


@triton.jit
def _fused_strided_qpad_kernel(
    q,
    padded_q,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    num_heads: tl.constexpr,
    padded_num_heads: tl.constexpr,
    BLOCK1024: tl.constexpr,
):
    batch = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK1024 + tl.arange(0, BLOCK1024)
    in_range = offset < padded_num_heads * 512
    head = offset // 512
    dim = offset - head * 512
    from_q = in_range & (head < num_heads)
    value = tl.load(q + batch * q_stride_b + head * q_stride_h + dim * q_stride_d, mask=from_q, other=0.0)
    tl.store(padded_q + batch * padded_num_heads * 512 + offset, value, mask=in_range)


class Fp8Glm5NextSparseAttBackend(NsaFlashMlaFp8SparseAttBackend):
    def create_att_prefill_state(self, infer_state):
        return Fp8Glm5NextSparsePrefillState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        config = getattr(self.model, "config", {})
        # 2048 pooled entries plus at most three kpool remainder entries fit the 2112 rounded direct window.
        bound = (
            2051
            if isinstance(config, dict) and config.get("index_topk") == 2048 and config.get("index_kpool") == 4
            else None
        )
        return Fp8Glm5NextSparseDecodeState(backend=self, infer_state=infer_state, decode_topk_bound=bound)


@dataclasses.dataclass
class Fp8Glm5NextSparsePrefillState(Glm5NextSparsePrefillState):
    def _nsa_prefill_att(self, q, kv, att_control):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            context_attention_fwd_fp8,
        )

        params = att_control.nsa_prefill_dict
        return context_attention_fwd_fp8(
            q,
            kv,
            self.ragged_mem_index,
            self.infer_state.mem_index,
            params["prefill_cache_kv"],
            self.ks,
            params["topk_indices"],
            params["softmax_scale"],
        )


@dataclasses.dataclass
class Fp8Glm5NextSparseDecodeState(NsaFlashMlaFp8SparseDecodeAttState):
    decode_topk_bound: int = None

    def _nsa_decode_att(self, q, packed_kv, att_control):
        import flash_mla

        params = att_control.nsa_decode_dict
        topk_mem_indices = params["topk_mem_indices"]
        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        assert topk_mem_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"

        q_nope, _ = q
        num_tokens, num_heads, _ = q_nope.shape
        if (
            num_tokens in (1, 8) or (num_tokens in (2, 4) and getattr(self.infer_state, "is_cuda_graph", False))
        ) and _can_use_glm53_direct_decode(
            q_nope,
            packed_kv,
            topk_mem_indices,
            params["softmax_scale"],
            params["kv_lora_rank"],
            self.decode_topk_bound,
        ):
            return _glm53_direct_decode(q_nope, packed_kv, topk_mem_indices, params["softmax_scale"])
        # FlashMLA uses 64-head tiles; NoPE keeps its original 512 dimensions.
        padded_num_heads = ((num_heads + 63) // 64) * 64
        if padded_num_heads != num_heads:
            if getattr(self.infer_state, "is_cuda_graph", False):
                padded_q = q_nope.new_empty((num_tokens, 1, padded_num_heads, 512))
                _fused_strided_qpad_kernel[(num_tokens, triton.cdiv(padded_num_heads * 512, 1024))](
                    q_nope,
                    padded_q,
                    q_nope.stride(0),
                    q_nope.stride(1),
                    q_nope.stride(2),
                    num_heads,
                    padded_num_heads,
                    BLOCK1024=1024,
                    num_warps=4,
                )
            else:
                padded_q = q_nope.new_zeros((num_tokens, 1, padded_num_heads, 512))
                padded_q[:, 0, :num_heads] = q_nope
        else:
            padded_q = q_nope.unsqueeze(1).contiguous()
        kv_cache = torch.as_strided(
            packed_kv,
            size=(packed_kv.shape[0], 1, 1, packed_kv.shape[-1]),
            stride=(packed_kv.stride(0), packed_kv.shape[-1], packed_kv.shape[-1], packed_kv.stride(-1)),
        )
        output, _ = flash_mla.flash_mla_with_kvcache(
            q=padded_q,
            k_cache=kv_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=params["kv_lora_rank"],
            tile_scheduler_metadata=self.flashmla_sched_meta,
            softmax_scale=params["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return output[:, 0, :num_heads, :]
