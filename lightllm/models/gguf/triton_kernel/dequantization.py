"""
ref: https://github.com/ggml-org/llama.cpp/discussions/17393
"""
import torch
import triton
import triton.language as tl
from gguf import GGMLQuantizationType
from typing import Callable, Optional

_GGUF_DEQUANT_REGISTRY: dict[GGMLQuantizationType, Callable[..., torch.Tensor]] = {}


def register_gguf_dequant(quant_type: GGMLQuantizationType):

    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if quant_type in _GGUF_DEQUANT_REGISTRY:
            raise ValueError(
                f"Duplicate GGUF dequant registration for {quant_type}: "
                f"{_GGUF_DEQUANT_REGISTRY[quant_type]!r} vs {fn!r}")
        _GGUF_DEQUANT_REGISTRY[quant_type] = fn
        return fn

    return _wrap


def get_gguf_dequant_fn(
    quant_type: GGMLQuantizationType,
) -> Optional[Callable[..., torch.Tensor]]:
    return _GGUF_DEQUANT_REGISTRY.get(quant_type)


QK5_0 = 32
BLOCK_Q5_0_BYTES = 22
"""
# Each block is represented by 22 consecutive values in the final ndarray.
typedef struct {
    ggml_half d;           // scale, total 16 bits
    uint8_t qh[4];         // all 32 high bits of quants, total 8*4=32 bits
    uint8_t qs[QK5_0 / 2]; // all 32 low bits of quants, total 8*(32/2)=128 bits
} block_q5_0;              // total 172 bits = 22 bytes
"""


@triton.jit
def dequantize_q5_0_kernel(
    w_ptr,
    out_ptr,
    k,
    QK: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    OUT_IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < k

    block_idx = offs // QK
    pos_in_block = offs % QK

    half_k = QK // 2
    first_half = pos_in_block < half_k
    # [0, 15]
    qs_low_bits_idx = tl.where(first_half, pos_in_block, pos_in_block - half_k)

    block_base = block_idx * BLOCK_BYTES
    w_f16 = tl.cast(w_ptr, tl.pointer_type(tl.float16))
    d = tl.load(w_f16 + block_base // 2, mask=mask, other=0).to(tl.float32)

    qh = (tl.load(w_ptr + block_base + 2, mask=mask, other=0).to(tl.uint32)
          | (tl.load(w_ptr + block_base + 3, mask=mask, other=0).to(tl.uint32)
             << 8)
          | (tl.load(w_ptr + block_base + 4, mask=mask, other=0).to(tl.uint32)
             << 16)
          | (tl.load(w_ptr + block_base + 5, mask=mask, other=0).to(tl.uint32)
             << 24))

    qsb = tl.load(w_ptr + block_base + 6 + qs_low_bits_idx, mask=mask,
                  other=0).to(tl.int32)
    nib_lo = qsb & 0xF
    nib_hi = (qsb >> 4) & 0xF

    qs_low_bits_idx_u32 = qs_low_bits_idx.to(tl.uint32)
    xh_0 = ((qh >> (qs_low_bits_idx_u32 + 0)) << 4) & 0x10
    xh_1 = (qh >> (qs_low_bits_idx_u32 + 12)) & 0x10

    val = tl.where(first_half, nib_lo | xh_0, nib_hi | xh_1)
    y = (val.to(tl.float32) - 16.0) * d

    if OUT_IS_BF16:
        tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offs, y.to(tl.float16), mask=mask)


@register_gguf_dequant(GGMLQuantizationType.Q5_0)
def dequantize_q5_0(
    weight_uint8: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("The output dtype must be float16 or bfloat16")
    if weight_uint8.dtype != torch.uint8:
        raise ValueError("The weight must be uint8 packed weights")

    k = m * n
    if k % QK5_0 != 0:
        raise ValueError(f"element count {k} must be a multiple of {QK5_0}")

    nb = k // QK5_0
    expected_bytes = nb * BLOCK_Q5_0_BYTES
    if weight_uint8.numel() < expected_bytes:
        raise ValueError(
            f"W has {weight_uint8.numel()} bytes, need at least {expected_bytes}"
        )

    if out is None:
        out = torch.empty((m, n), device=weight_uint8.device, dtype=dtype)
    BLOCK = 1024
    grid = (triton.cdiv(k, BLOCK), )
    dequantize_q5_0_kernel[grid](
        weight_uint8,
        out.view(-1),
        k,
        QK=QK5_0,
        BLOCK_BYTES=BLOCK_Q5_0_BYTES,
        OUT_IS_BF16=dtype == torch.bfloat16,
        BLOCK=BLOCK,
    )
    return out


QK8_0 = 32
BLOCK_Q8_0_BYTES = 34
"""
# Each block is represented by 34 consecutive values in the final ndarray.
typedef struct {
    ggml_half d;       // scale, total 16 bits
    int8_t  qs[QK8_0]; // all 32 bits of quants, total 8*32=256 bits
} block_q8_0;          // total 272 bits = 34 bytes
"""


@triton.jit
def dequantize_q8_0_kernel(
    w_ptr,
    out_ptr,
    k,
    QK: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    OUT_IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < k

    block_idx = offs // QK
    pos_in_block = offs % QK

    block_base = block_idx * BLOCK_BYTES
    w_f16 = tl.cast(w_ptr, tl.pointer_type(tl.float16))
    d = tl.load(w_f16 + block_base // 2, mask=mask, other=0).to(tl.float32)

    qb = tl.load(w_ptr + block_base + 2 + pos_in_block, mask=mask,
                 other=0).to(tl.int32)
    qs = tl.where(qb > 127, qb - 256, qb)
    y = qs.to(tl.float32) * d

    if OUT_IS_BF16:
        tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offs, y.to(tl.float16), mask=mask)


@register_gguf_dequant(GGMLQuantizationType.Q8_0)
def dequantize_q8_0(
    weight_uint8: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("The output dtype must be float16 or bfloat16")
    if weight_uint8.dtype != torch.uint8:
        raise ValueError("The weight must be uint8 packed weights")

    k = m * n
    if k % QK8_0 != 0:
        raise ValueError(f"element count {k} must be a multiple of {QK8_0}")

    nb = k // QK8_0
    expected_bytes = nb * BLOCK_Q8_0_BYTES
    if weight_uint8.numel() < expected_bytes:
        raise ValueError(
            f"W has {weight_uint8.numel()} bytes, need at least {expected_bytes}"
        )

    if out is None:
        out = torch.empty((m, n), device=weight_uint8.device, dtype=dtype)
    BLOCK = 1024
    grid = (triton.cdiv(k, BLOCK), )
    dequantize_q8_0_kernel[grid](
        weight_uint8,
        out.view(-1),
        k,
        QK=QK8_0,
        BLOCK_BYTES=BLOCK_Q8_0_BYTES,
        OUT_IS_BF16=dtype == torch.bfloat16,
        BLOCK=BLOCK,
    )
    return out


"""
Two uint8 values form a uint16 value, which is then converted to bfloat16.
"""


@triton.jit
def dequantize_bf16_kernel(
    w_ptr,
    out_ptr,
    k,
    OUT_IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < k

    w_bf16 = tl.cast(w_ptr, tl.pointer_type(tl.bfloat16))
    y = tl.load(w_bf16 + offs, mask=mask, other=0).to(tl.float32)

    if OUT_IS_BF16:
        tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offs, y.to(tl.float16), mask=mask)


@register_gguf_dequant(GGMLQuantizationType.BF16)
def dequantize_bf16(
    weight_uint8: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("The output dtype must be float16 or bfloat16")
    if weight_uint8.dtype != torch.uint8:
        raise ValueError("The weight must be uint8 packed weights")

    k = m * n
    expected_bytes = k * 2
    if weight_uint8.numel() < expected_bytes:
        raise ValueError(
            f"W has {weight_uint8.numel()} bytes, need at least {expected_bytes}"
        )

    if out is None:
        out = torch.empty((m, n), device=weight_uint8.device, dtype=dtype)
    BLOCK = 1024
    grid = (triton.cdiv(k, BLOCK), )
    dequantize_bf16_kernel[grid](
        weight_uint8,
        out.view(-1),
        k,
        OUT_IS_BF16=dtype == torch.bfloat16,
        BLOCK=BLOCK,
    )
    return out
