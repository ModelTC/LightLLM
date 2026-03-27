import torch

import triton
import triton.language as tl


@triton.jit
def _quant_scale(max_nope, fp8_max):
    return tl.exp2(tl.ceil(tl.log2(tl.maximum(max_nope / fp8_max, 1e-4))))


@triton.jit
def _fwd_kernel_destindex_copy_kv_flashmla_fp8(
    KV_nope,
    KV_rope,
    Dest_loc,
    O_nope,
    O_scale,
    O_rope,
    stride_kv_nope_bs,
    stride_kv_nope_h,
    stride_kv_nope_d,
    stride_kv_rope_bs,
    stride_kv_rope_h,
    stride_kv_rope_d,
    stride_o_nope_bs,
    stride_o_nope_h,
    stride_o_nope_d,
    stride_o_scale_bs,
    stride_o_scale_h,
    stride_o_scale_d,
    stride_o_rope_bs,
    stride_o_rope_h,
    stride_o_rope_d,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)

    offs_rope = tl.arange(0, BLOCK_DMODEL_ROPE)

    # This kernel is only used by the DeepSeek-V3.2 DSA FP8 path, which
    # stores a single MQA-style KV head per token. Keep all accesses 1-D so
    # Triton treats per-tile scales as scalars instead of 1-element blocks.
    kv_rope_ptrs = KV_rope + cur_index * stride_kv_rope_bs + stride_kv_rope_d * offs_rope

    kv_rope = tl.load(kv_rope_ptrs)

    o_rope_ptrs = O_rope + dest_index * stride_o_rope_bs + stride_o_rope_d * offs_rope
    tl.store(o_rope_ptrs, kv_rope)

    num_tiles = BLOCK_DMODEL_NOPE // GROUP_SIZE
    for tile_idx in range(0, num_tiles):
        offs_tile = tile_idx * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        kv_nope_tile_ptrs = KV_nope + cur_index * stride_kv_nope_bs + stride_kv_nope_d * offs_tile
        kv_nope_tile = tl.load(kv_nope_tile_ptrs)
        max_nope = tl.max(tl.abs(kv_nope_tile), axis=0)
        kv_scale = _quant_scale(max_nope, FP8_MAX)
        kv_nope_fp8 = tl.clamp(kv_nope_tile / kv_scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)

        o_nope_ptrs = (
            O_nope
            + dest_index * stride_o_nope_bs
            + (tile_idx * GROUP_SIZE) * stride_o_nope_d
            + tl.arange(0, GROUP_SIZE) * stride_o_nope_d
        )
        tl.store(o_nope_ptrs, kv_nope_fp8)

        o_scale_ptrs = O_scale + dest_index * stride_o_scale_bs + tile_idx * stride_o_scale_d
        tl.store(o_scale_ptrs, kv_scale.to(tl.float32))
    return


@torch.no_grad()
def destindex_copy_kv_flashmla_fp8(
    KV_nope: torch.Tensor,
    KV_rope: torch.Tensor,
    DestLoc: torch.Tensor,
    O_nope: torch.Tensor,
    O_scale: torch.Tensor,
    O_rope: torch.Tensor,
):
    seq_len = DestLoc.shape[0]
    kv_nope_head_dim = KV_nope.shape[2]
    kv_rope_head_dim = KV_rope.shape[2]

    assert kv_nope_head_dim == 512, f"Expected kv_nope_head_dim=512, got {kv_nope_head_dim}"
    assert kv_rope_head_dim == 64, f"Expected kv_rope_head_dim=64, got {kv_rope_head_dim}"
    assert O_nope.shape[2] == 512
    assert O_scale.shape[2] == 4
    assert O_rope.shape[2] == 64

    _fwd_kernel_destindex_copy_kv_flashmla_fp8[(seq_len,)](
        KV_nope,
        KV_rope,
        DestLoc,
        O_nope,
        O_scale,
        O_rope,
        KV_nope.stride(0),
        KV_nope.stride(1),
        KV_nope.stride(2),
        KV_rope.stride(0),
        KV_rope.stride(1),
        KV_rope.stride(2),
        O_nope.stride(0),
        O_nope.stride(1),
        O_nope.stride(2),
        O_scale.stride(0),
        O_scale.stride(1),
        O_scale.stride(2),
        O_rope.stride(0),
        O_rope.stride(1),
        O_rope.stride(2),
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        BLOCK_DMODEL_NOPE=512,
        BLOCK_DMODEL_ROPE=64,
        GROUP_SIZE=128,
        num_warps=4,
        num_stages=1,
    )
    return


def pack_kv_reference(kv: torch.Tensor) -> torch.Tensor:
    assert kv.shape[-1] == 576
    kv = kv.view(-1, 1, 576)
    kv_nope = kv[:, :, :512].contiguous().view(-1, 512)
    kv_rope = kv[:, :, 512:].contiguous().view(-1, 64)
    out = torch.empty((kv.shape[0], 1, 656), dtype=torch.uint8, device=kv.device)
    out_nope = out[:, :, :512].view(torch.float8_e4m3fn).view(-1, 1, 512)
    out_scale = out[:, :, 512:528].view(torch.float32).view(-1, 1, 4)
    out_rope = out[:, :, 528:].view(torch.bfloat16).view(-1, 1, 64)
    out_rope.copy_(kv_rope.view(-1, 1, 64))
    for tile_idx in range(4):
        start = tile_idx * 128
        end = start + 128
        tile = kv_nope[:, start:end]
        scale = torch.pow(2, torch.clamp_min(tile.abs().amax(dim=-1).float() / 448.0, 1e-4).log2().ceil())
        out_scale[:, 0, tile_idx] = scale
        out_nope[:, 0, start:end] = (tile.float() / scale[:, None]).to(torch.float8_e4m3fn)
    return out


def dequantize_kv_reference(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.view(-1, 1, 656)
    out = torch.empty((packed.shape[0], 1, 576), dtype=torch.bfloat16, device=packed.device)
    packed_nope = packed[:, :, :512].view(torch.float8_e4m3fn).view(-1, 1, 512)
    packed_scale = packed[:, :, 512:528].view(torch.float32).view(-1, 1, 4)
    packed_rope = packed[:, :, 528:].view(torch.bfloat16).view(-1, 1, 64)
    out[:, :, 512:].copy_(packed_rope)
    for tile_idx in range(4):
        start = tile_idx * 128
        end = start + 128
        out[:, :, start:end] = (
            packed_nope[:, :, start:end].to(torch.float32) * packed_scale[:, :, tile_idx : tile_idx + 1]
        )
    return out
