from functools import lru_cache
from pathlib import Path

import torch
import triton

from lightllm.common.quantization.quantize_method import WeightPack


_KERNEL_DIR = Path("/opt/lightllm-marlin")
_KERNEL_DTYPE = {torch.float16: "fp16", torch.bfloat16: "bf16"}


@lru_cache(maxsize=None)
def _load_module(name):
    path = _KERNEL_DIR / f"{name}.so"
    if not path.is_file():
        return None
    import tvm_ffi

    return tvm_ffi.load_module(str(path))


def is_marlin_packed(pack: WeightPack):
    return pack.weight.shape[-1] == pack.weight_scale.shape[-1] * 2


def can_prepare_awq(pack: WeightPack, kind):
    height, packed_width = pack.weight.shape[-2:]
    width = packed_width * 8
    dtype = _KERNEL_DTYPE.get(pack.weight_scale.dtype)
    return (
        dtype is not None
        and pack.weight.is_cuda
        and height > 0
        and width > 0
        and height % 128 == 0
        and width % 64 == 0
        and pack.weight_scale.shape[-2] == height // 128
        and torch.cuda.get_device_capability(pack.weight.device) == (9, 0)
        and _load_module(f"{kind}_{dtype}") is not None
        and _load_module("repack") is not None
    )


def prepare_awq(pack: WeightPack):
    height, packed_width = pack.weight.shape[-2:]
    width = packed_width * 8
    groups = pack.weight_scale.shape[-2]
    repack = _load_module("repack").awq_marlin_repack
    packed = torch.empty((height // 16, width * 2), dtype=torch.int32, device=pack.weight.device)
    permutation = [row + 8 * column for row in range(8) for column in range(8)]
    shifts = torch.arange(8, device=pack.weight.device, dtype=torch.int32) * 4
    for weight, scales, zeros in zip(
        pack.weight.view(-1, height, packed_width),
        pack.weight_scale.view(-1, groups, width),
        pack.weight_zero_point.view(-1, groups, packed_width),
    ):
        repack(packed, weight, height, width, 4)
        weight.copy_(packed.view_as(weight))
        unpacked_zeros = ((zeros[..., None] >> shifts) & 15)[..., [0, 4, 1, 5, 2, 6, 3, 7]]
        permuted_zeros = unpacked_zeros.reshape(-1, 64)[:, permutation].reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        zeros.copy_((permuted_zeros << shifts).sum(-1).int().reshape_as(zeros))
        scales.copy_(scales.reshape(-1, 64)[:, permutation].reshape_as(scales))
    pack.weight = pack.weight.view(*pack.weight.shape[:-2], height // 16, width * 2)


@lru_cache(maxsize=None)
def _sm_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def dense_awq_marlin(inputs, pack: WeightPack, out=None):
    from sgl_kernel.scalar_type import scalar_types

    height, width = inputs.shape[-1], pack.weight_scale.shape[-1]
    activations = inputs.reshape(-1, height).contiguous()
    if out is None:
        out = torch.empty((*inputs.shape[:-1], width), device=inputs.device, dtype=inputs.dtype)
    if activations.shape[0] == 0:
        return out
    dtype = _KERNEL_DTYPE[inputs.dtype]
    sms = _sm_count(inputs.device)
    row_tile = min(((activations.shape[0] + 15) // 16) * 16, 64)
    temporary = torch.empty(sms * row_tile * 256, dtype=torch.float32, device=inputs.device)
    locks = torch.zeros(sms * 4, dtype=torch.int32, device=inputs.device)
    empty_float = torch.empty(0, dtype=inputs.dtype, device=inputs.device)
    empty_int = torch.empty(0, dtype=torch.int32, device=inputs.device)
    _load_module(f"dense_{dtype}").gptq_marlin_gemm(
        activations,
        pack.weight,
        pack.weight_scale,
        empty_float,
        pack.weight_zero_point,
        empty_int,
        empty_int,
        out.view(-1, width),
        temporary,
        empty_float,
        locks,
        scalar_types.uint4.id,
        True,
        False,
        True,
        False,
    )
    return out


def moe_awq_marlin(inputs, gate, down, route_weights, route_ids, alpha, limit, clamp_up_add_one):
    from sgl_kernel import moe_align_block_size, moe_sum_reduce, silu_and_mul
    from sgl_kernel.scalar_type import scalar_types

    tokens, topk = route_ids.shape
    if tokens == 0:
        return inputs
    experts, hidden, intermediate = gate.weight.shape[0], inputs.shape[-1], down.weight.shape[-2] * 16
    block = next((block for block in (8, 16, 32, 48, 64) if tokens * topk / experts / block < 0.9), 64)
    use_fp32_reduce = tokens >= 96
    if topk == 2 and experts == 8 and 128 <= tokens <= 256 and hidden == 2048 and intermediate == 768:
        block = 32
    routes = route_ids.numel()
    maximum = min(routes * block, routes + (experts + 1) * (block - 1))
    device = inputs.device
    sorted_ids = torch.empty(maximum, device=device, dtype=torch.int32)
    expert_ids = torch.empty(triton.cdiv(maximum, block), device=device, dtype=torch.int32)
    padded = torch.empty(1, device=device, dtype=torch.int32)
    cumsum = torch.empty(experts + 2, device=device, dtype=torch.int32)
    moe_align_block_size(
        route_ids.contiguous().to(torch.int32), experts + 1, block, sorted_ids, expert_ids, padded, cumsum, True
    )
    route_weights = route_weights.contiguous().to(torch.float32)
    empty_float = torch.empty(0, dtype=inputs.dtype, device=device)
    empty_int = torch.empty(0, dtype=torch.int32, device=device)
    max_blocks = _sm_count(device) * 4
    locks = torch.zeros(max_blocks, dtype=torch.int32, device=device)
    scratch = torch.empty(
        max_blocks * max(block, 16) * 256 if use_fp32_reduce else 0, dtype=torch.float32, device=device
    )
    kernel = _load_module(f"moe_{_KERNEL_DTYPE[inputs.dtype]}").moe_wna16_marlin_gemm

    def gemm(activation, pack, width, route_count, apply_route_weight):
        out = torch.empty((routes, width), dtype=inputs.dtype, device=device)
        kernel(
            activation,
            out,
            pack.weight,
            empty_float,
            pack.weight_scale,
            empty_float,
            pack.weight_zero_point,
            empty_int,
            empty_int,
            locks,
            sorted_ids,
            expert_ids,
            padded,
            route_weights,
            empty_float,
            scratch,
            block,
            route_count,
            apply_route_weight,
            False,
            scalar_types.uint4.id,
            activation.shape[0],
            width,
            activation.shape[1],
            False,
            False,
            True,
            True,
            pack.weight_scale.shape[1],
            128 if pack.weight_scale.shape[1] > 1 else -1,
            not use_fp32_reduce,
            use_fp32_reduce,
            False,
        )
        return out

    gate_out = gemm(inputs.contiguous(), gate, 2 * intermediate, topk, False)
    if alpha is None and limit is None:
        activated = silu_and_mul(gate_out)
    else:
        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import silu_and_mul_fwd

        activated = torch.empty((routes, intermediate), dtype=inputs.dtype, device=device)
        silu_and_mul_fwd(gate_out, activated, limit=limit, alpha=alpha, clamp_up_add_one=clamp_up_add_one)
    expert_out = gemm(activated, down, hidden, 1, True)
    moe_sum_reduce(expert_out.view(tokens, topk, hidden), inputs, 1.0)
    return inputs
