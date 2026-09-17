import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_dynamic_conv_kernel(
    hidden,
    dynamic,
    base_kernel,
    output,
    element_num,
    hidden_size: tl.constexpr,
    dynamic_stride: tl.constexpr,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    group_num: tl.constexpr,
    kernel_size: tl.constexpr,
    side: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = offsets // hidden_size
    channel = offsets % hidden_size
    valid = offsets < element_num
    group = channel // group_size
    block_offset = row % block_size

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for tap in tl.static_range(0, kernel_size):
        has_predecessor = block_offset >= tap
        value = tl.load(
            hidden + (row - tap) * hidden_size + channel,
            mask=valid & has_predecessor,
            other=0.0,
        ).to(tl.float32)
        base_offset = (side * kernel_size + tap) * hidden_size + channel
        dynamic_offset = row * dynamic_stride + (side * kernel_size + tap) * group_num + group
        weight = tl.load(base_kernel + base_offset, mask=valid, other=0.0).to(tl.float32)
        weight += tl.load(dynamic + dynamic_offset, mask=valid, other=0.0).to(tl.float32)
        accumulator += value * weight

    tl.store(output + offsets, accumulator, mask=valid)


def grouped_dynamic_conv(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel: torch.Tensor,
    block_size: int,
    group_size: int,
    side: int,
) -> torch.Tensor:
    """Apply one side of DFlash2's grouped dynamic causal convolution."""

    assert hidden.ndim == 2 and hidden.is_contiguous()
    assert dynamic.ndim == 2 and dynamic.is_contiguous()
    assert base_kernel.ndim == 3 and base_kernel.is_contiguous()
    assert hidden.shape[0] % block_size == 0
    assert hidden.shape[1] % group_size == 0
    assert side in (0, 1)

    token_num, hidden_size = hidden.shape
    side_num, kernel_size, base_hidden_size = base_kernel.shape
    group_num = hidden_size // group_size
    assert side_num == 2
    assert base_hidden_size == hidden_size
    assert dynamic.shape == (token_num, side_num * kernel_size * group_num)

    output = torch.empty_like(hidden)
    element_num = token_num * hidden_size
    block = 256
    _grouped_dynamic_conv_kernel[(triton.cdiv(element_num, block),)](
        hidden,
        dynamic,
        base_kernel,
        output,
        element_num,
        hidden_size=hidden_size,
        dynamic_stride=dynamic.shape[1],
        block_size=block_size,
        group_size=group_size,
        group_num=group_num,
        kernel_size=kernel_size,
        side=side,
        BLOCK_SIZE=block,
    )
    return output
