import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_dynamic_conv_kernel(
    hidden,
    dynamic_weights,
    base_kernel,
    conv_output,
    element_num,
    hidden_size: tl.constexpr,
    dynamic_weights_stride: tl.constexpr,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    group_num: tl.constexpr,
    kernel_size: tl.constexpr,
    conv_side: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    element_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token_idx = element_offsets // hidden_size
    hidden_idx = element_offsets % hidden_size
    valid_mask = element_offsets < element_num
    group_idx = hidden_idx // group_size
    block_token_idx = token_idx % block_size

    # Each request occupies one physical draft block; convolution cannot cross it.
    conv_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for kernel_offset in tl.static_range(0, kernel_size):
        in_same_block = block_token_idx >= kernel_offset
        hidden_value = tl.load(
            hidden + (token_idx - kernel_offset) * hidden_size + hidden_idx,
            mask=valid_mask & in_same_block,
            other=0.0,
        ).to(tl.float32)
        base_kernel_offset = (conv_side * kernel_size + kernel_offset) * hidden_size + hidden_idx
        dynamic_weight_offset = (
            token_idx * dynamic_weights_stride + (conv_side * kernel_size + kernel_offset) * group_num + group_idx
        )
        conv_weight = tl.load(base_kernel + base_kernel_offset, mask=valid_mask, other=0.0).to(tl.float32)
        conv_weight += tl.load(dynamic_weights + dynamic_weight_offset, mask=valid_mask, other=0.0).to(tl.float32)
        conv_sum += hidden_value * conv_weight

    tl.store(conv_output + element_offsets, conv_sum, mask=valid_mask)


def grouped_dynamic_conv(
    hidden: torch.Tensor,
    dynamic_weights: torch.Tensor,
    base_kernel: torch.Tensor,
    block_size: int,
    group_size: int,
    conv_side: int,
) -> torch.Tensor:
    """Apply the convolution before (conv_side=0) or after (conv_side=1) a sublayer.

    hidden: [token_num, hidden_size], with consecutive blocks of block_size tokens.
    dynamic_weights: [token_num, 2 * kernel_size * group_num], projected from the
        normalized attention/MLP input; each group shares an additive kernel weight.
    base_kernel: [2, kernel_size, hidden_size], the checkpoint's per-channel weights.
    """

    assert hidden.ndim == 2 and hidden.is_contiguous()
    assert dynamic_weights.ndim == 2 and dynamic_weights.is_contiguous()
    assert base_kernel.ndim == 3 and base_kernel.is_contiguous()
    assert hidden.shape[0] % block_size == 0
    assert hidden.shape[1] % group_size == 0
    assert conv_side in (0, 1)

    token_num, hidden_size = hidden.shape
    conv_side_num, kernel_size, base_hidden_size = base_kernel.shape
    group_num = hidden_size // group_size
    assert conv_side_num == 2
    assert base_hidden_size == hidden_size
    assert dynamic_weights.shape == (token_num, conv_side_num * kernel_size * group_num)

    conv_output = torch.empty_like(hidden)
    element_num = token_num * hidden_size
    num_elements_per_program = 256
    _grouped_dynamic_conv_kernel[(triton.cdiv(element_num, num_elements_per_program),)](
        hidden,
        dynamic_weights,
        base_kernel,
        conv_output,
        element_num,
        hidden_size=hidden_size,
        dynamic_weights_stride=dynamic_weights.shape[1],
        block_size=block_size,
        group_size=group_size,
        group_num=group_num,
        kernel_size=kernel_size,
        conv_side=conv_side,
        BLOCK_SIZE=num_elements_per_program,
    )
    return conv_output
