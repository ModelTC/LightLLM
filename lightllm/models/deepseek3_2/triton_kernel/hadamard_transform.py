import torch
import triton
import triton.language as tl


@triton.jit
def _butterfly_stage(x, GROUPS: tl.constexpr, STEP: tl.constexpr, BLOCK_N: tl.constexpr):
    x_grouped = tl.reshape(x, (GROUPS, 2, STEP))
    x_grouped = tl.permute(x_grouped, (0, 2, 1))
    left, right = tl.split(x_grouped)
    x_pair = tl.join(left + right, left - right)
    x_pair = tl.permute(x_pair, (0, 2, 1))
    return tl.reshape(x_pair, (BLOCK_N,))


@triton.jit
def _hadamard_transform_kernel(
    X,
    Y,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    x = tl.load(X + row * BLOCK_N + offsets).to(tl.float32)

    x = _butterfly_stage(x, 64, 1, BLOCK_N)
    x = _butterfly_stage(x, 32, 2, BLOCK_N)
    x = _butterfly_stage(x, 16, 4, BLOCK_N)
    x = _butterfly_stage(x, 8, 8, BLOCK_N)
    x = _butterfly_stage(x, 4, 16, BLOCK_N)
    x = _butterfly_stage(x, 2, 32, BLOCK_N)
    x = _butterfly_stage(x, 1, 64, BLOCK_N)

    tl.store(Y + row * BLOCK_N + offsets, x * scale)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    assert x.is_cuda, "hadamard_transform only supports CUDA tensors"
    assert x.dtype == torch.bfloat16, "hadamard_transform expects bfloat16 input"

    original_shape = x.shape
    hidden_size = x.size(-1)
    assert hidden_size == 128, "DeepSeek-V3.2 Hadamard transform expects hidden size 128"

    x = x.contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // hidden_size
    _hadamard_transform_kernel[(rows,)](
        x,
        out,
        scale,
        BLOCK_N=hidden_size,
        num_warps=4,
    )

    return out.view(original_shape)
