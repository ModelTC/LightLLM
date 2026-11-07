import torch
import triton
import triton.language as tl

from lightllm.distributed import all_gather_into_tensor


@triton.jit
def _sum_shards(
    X,
    Y,
    N: tl.constexpr,
    START: tl.constexpr,
    SIZE: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.full((BLOCK,), 0, tl.float64)
    for rank in range(WORLD):
        value = tl.load(X + rank * N + START + index, index < SIZE, other=0)
        accumulator += value.to(tl.float64)
    tl.store(Y + index, accumulator, index < SIZE)


def ordered_tp_sum(input: torch.Tensor, group, world_size: int, *, scatter_rank=None) -> torch.Tensor:
    if world_size == 1:
        return input
    if scatter_rank is not None:
        if not 0 <= scatter_rank < world_size or input.shape[0] % world_size:
            raise ValueError("Sequence-parallel input must divide evenly across valid ranks")
        shape = (input.shape[0] // world_size, *input.shape[1:])
        output = torch.empty(shape, device=input.device, dtype=input.dtype)
        start = scatter_rank * output.numel()
    else:
        output = input if input.is_contiguous() else torch.empty(input.shape, device=input.device, dtype=input.dtype)
        start = 0
    if input.numel():
        source = input.contiguous()
        symmetric = getattr(group, "symm_mem_reduce", None) if input.is_cuda else None
        gathered = symmetric.ordered_all_gather(source) if symmetric is not None else None
        if gathered is None:
            gathered = torch.empty((world_size, *input.shape), device=input.device, dtype=input.dtype)
            all_gather_into_tensor(gathered.view(-1, *input.shape[1:]), source, group=group)
        if input.is_cuda:
            _sum_shards[(triton.cdiv(output.numel(), 256),)](
                gathered, output, input.numel(), start, output.numel(), world_size, 256, enable_fp_fusion=False
            )
        else:
            accumulator = torch.zeros(output.numel(), dtype=torch.float64, device=input.device)
            shards = gathered.reshape(world_size, -1)
            for rank in range(world_size):
                accumulator.add_(shards[rank, start : start + output.numel()])
            output.copy_(accumulator.reshape(output.shape))
    if scatter_rank is None and output is not input:
        input.copy_(output)
        return input
    return output
