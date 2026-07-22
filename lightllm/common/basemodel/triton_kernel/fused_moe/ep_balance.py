import torch
import triton
import triton.language as tl


PREFILL_PHASE = 0
DECODE_PHASE = 1
ROUTE_LOAD = 0
COMPUTE_LOAD = 1


@triton.jit
def _accumulate_ep_balance_kernel(
    expert_counts_ptr,
    counters_ptr,
    expert_num: tl.constexpr,
    phase: tl.constexpr,
    alignment: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    counts = tl.load(expert_counts_ptr + offsets, mask=offsets < expert_num, other=0).to(tl.int64)
    route_load = tl.sum(counts)
    compute_load = tl.sum((counts + alignment - 1) // alignment * alignment)
    tl.atomic_add(counters_ptr + phase * 2, route_load)
    tl.atomic_add(counters_ptr + phase * 2 + 1, compute_load)


@torch.no_grad()
def accumulate_ep_balance(
    expert_counts: torch.Tensor,
    counters: torch.Tensor,
    *,
    is_prefill: bool,
):
    """Accumulate one rank's route and effective-compute loads without a CPU sync."""
    assert expert_counts.is_cuda and expert_counts.is_contiguous() and expert_counts.ndim == 1
    assert counters.is_cuda and counters.dtype == torch.int64 and tuple(counters.shape) == (2, 2)
    expert_num = expert_counts.numel()
    if expert_num == 0:
        return
    phase = PREFILL_PHASE if is_prefill else DECODE_PHASE
    alignment = 128 if is_prefill else 1
    _accumulate_ep_balance_kernel[(1,)](
        expert_counts,
        counters,
        expert_num=expert_num,
        phase=phase,
        alignment=alignment,
        BLOCK_SIZE=triton.next_power_of_2(expert_num),
        num_warps=4,
    )
