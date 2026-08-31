#!/usr/bin/env python3
"""Exercise the production SymmMem alias path in normal and inference modes."""

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from lightllm.distributed.symm_mem_all_reduce import SymmMemAllreduce


def _check(reducer: SymmMemAllreduce, *, inference: bool) -> None:
    rank = dist.get_rank()
    expected = sum(range(1, dist.get_world_size() + 1))
    context = torch.inference_mode() if inference else nullcontext()
    with context:
        for rows in (64, 32, 64):
            value = torch.full(
                (rows, 4096),
                rank + 1,
                dtype=torch.bfloat16,
                device="cuda",
            )
            value.data = reducer.all_reduce_out_of_place(value)
            # Match the model's immediate post-reduction view operation.
            viewed = value.view(rows, 4, 1024)
            torch.testing.assert_close(
                viewed,
                torch.full_like(viewed, expected),
                rtol=0,
                atol=0,
            )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    reducer = SymmMemAllreduce(
        dist.group.WORLD,
        torch.cuda.current_device(),
        dtype=torch.bfloat16,
    )
    if reducer.disabled:
        raise RuntimeError("SymmMemAllreduce unexpectedly disabled")
    if not reducer.buffer.is_inference():
        raise RuntimeError("SymmMem workspace is not an inference tensor")
    _check(reducer, inference=False)
    _check(reducer, inference=True)
    dist.barrier()
    if dist.get_rank() == 0:
        print("SymmMem out-of-place normal+inference alias checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
