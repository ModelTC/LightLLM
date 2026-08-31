#!/usr/bin/env python3
"""Compare NCCL and symmetric-memory all-reduce for GLM-5.3 TP8 tensors."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import flashinfer.comm as flashinfer_comm
from flashinfer.comm.mnnvl import TorchDistBackend


def elapsed_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def graph_elapsed_ms(fn, warmup: int, iterations: int) -> float:
    """Measure replay cost, matching LightLLM's decode CUDA-graph path."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def max_rank(value: float) -> float:
    tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=17152)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group_name = dist.group.WORLD.group_name
    rank = dist.get_rank()

    shape = (args.tokens, args.hidden_size)
    # Keep the repeated in-place NCCL input finite without adding a reset copy to
    # the timed region. Correctness of the transport setup is checked separately.
    source = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)

    nccl_input = source.clone()
    dist.all_reduce(nccl_input)
    torch.cuda.synchronize()
    torch.testing.assert_close(nccl_input, torch.zeros_like(nccl_input), rtol=0, atol=0)
    nccl_input.copy_(source)
    torch.cuda.synchronize()
    nccl_ms = elapsed_ms(lambda: dist.all_reduce(nccl_input), args.warmup, args.iterations)
    nccl_graph_ms = graph_elapsed_ms(lambda: dist.all_reduce(nccl_input), args.warmup, args.iterations)

    buffer = symm_mem.empty(source.numel(), device="cuda", dtype=source.dtype)
    handle = symm_mem.rendezvous(buffer, group_name)
    if getattr(handle, "multicast_ptr", 0) == 0:
        raise RuntimeError("symmetric-memory multicast pointer is unavailable")
    symm_output = torch.empty_like(source)

    def symm_all_reduce() -> None:
        buffer.copy_(source.view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
        symm_output.view(-1).copy_(buffer)

    def symm_all_reduce_out_of_place() -> torch.Tensor:
        buffer.copy_(source.view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
        return buffer.view_as(source)

    symm_all_reduce()
    torch.cuda.synchronize()
    torch.testing.assert_close(symm_output, torch.zeros_like(symm_output), rtol=0, atol=0)
    torch.cuda.synchronize()
    symm_ms = elapsed_ms(symm_all_reduce, args.warmup, args.iterations)
    symm_graph_ms = graph_elapsed_ms(symm_all_reduce, args.warmup, args.iterations)
    symm_out_ms = elapsed_ms(symm_all_reduce_out_of_place, args.warmup, args.iterations)
    symm_out_graph_ms = graph_elapsed_ms(symm_all_reduce_out_of_place, args.warmup, args.iterations)

    nccl_max_ms = max_rank(nccl_ms)
    symm_max_ms = max_rank(symm_ms)
    symm_out_max_ms = max_rank(symm_out_ms)
    nccl_graph_max_ms = max_rank(nccl_graph_ms)
    symm_graph_max_ms = max_rank(symm_graph_ms)
    symm_out_graph_max_ms = max_rank(symm_out_graph_ms)

    cpu_group = dist.new_group(list(range(dist.get_world_size())), backend="gloo")
    workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=dist.get_world_size(),
        rank=rank,
        max_token_num=args.tokens,
        hidden_dim=args.hidden_size,
        dtype=source.dtype,
        comm_backend=TorchDistBackend(group=cpu_group),
    )

    def flashinfer_all_reduce() -> torch.Tensor:
        return flashinfer_comm.allreduce_fusion(
            input=source,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
        )

    fi_output = flashinfer_all_reduce()
    torch.cuda.synchronize()
    torch.testing.assert_close(fi_output, torch.zeros_like(fi_output), rtol=0, atol=0)
    fi_ms = elapsed_ms(flashinfer_all_reduce, args.warmup, args.iterations)
    fi_graph_ms = graph_elapsed_ms(flashinfer_all_reduce, args.warmup, args.iterations)
    fi_max_ms = max_rank(fi_ms)
    fi_graph_max_ms = max_rank(fi_graph_ms)

    if rank == 0:
        nbytes = source.numel() * source.element_size()
        eager_best = min(
            (nccl_max_ms, "nccl"),
            (symm_max_ms, "symm"),
            (symm_out_max_ms, "symm_out"),
            (fi_max_ms, "flashinfer"),
        )[1]
        graph_best = min(
            (nccl_graph_max_ms, "nccl"),
            (symm_graph_max_ms, "symm"),
            (symm_out_graph_max_ms, "symm_out"),
            (fi_graph_max_ms, "flashinfer"),
        )[1]
        print(
            f"shape={shape} bytes={nbytes} nccl_ms={nccl_max_ms:.6f} "
            f"symm_multimem_ms={symm_max_ms:.6f} symm_out_ms={symm_out_max_ms:.6f} "
            f"flashinfer_ms={fi_max_ms:.6f} "
            f"best={eager_best} "
            f"graph_nccl_ms={nccl_graph_max_ms:.6f} "
            f"graph_symm_ms={symm_graph_max_ms:.6f} "
            f"graph_symm_out_ms={symm_out_graph_max_ms:.6f} "
            f"graph_flashinfer_ms={fi_graph_max_ms:.6f} "
            f"graph_best={graph_best}"
        )
    workspace.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
