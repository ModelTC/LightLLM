"""Prototype TP head/token transpose for GLM-5.3 sparse prefill attention."""

import argparse
import gc
import json
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist


def make_causal_indices(tokens: int, sequence_length: int, topk: int) -> torch.Tensor:
    token_ids = torch.arange(tokens, dtype=torch.int32, device="cuda")
    positions = token_ids.remainder(sequence_length)
    sequence_starts = token_ids - positions
    columns = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, -1)
    indices = sequence_starts.view(-1, 1) + columns
    indices.masked_fill_(columns > positions.view(-1, 1), -1)
    return indices.unsqueeze(1)


def head_shards_to_token_shards(q: torch.Tensor, world_size: int) -> torch.Tensor:
    tokens, local_heads, head_dim = q.shape
    if tokens % world_size:
        raise ValueError(f"tokens={tokens} must be divisible by world_size={world_size}")
    tokens_per_rank = tokens // world_size
    received = torch.empty_like(q)
    dist.all_to_all_single(received, q)
    return (
        received.view(world_size, tokens_per_rank, local_heads, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(tokens_per_rank, world_size * local_heads, head_dim)
    )


def token_shards_to_head_shards(output: torch.Tensor, world_size: int) -> torch.Tensor:
    tokens_per_rank, global_heads, head_dim = output.shape
    if global_heads % world_size:
        raise ValueError(f"global_heads={global_heads} must be divisible by world_size={world_size}")
    local_heads = global_heads // world_size
    send = (
        output.view(tokens_per_rank, world_size, local_heads, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(world_size * tokens_per_rank, local_heads, head_dim)
    )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send)
    return received


def global_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def measure_ms(
    name: str,
    operation: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float | str]:
    result = None
    for _ in range(warmup):
        result = operation()
        torch.cuda.synchronize()
        del result

    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    samples = []
    for _ in range(iterations):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        samples.append(global_max(start.elapsed_time(end), device))
        del result

    peak_delta_gib = global_max(
        (torch.cuda.max_memory_allocated() - baseline_bytes) / 2 ** 30,
        device,
    )
    return {
        "name": name,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "peak_delta_gib": peak_delta_gib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--local-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    global_heads = args.local_heads * world_size
    if args.tokens % world_size:
        raise ValueError("--tokens must be divisible by the TP world size")
    if global_heads != 64:
        raise ValueError(f"FlashMLA on Hopper requires 64 global heads, got {global_heads}")

    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    torch.manual_seed(1000 + rank)
    q = torch.randn(
        (args.tokens, args.local_heads, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    torch.manual_seed(0)
    kv = torch.randn(
        (args.tokens, 1, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    indices = make_causal_indices(args.tokens, args.sequence_length, args.topk)
    scale = args.head_dim ** -0.5
    token_start = rank * (args.tokens // world_size)
    token_end = token_start + args.tokens // world_size
    local_indices = indices[token_start:token_end]
    comm_workspace = torch.empty_like(q)
    transpose_workspace = torch.empty(
        (args.tokens // world_size, global_heads, args.head_dim),
        dtype=q.dtype,
        device=q.device,
    )

    def padded_flashmla() -> torch.Tensor:
        q_input = q.new_zeros((args.tokens, global_heads, args.head_dim))
        q_input[:, : args.local_heads].copy_(q)
        return flash_mla_sparse_fwd(q_input, kv, indices, scale, d_v=args.head_dim,)[
            0
        ][:, : args.local_heads]

    def transposed_flashmla() -> torch.Tensor:
        transposed_q = head_shards_to_token_shards(q, world_size)
        transposed_output = flash_mla_sparse_fwd(
            transposed_q,
            kv,
            local_indices,
            scale,
            d_v=args.head_dim,
        )[0]
        return token_shards_to_head_shards(transposed_output, world_size)

    def transposed_flashmla_workspace() -> torch.Tensor:
        dist.all_to_all_single(comm_workspace, q)
        comm_rank_major = comm_workspace.view(
            world_size,
            args.tokens // world_size,
            args.local_heads,
            args.head_dim,
        )
        transpose_workspace.view(
            args.tokens // world_size,
            world_size,
            args.local_heads,
            args.head_dim,
        ).copy_(comm_rank_major.permute(1, 0, 2, 3))
        transposed_output = flash_mla_sparse_fwd(
            transpose_workspace,
            kv,
            local_indices,
            scale,
            d_v=args.head_dim,
        )[0]
        comm_workspace.view(world_size, args.tokens // world_size, args.local_heads, args.head_dim,).copy_(
            transposed_output.view(
                args.tokens // world_size,
                world_size,
                args.local_heads,
                args.head_dim,
            ).permute(1, 0, 2, 3)
        )
        dist.all_to_all_single(transpose_workspace.view_as(q), comm_workspace)
        return transpose_workspace.view_as(q)

    validation = None
    if args.validate:
        baseline = padded_flashmla().contiguous()
        candidate = transposed_flashmla_workspace().clone()
        torch.cuda.synchronize()
        difference = (baseline - candidate).abs().float()
        validation = {
            "max_abs_diff": global_max(difference.max().item(), device),
            "mean_abs_diff_max_rank": global_max(difference.mean().item(), device),
            "allclose": bool(torch.allclose(baseline, candidate, rtol=0.05, atol=0.05)),
        }
        validation_tensor = torch.tensor(int(validation["allclose"]), device=device)
        dist.all_reduce(validation_tensor, op=dist.ReduceOp.MIN)
        validation["allclose"] = bool(validation_tensor.item())
        del baseline, candidate, difference
        torch.cuda.empty_cache()

    results = []
    for name, operation in (
        ("flashmla_head_padding", padded_flashmla),
        ("flashmla_tp_head_token_transpose", transposed_flashmla),
        ("flashmla_tp_transpose_workspace", transposed_flashmla_workspace),
    ):
        results.append(measure_ms(name, operation, args.warmup, args.iterations, device))

    if rank == 0:
        baseline_ms = results[0]["median_ms"]
        candidate_ms = results[-1]["median_ms"]
        print(
            json.dumps(
                {
                    "shape": {
                        "tokens_per_rank_before_transpose": args.tokens,
                        "tokens_per_rank_after_transpose": args.tokens // world_size,
                        "sequence_length": args.sequence_length,
                        "topk": args.topk,
                        "local_heads": args.local_heads,
                        "global_heads": global_heads,
                        "head_dim": args.head_dim,
                        "world_size": world_size,
                    },
                    "validation": validation,
                    "results": results,
                    "transpose_speedup_over_padding": baseline_ms / candidate_ms,
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
