"""固定节点优先、覆盖数次优路由的指标与 launch 对照；通信指标不冒充实际通信耗时。

PYTHONPATH=. python test/benchmark/kernel/benchmark_eplb_decode_rank_weighted.py \
    --metrics-only --output /tmp/weighted_metrics.json
PYTHONPATH=. python test/benchmark/kernel/benchmark_eplb_decode_rank_weighted.py \
    --launch-only --output /tmp/weighted_launch.json

默认全局 128 token、每源 rank 16，统计样本按真实小 batch 划分；各 batch 的 token_index
重新从 0 开始，不使用不断递增的全局行号来掩盖固定 Graph 的偏好。校准种子 2025，评测种子独立。
node_world_size=4 只模拟两个节点的元数据，并不把单机 NVLink 测量声称为真实 RDMA 收益。
历史标签 current 表示 hash_targets 对照，不是当前服务默认值；weighted 为默认加权路由。
"""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch

import eplb_decode_benchmark_utils as benchmark_utils
from eplb_decode_benchmark_utils import make_placement, make_traffic, capture, elapsed_us, token_random_values
from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
    eplb_hash_targets_topk_ids,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_rank_weighted import (
    eplb_rank_weighted_topk_ids,
)
from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    build_logical_to_rank_physical_map,
    build_logical_to_physical_map,
)


def route_cpu(
    logical, placement, source_rank, node_world_size, variant, token_period=0, normalize_replica_coverage=True
):
    """独立 NumPy 参考；token_period 非零时，每个真实小 batch 重置行号。"""
    experts, world, slots = max(max(row) for row in placement) + 1, len(placement), len(placement[0])
    logical_to_rank_physical_map = np.asarray(build_logical_to_rank_physical_map(placement, experts), dtype=np.int32)
    available = logical_to_rank_physical_map[logical]
    local_ids = available[:, :, source_rank]
    remote = local_ids < 0
    can_execute = (available >= 0) & remote[:, :, None]
    tokens = np.arange(len(logical), dtype=np.uint32)
    if token_period:
        tokens %= token_period
    if variant == "current":
        rows = np.asarray(
            build_logical_to_physical_map(
                placement, experts, current_rank=source_rank, node_world_size=node_world_size
            ),
            dtype=np.int32,
        )
        counts = rows[logical, 0]
        if node_world_size > 1:
            node_counts = rows[logical, 1]
            counts = np.where(node_counts > 0, node_counts, counts)
        initial_random = token_random_values(source_rank=source_rank, token_indices=tokens)[:, None]
        initial_ids = rows[logical, 3 + initial_random % counts.astype(np.uint32)]
        initial_ids = np.where(remote, initial_ids, local_ids)
        initial_ranks = initial_ids // slots
        # 与生产 uint32 offset 的回绕及独立 seed 一致。
        rank_bits = (world - 1).bit_length()
        rank_offsets = (tokens[:, None] << np.uint32(rank_bits)) | np.arange(world, dtype=np.uint32)[None, :]
        rank_scores = benchmark_utils.token_random_values(
            source_rank=source_rank | (1 << 32), token_indices=rank_offsets
        )
        best_ranks = initial_ranks.copy()
        best_scores = np.take_along_axis(rank_scores, initial_ranks, axis=1)
        for partner_slot in range(logical.shape[1]):
            partner_ranks = initial_ranks[:, partner_slot]
            indices = np.broadcast_to(partner_ranks[:, None, None], (*logical.shape, 1))
            physical = np.take_along_axis(available, indices, axis=2)[:, :, 0]
            scores = rank_scores[np.arange(len(logical)), partner_ranks][:, None]
            better = remote & (physical >= 0) & (scores < best_scores)
            best_ranks = np.where(better, partner_ranks[:, None], best_ranks)
            best_scores = np.where(better, scores, best_scores)
    elif variant == "weighted":
        if normalize_replica_coverage:
            replica_counts = can_execute.sum(axis=2)
            expert_weights = 1024 // np.maximum(replica_counts, 1)
            coverage = (can_execute * expert_weights[:, :, None]).sum(axis=1)
        else:
            coverage = can_execute.sum(axis=1)
        same_node = np.arange(world) // node_world_size == source_rank // node_world_size
        score = coverage + same_node[None, :] * 10000000
        # CPU 模拟同分时选最小 rank，仅是一种合法代表；不模拟 GPU 的归约顺序。
        best_ranks = np.where(can_execute, score[:, None, :], -1).argmax(axis=2)
    else:
        raise ValueError(f"Unsupported routing variant: {variant!r}")
    selected = np.take_along_axis(available, best_ranks[:, :, None], axis=2)[:, :, 0]
    output = np.where(remote, selected, local_ids)
    assert np.array_equal(np.asarray(placement).flatten()[output], logical)
    return output


def validate_route_cpu(
    logical, output, placement, source_rank, node_world_size, variant, normalize_replica_coverage=True
):
    """校验真实输出；加权模式验证最高分副本集合，而非指定同分物理槽。"""
    assert output.shape == logical.shape
    physical_to_logical = np.asarray(placement).flatten()
    assert np.all((output >= 0) & (output < len(physical_to_logical)))
    assert np.array_equal(physical_to_logical[output], logical)
    if variant == "current":
        assert np.array_equal(output, route_cpu(logical, placement, source_rank, node_world_size, variant))
        return
    assert variant == "weighted"
    experts = int(physical_to_logical.max()) + 1
    logical_to_rank_physical_map = np.asarray(build_logical_to_rank_physical_map(placement, experts), dtype=np.int32)
    available = logical_to_rank_physical_map[logical]  # [T,K,W]：每个逻辑专家的候选物理槽。
    local_ids = available[:, :, source_rank]  # [T,K]：本卡槽。
    remote = local_ids < 0
    can_execute = (available >= 0) & remote[:, :, None]
    if normalize_replica_coverage:
        replica_counts = can_execute.sum(axis=2)  # [T,K]：全局副本 rank 数，本卡/补齐不贡献评分。
        expert_weights = 1024 // np.maximum(replica_counts, 1)
        coverage = (can_execute * expert_weights[:, :, None]).sum(axis=1)  # [T,W]：整数倒数加权分。
    else:
        coverage = can_execute.sum(axis=1)  # [T,W]：仅统计远端专家覆盖数。
    same_node = np.arange(len(placement)) // node_world_size == source_rank // node_world_size
    scores = coverage + same_node[None, :] * 10000000
    candidate_scores = np.where(can_execute, scores[:, None, :], -1)
    selected_ranks = output // len(placement[0])
    selected_scores = np.take_along_axis(candidate_scores, selected_ranks[:, :, None], axis=2)[:, :, 0]
    assert np.all(output[~remote] == local_ids[~remote])
    assert np.all(selected_scores[remote] == candidate_scores.max(axis=2)[remote])


def metrics(placement, traffic, node_world_size, global_tokens, variant, normalize_replica_coverage=True):
    """CPU 布局指标；weighted 同分选最小 rank，不代表 GPU 实际通信或计算负载。"""
    world, slots = len(placement), len(placement[0])
    local_tokens = global_tokens // world
    batches = len(traffic) // global_tokens
    logical = traffic.numpy().reshape(batches, world, local_tokens, -1)
    all_routes = np.empty(logical.shape, dtype=np.int32)
    for source_rank in range(world):
        output = route_cpu(
            logical[:, source_rank].reshape(-1, logical.shape[-1]),
            placement,
            source_rank,
            node_world_size,
            variant,
            token_period=local_tokens,
            normalize_replica_coverage=normalize_replica_coverage,
        )
        all_routes[:, source_rank] = output.reshape(batches, local_tokens, -1)
    ranks = all_routes // slots
    targets = (ranks[:, :, :, :, None] == np.arange(world)[None, None, None, None, :]).any(axis=3)
    sources = np.arange(world)[:, None]
    remote = targets & (np.arange(world)[None, :] != sources)[None, :, None, :]
    cross_node = remote & (np.arange(world)[None, :] // node_world_size != sources // node_world_size)[None, :, None, :]
    counts = np.zeros((batches, world * slots), dtype=np.int64)
    for batch in range(batches):
        counts[batch] = np.bincount(all_routes[batch].flatten(), minlength=world * slots)
    counts = counts.reshape(batches, world, slots)
    rank_load = counts.sum(axis=2)
    active = (counts > 0).sum(axis=2)
    padded = ((counts + 127) // 128 * 128).sum(axis=2)
    return dict(
        mean_remote_ranks=float(remote.sum() / len(traffic)),
        mean_cross_node_ranks=float(cross_node.sum() / len(traffic)),
        median_max_rank_load_over_mean=float(np.median(rank_load.max(axis=1) / rank_load.mean(axis=1))),
        p95_max_rank_load_over_mean=float(np.percentile(rank_load.max(axis=1) / rank_load.mean(axis=1), 95)),
        median_max_active_experts=float(np.median(active.max(axis=1))),
        median_total_padded_rows=float(np.median(padded.sum(axis=1))),
        median_critical_padded_rows=float(np.median(padded.max(axis=1))),
    )


def screen(args):
    results = []
    for workload in args.workloads:
        for redundant in args.redundant:
            placement = make_placement(args.experts, args.world_size, redundant, workload, args.topk)
            for seed in args.seeds:
                traffic = make_traffic(args.metric_tokens, args.experts, args.topk, workload, seed)
                for node_world_size in args.node_world_sizes:
                    variants = [
                        dict(variant="current"),
                        dict(variant="weighted", normalize_replica_coverage=False),
                        dict(variant="weighted", normalize_replica_coverage=True),
                    ]
                    for spec in variants:
                        result = dict(
                            workload=workload,
                            R=redundant,
                            seed=seed,
                            node_world_size=node_world_size,
                            **spec,
                            **metrics(placement, traffic, node_world_size, args.global_tokens, **spec),
                        )
                        results.append(result)
                    print(
                        json.dumps(
                            dict(
                                progress="metrics",
                                workload=workload,
                                R=redundant,
                                seed=seed,
                                node_world_size=node_world_size,
                                cases=len(variants),
                            )
                        ),
                        flush=True,
                    )
    return results


def launch_benchmark(args):
    results = []
    for world in (8, 16, 32):
        placement = make_placement(args.experts, world, 4, "uniform", args.topk)
        logical_to_rank_physical_map = torch.tensor(
            build_logical_to_rank_physical_map(placement, args.experts), dtype=torch.int32, device="cuda"
        )
        rows = torch.tensor(
            build_logical_to_physical_map(placement, args.experts, current_rank=0, node_world_size=world),
            dtype=torch.int32,
            device="cuda",
        )
        for tokens in (16, 128):
            logical = make_traffic(tokens, args.experts, args.topk, "uniform", 2026).contiguous().cuda()
            common = dict(
                logical_topk_ids=logical, logical_to_rank_physical_map=logical_to_rank_physical_map, current_rank=0
            )
            variants = [
                (
                    "current",
                    eplb_hash_targets_topk_ids,
                    dict(
                        common,
                        logical_to_physical_map=rows,
                        num_physical_experts_per_rank=len(placement[0]),
                        node_world_size=world,
                    ),
                )
            ]
            # 评测公开接口实际使用的内部布局，不再从外部覆盖分块和 warp 数。
            variants.append(
                (
                    "matrix",
                    eplb_rank_weighted_topk_ids,
                    dict(common, node_world_size=world, normalize_replica_coverage=False),
                )
            )
            variants.append(
                (
                    "matrix_normalized",
                    eplb_rank_weighted_topk_ids,
                    dict(common, node_world_size=world, normalize_replica_coverage=True),
                )
            )
            graphs = []
            for name, function, kwargs in variants:
                result = function(**kwargs)
                validate_route_cpu(
                    logical.cpu().numpy(),
                    result.cpu().numpy(),
                    placement,
                    0,
                    world,
                    "current" if name == "current" else "weighted",
                    normalize_replica_coverage=kwargs.get("normalize_replica_coverage", True),
                )
                graphs.append(capture(function, kwargs, args.calls))
            samples = [[] for _ in graphs]
            for round_index in range(args.rounds):
                indices = range(len(graphs)) if round_index % 2 == 0 else range(len(graphs) - 1, -1, -1)
                for index in indices:
                    samples[index].append(elapsed_us(graphs[index], args.calls))
            result = dict(
                world_size=world,
                tokens=tokens,
                median_us={spec[0]: statistics.median(samples[index]) for index, spec in enumerate(variants)},
                samples_us={spec[0]: samples[index] for index, spec in enumerate(variants)},
            )
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "samples_us"}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--launch-only", action="store_true")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--global-tokens", type=int, default=128)
    parser.add_argument("--metric-tokens", type=int, default=4096)
    parser.add_argument("--workloads", nargs="+", default=["uniform", "cooccurrence"])
    parser.add_argument("--redundant", nargs="+", type=int, default=[1, 2, 4, 16, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--node-world-sizes", nargs="+", type=int, default=[8, 4])
    parser.add_argument("--calls", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.global_tokens % args.world_size == 0 and args.metric_tokens % args.global_tokens == 0
    assert not (args.metrics_only and args.launch_only)
    result = dict(
        metadata=dict(
            global_tokens=args.global_tokens,
            tokens_per_source=args.global_tokens // args.world_size,
            metric_tokens=args.metric_tokens,
            calls=args.calls,
            rounds=args.rounds,
            scope=(
                "CPU weighted metrics use smallest-rank ties, not GPU reduction order; "
                "routing-only latency, not RDMA or whole-model throughput"
            ),
        )
    )
    if not args.launch_only:
        result["metrics"] = screen(args)
    if not args.metrics_only:
        result["launch"] = launch_benchmark(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
