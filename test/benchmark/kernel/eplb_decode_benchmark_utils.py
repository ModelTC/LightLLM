"""Decode 路由评测共用的合成流量、专家布局和 CUDA Graph 计时工具。"""

import numpy as np
import torch

from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    DecodeCooccurrenceEPLBPlanner,
    build_initial_local_expert_ids,
)


def token_random_values(source_rank, token_indices):
    """NumPy Philox4x32-10 参考：支持 tl.randint 的 uint32/uint64 offset 与 64 位种子。"""
    # uint64 保存乘积，再显式拆成高/低 32 位，不依赖 NumPy 的溢出行为。
    mask = np.uint64(0xFFFFFFFF)
    offsets = np.asarray(token_indices, dtype=np.uint64)
    c0, c1 = offsets & mask, offsets >> 32
    c2, c3 = np.zeros_like(c0), np.zeros_like(c0)
    k0, k1 = np.uint64(source_rank & 0xFFFFFFFF), np.uint64((source_rank >> 32) & 0xFFFFFFFF)
    for _ in range(10):
        product0 = c0 * np.uint64(0xD2511F53)
        product2 = c2 * np.uint64(0xCD9E8D57)
        c0, c1 = (product2 >> 32) ^ c1 ^ k0, product2 & mask
        c2, c3 = (product0 >> 32) ^ c3 ^ k1, product0 & mask
        k0 = (k0 + np.uint64(0x9E3779B9)) & mask
        k1 = (k1 + np.uint64(0xBB67AE85)) & mask
    return c0.astype(np.uint32)


def make_traffic(tokens, experts, topk, workload, seed):
    generator = torch.Generator().manual_seed(seed)
    if workload == "uniform":
        return torch.rand((tokens, experts), generator=generator).topk(k=topk, dim=1).indices.to(torch.int32)
    # 每个 token 从同一个 16-expert 群中选 top-k，群内专家跨 rank 分布。
    # 独立的 calibration/evaluation 种子，避免重复请求或训练样本泄漏。
    groups = torch.randperm(experts, generator=torch.Generator().manual_seed(123)).reshape(-1, 16)
    probabilities = 1 / torch.arange(1, groups.shape[0] + 1, dtype=torch.float64)
    group_ids = torch.multinomial(probabilities, tokens, replacement=True, generator=generator)
    slots = torch.rand((tokens, 16), generator=generator).topk(k=topk, dim=1).indices
    return torch.gather(input=groups[group_ids], dim=1, index=slots).to(torch.int32)


def make_placement(experts, world_size, redundant, workload, topk):
    placement = build_initial_local_expert_ids(experts, world_size, redundant)
    if workload == "uniform":
        generator = torch.Generator().manual_seed(42)
        primary_count = experts // world_size
        for rank in range(world_size):
            primary = placement[rank][:primary_count]
            candidates = [expert for expert in range(experts) if expert not in primary]
            indices = torch.randperm(len(candidates), generator=generator)[:redundant].tolist()
            placement[rank] = primary + [candidates[index] for index in indices]
        return placement
    if redundant == 0:
        return placement
    calibration = make_traffic(tokens=4096, experts=experts, topk=topk, workload=workload, seed=2025)
    selected = torch.nn.functional.one_hot(calibration.to(torch.long), num_classes=experts).sum(dim=1)
    cooccurrence = torch.triu(input=selected.T @ selected)
    route_statistics = torch.zeros((world_size, 1, experts, experts), dtype=torch.int64)
    route_statistics[0, 0] = cooccurrence
    planner = DecodeCooccurrenceEPLBPlanner(world_size=world_size, num_redundant_experts_per_rank=redundant)
    return planner.plan(route_statistics=route_statistics, current_placement=[placement])[0]


def capture(function, kwargs, calls):
    function(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            function(**kwargs)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def elapsed_us(graph, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / calls
