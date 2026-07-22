import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.force_balanced_routing import (
    force_balanced_routing,
)
from lightllm.utils import envs_utils


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for force-balanced routing")


def _sentinel_ids(token_num: int, top_k: int, expert_num: int):
    return torch.full((token_num, top_k), expert_num + 17, dtype=torch.int64, device="cuda")


@CUDA_REQUIRED
def test_force_balanced_routing_ratio_zero_preserves_ids():
    topk_ids = _sentinel_ids(token_num=3, top_k=8, expert_num=128)
    original = topk_ids.clone()

    assert force_balanced_routing(topk_ids, 0.0, expert_num=128, global_rank=0, world_size=8) is None

    assert torch.equal(topk_ids, original)


@CUDA_REQUIRED
@pytest.mark.parametrize("expert_num", [128, 256])
def test_force_balanced_routing_full_rows_are_globally_rank_balanced(expert_num):
    world_size = 8
    top_k = 8
    rows = []
    for global_rank in range(world_size):
        topk_ids = _sentinel_ids(token_num=1, top_k=top_k, expert_num=expert_num)
        force_balanced_routing(
            topk_ids,
            1.0,
            expert_num=expert_num,
            global_rank=global_rank,
            world_size=world_size,
        )
        rows.append(topk_ids)

    assigned_ids = torch.cat(rows).flatten()
    destination_ranks = assigned_ids // (expert_num // world_size)
    assert torch.equal(
        torch.bincount(destination_ranks, minlength=world_size),
        torch.full((world_size,), 8, device="cuda"),
    )
    assert assigned_ids.min() >= 0
    assert assigned_ids.max() < expert_num
    assert all(torch.unique(row).numel() == top_k for row in rows)


@CUDA_REQUIRED
def test_force_balanced_routing_fractional_rows_are_globally_balanced_with_unequal_local_token_counts():
    expert_num = 128
    world_size = 8
    top_k = 8
    rows = []
    changed_rows = 0
    for global_rank in range(world_size):
        topk_ids = _sentinel_ids(token_num=global_rank + 1, top_k=top_k, expert_num=expert_num)
        original = topk_ids.clone()
        force_balanced_routing(
            topk_ids,
            0.5,
            expert_num=expert_num,
            global_rank=global_rank,
            world_size=world_size,
        )
        changed = torch.any(topk_ids != original, dim=1)
        changed_rows += int(changed.sum().item())
        rows.append(topk_ids[changed])

    assigned_ids = torch.cat(rows).flatten()
    destination_ranks = assigned_ids // (expert_num // world_size)
    assert changed_rows > 0
    assert torch.equal(
        torch.bincount(destination_ranks, minlength=world_size),
        torch.full((world_size,), changed_rows, device="cuda"),
    )


@CUDA_REQUIRED
def test_force_balanced_routing_supports_cuda_graph_replay():
    source_ids = _sentinel_ids(token_num=1, top_k=8, expert_num=128)
    graph_ids = torch.empty_like(source_ids)
    force_balanced_routing(graph_ids, 1.0, expert_num=128, global_rank=0, world_size=8)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ids.copy_(source_ids)
        force_balanced_routing(graph_ids, 1.0, expert_num=128, global_rank=0, world_size=8)

    graph.replay()
    torch.cuda.synchronize()
    first_replay = graph_ids.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_ids, first_replay)
    assert torch.equal(
        torch.bincount(graph_ids.flatten() // 16, minlength=8),
        torch.ones(8, device="cuda"),
    )


@CUDA_REQUIRED
def test_force_balanced_routing_rejects_invalid_ratio():
    topk_ids = torch.empty((1, 1), dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError):
        force_balanced_routing(topk_ids, 1.1, expert_num=8, global_rank=0, world_size=8)


@CUDA_REQUIRED
def test_force_balanced_routing_rejects_top_k_not_equal_to_world_size():
    topk_ids = torch.empty((1, 2), dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="top_k == world_size"):
        force_balanced_routing(topk_ids, 1.0, expert_num=8, global_rank=0, world_size=8)


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_force_balanced_routing_env_rejects_invalid_ratio(monkeypatch, value):
    monkeypatch.setenv("LIGHTLLM_EP_FORCE_BALANCED_PREFILL_ROUTING_RATIO", value)
    envs_utils.get_force_balanced_prefill_routing_ratio.cache_clear()
    with pytest.raises(ValueError):
        envs_utils.get_force_balanced_prefill_routing_ratio()
    envs_utils.get_force_balanced_prefill_routing_ratio.cache_clear()
