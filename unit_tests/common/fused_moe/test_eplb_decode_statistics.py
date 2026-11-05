"""共现统计的设备分支开关：不重新捕获 Graph，也不影响专家映射结果。"""

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
    eplb_hash_targets_topk_ids,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_rank_weighted import (
    eplb_rank_weighted_topk_ids,
)
from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    build_initial_local_expert_ids,
    build_logical_to_physical_map,
    build_logical_to_rank_physical_map,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for decode statistics")


@pytest.mark.parametrize("routing_function", [eplb_hash_targets_topk_ids, eplb_rank_weighted_topk_ids])
@pytest.mark.parametrize("tokens", [1, 33, 129])
@pytest.mark.parametrize("topk", [1, 6, 8, 32])
@pytest.mark.parametrize("initial_enabled", [0, 1])
def test_statistics_flag_toggles_inside_same_graph(routing_function, tokens, topk, initial_enabled):
    experts, world_size, source_rank, redundant = 64, 4, 1, 4
    placement = build_initial_local_expert_ids(experts, world_size, redundant)
    rows = build_logical_to_physical_map(placement, experts, current_rank=source_rank, node_world_size=2)
    logical_to_rank_physical_map = build_logical_to_rank_physical_map(placement, experts)
    # 每个 token 内 ID 唯一，且反复覆盖最小/最大 ID；33/129 行覆盖末尾 program 的 mask。
    logical = (
        torch.arange(tokens * topk, dtype=torch.int32, device="cuda").reshape(tokens, topk) % experts
    ).contiguous()
    counter = torch.zeros((experts, experts), dtype=torch.int64, device="cuda")
    enabled = torch.full((1,), initial_enabled, dtype=torch.int32, device="cuda")
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=torch.tensor(rows, dtype=torch.int32, device="cuda"),
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map, dtype=torch.int32, device="cuda"),
        current_rank=source_rank,
        num_physical_experts_per_rank=len(placement[0]),
        node_world_size=2,
    )
    if routing_function is eplb_rank_weighted_topk_ids:
        kwargs.pop("logical_to_physical_map")
        kwargs.pop("num_physical_experts_per_rank")
    addresses = (counter.data_ptr(), enabled.data_ptr(), logical.data_ptr())
    routing_function(**kwargs, route_counter=counter, update_route_counter=True, route_recording_enabled=enabled)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = routing_function(
            **kwargs, route_counter=counter, update_route_counter=True, route_recording_enabled=enabled
        )

    # 从非零统计开始，关闭必须保持原值，不能把“不更新”误实现为清零。
    counter.fill_(7)
    expected_counter = torch.full((experts, experts), 7, dtype=torch.int64)
    for iteration, flag in enumerate((0, 1, 0, 1, 1, 0)):
        # 每次重放更新同一输入地址，验证分支及计数不是捕获阶段的固定结果。
        updated = (torch.arange(tokens * topk, dtype=torch.int32).reshape(tokens, topk) + iteration) % experts
        logical.copy_(updated)
        enabled.fill_(flag)
        graph.replay()
        reference_output = routing_function(**kwargs, update_route_counter=False)
        assert torch.equal(output, reference_output)
        assert torch.equal(logical.cpu(), updated)
        if flag:
            selected = torch.nn.functional.one_hot(updated.long(), num_classes=experts).sum(dim=1)
            expected_counter += torch.triu(selected.T @ selected)
        assert torch.equal(counter.cpu(), expected_counter)
        assert (counter.data_ptr(), enabled.data_ptr(), logical.data_ptr()) == addresses
