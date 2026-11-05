"""小批量最小映射的 physical ID 正确性、dtype、补齐和共现采样测试。"""

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_minimal import eplb_minimal_topk_ids
from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    build_initial_local_expert_ids,
    build_minimal_logical_to_physical_map,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size,redundant", [(2, 0), (8, 1), (16, 4)])
@pytest.mark.parametrize("tokens,topk", [(1, 1), (3, 3), (31, 8), (33, 33)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("update_counter,enabled", [(False, 0), (True, 0), (True, 1)])
def test_minimal_mapping_matches_reference_and_preserves_logical_statistics(
    world_size, redundant, tokens, topk, dtype, update_counter, enabled
):
    experts = 64
    # 反转物理槽序，避免 logical ID 恰好等于 physical ID 而掩盖查表错误。
    placement = [list(reversed(row)) for row in build_initial_local_expert_ids(experts, world_size, redundant)]
    minimal_map = torch.tensor(
        build_minimal_logical_to_physical_map(placement, experts), dtype=torch.int32, device="cuda"
    )
    logical = (torch.arange(tokens * topk, dtype=dtype, device="cuda") % experts).reshape(tokens, topk)
    original_logical = logical.clone()
    counter = torch.zeros((experts, experts), dtype=torch.int64, device="cuda")
    recording_enabled = torch.tensor([enabled], dtype=torch.int32, device="cuda")
    output = eplb_minimal_topk_ids(
        logical_topk_ids=logical,
        minimal_logical_to_physical_map=minimal_map,
        route_counter=counter,
        update_route_counter=update_counter,
        route_recording_enabled=recording_enabled,
    )
    assert output.dtype == dtype and output.shape == logical.shape
    assert torch.equal(output, minimal_map[logical.to(torch.long)].to(dtype))
    assert torch.equal(logical, original_logical)
    physical_to_logical = torch.tensor(placement, dtype=dtype, device="cuda").flatten()
    assert torch.equal(physical_to_logical[output.to(torch.long)], logical)
    expected_counter = torch.zeros_like(counter, device="cpu")
    if update_counter and enabled:
        selected = torch.nn.functional.one_hot(logical.cpu().to(torch.long), num_classes=experts).sum(dim=1)
        expected_counter = torch.triu(selected.T @ selected)
    assert torch.equal(counter.cpu(), expected_counter)


@pytest.mark.parametrize(
    "invalid_map",
    [
        torch.zeros((8, 1), dtype=torch.int32),
        torch.zeros(8, dtype=torch.int64),
        torch.zeros(16, dtype=torch.int32)[::2],
    ],
)
def test_minimal_mapping_rejects_invalid_table_before_kernel_launch(invalid_map):
    with pytest.raises(AssertionError):
        eplb_minimal_topk_ids(
            logical_topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            minimal_logical_to_physical_map=invalid_map,
        )
