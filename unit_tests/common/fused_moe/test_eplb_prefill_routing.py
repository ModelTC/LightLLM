"""Prefill 统一入口的规模切换、逻辑负载采样与 CUDA Graph 布局更新测试。"""

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import prefill_eplb_topk_ids as routing
from lightllm.common.basemodel.triton_kernel.fused_moe.prefill_eplb_topk_ids_replica_hash import (
    eplb_prefill_replica_hash_topk_ids,
)
from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    build_initial_local_expert_ids,
    build_logical_to_physical_map,
    build_minimal_logical_to_physical_map,
)


MODES = ["current_gpu_first", "current_node_first", "global_first"]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _build_tables(world_size):
    experts = 64
    # 反转物理槽序，避免 logical ID 恰好等于 physical ID 而掩盖查表错误。
    placement = [
        list(reversed(row)) for row in build_initial_local_expert_ids(experts, world_size, 0 if world_size == 1 else 4)
    ]
    packed = build_logical_to_physical_map(
        rank_to_logic_expert_ids=placement,
        num_logical_experts=experts,
        current_rank=world_size - 1,
        node_world_size=max(1, world_size // 2),
    )
    minimal = build_minimal_logical_to_physical_map(rank_to_logic_expert_ids=placement, num_logical_experts=experts)
    return placement, torch.tensor(packed, dtype=torch.int32), torch.tensor(minimal, dtype=torch.int32)


@pytest.mark.parametrize("world_size", [1, 8, 16])
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("update_counter", [False, True])
def test_entrypoint_selects_path_and_forwards_parameters(monkeypatch, world_size, offset, mode, update_counter):
    tokens = 512 // world_size + offset
    logical = torch.zeros((tokens, 8), dtype=torch.int32)
    packed = torch.empty((64, 67), dtype=torch.int32)
    minimal = torch.arange(64, dtype=torch.int32)
    counter = torch.zeros((3, 64), dtype=torch.int64)
    sample_index = torch.zeros(2, dtype=torch.int64)
    calls = []
    expected = torch.empty_like(logical)

    def minimal_route(**kwargs):
        calls.append(("minimal", kwargs))
        return expected

    def hash_route(**kwargs):
        calls.append(("replica_hash", kwargs))
        return expected

    monkeypatch.setattr(routing, "eplb_prefill_minimal_topk_ids", minimal_route)
    monkeypatch.setattr(routing, "eplb_prefill_replica_hash_topk_ids", hash_route)
    output = routing.eplb_repair_prefill_topk_ids(
        logical_topk_ids=logical,
        logical_to_physical_map=packed,
        minimal_logical_to_physical_map=minimal,
        world_size=world_size,
        prefill_route_counter=counter,
        prefill_route_sample_index=sample_index,
        update_prefill_route_counter=update_counter,
        mode=mode,
    )
    assert output is expected
    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == ("minimal" if offset < 0 else "replica_hash")
    assert kwargs["logical_topk_ids"] is logical
    assert kwargs["prefill_route_counter"] is counter
    assert kwargs["prefill_route_sample_index"] is sample_index
    assert kwargs["update_prefill_route_counter"] is update_counter
    if offset < 0:
        assert kwargs["minimal_logical_to_physical_map"] is minimal
        assert "mode" not in kwargs and "logical_to_physical_map" not in kwargs
    else:
        assert kwargs["logical_to_physical_map"] is packed
        assert kwargs["mode"] == mode
        assert "minimal_logical_to_physical_map" not in kwargs


@requires_cuda
@pytest.mark.parametrize("world_size", [1, 8, 16])
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("counter_mode", ["absent", "disabled", "enabled"])
def test_physical_ids_and_logical_sampling_match_reference(world_size, offset, mode, dtype, counter_mode):
    placement, packed, minimal = _build_tables(world_size)
    packed, minimal = packed.cuda(), minimal.cuda()
    tokens = 512 // world_size + offset
    logical = (torch.arange(tokens * 8, dtype=dtype, device="cuda") % 64).reshape(tokens, 8)
    original_logical = logical.clone()
    counter = None if counter_mode == "absent" else torch.full((3, 64), 17, dtype=torch.int64, device="cuda")
    sample_index = None if counter is None else torch.zeros(2, dtype=torch.int64, device="cuda")
    output = routing.eplb_repair_prefill_topk_ids(
        logical_topk_ids=logical,
        logical_to_physical_map=packed,
        minimal_logical_to_physical_map=minimal,
        world_size=world_size,
        prefill_route_counter=counter,
        prefill_route_sample_index=sample_index,
        update_prefill_route_counter=counter_mode == "enabled",
        mode=mode,
    )
    if offset < 0:
        expected = minimal[logical.to(torch.long)].to(dtype)
    else:
        # 原始副本哈希的精确数值由 test_eplb.py 的独立参考实现覆盖。
        expected = eplb_prefill_replica_hash_topk_ids(
            logical_topk_ids=logical,
            logical_to_physical_map=packed,
            prefill_route_counter=None,
            prefill_route_sample_index=None,
            update_prefill_route_counter=False,
            mode=mode,
        )
    assert output.shape == logical.shape and output.dtype == logical.dtype
    assert torch.equal(output, expected)
    assert torch.equal(logical, original_logical)
    physical_to_logical = torch.tensor(placement, dtype=dtype, device="cuda").flatten()
    assert torch.equal(physical_to_logical[output.to(torch.long)], logical)
    if counter is not None:
        expected_counter = torch.full((3, 64), 17, dtype=torch.int64)
        if counter_mode == "enabled":
            expected_counter[0] = torch.bincount(logical.cpu().to(torch.long).flatten(), minlength=64)
        assert torch.equal(counter.cpu(), expected_counter)
        assert sample_index.tolist() == ([1, 0] if counter_mode == "enabled" else [0, 0])


@requires_cuda
@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_graph_replay_alternates_paths_wraps_samples_and_updates_layout(world_size, dtype):
    placement, packed, minimal = _build_tables(world_size)
    packed, minimal = packed.cuda(), minimal.cuda()
    counter = torch.zeros((3, 64), dtype=torch.int64, device="cuda")
    sample_index = torch.zeros(2, dtype=torch.int64, device="cuda")
    captured = []
    # W=1 时小批量路径也有多个 program，验证核内同步与同一缓冲区的路径切换。
    for tokens in (512 // world_size - 1, 512 // world_size):
        logical = (torch.arange(tokens * 8, dtype=dtype, device="cuda") % 64).reshape(tokens, 8)
        kwargs = dict(
            logical_topk_ids=logical,
            logical_to_physical_map=packed,
            minimal_logical_to_physical_map=minimal,
            world_size=world_size,
            prefill_route_counter=counter,
            prefill_route_sample_index=sample_index,
            update_prefill_route_counter=True,
            mode="global_first",
        )
        routing.eplb_repair_prefill_topk_ids(**kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = routing.eplb_repair_prefill_topk_ids(**kwargs)
        captured.append((graph, logical, output))
    counter.zero_()
    sample_index.zero_()
    addresses = (packed.data_ptr(), minimal.data_ptr())
    expected_counter = torch.zeros((3, 64), dtype=torch.int64)
    for step in range(11):
        if step == 5:
            # 原地更改槽位排布：Graph 仍引用原地址，但两条路由都应使用新内容。
            placement = [list(reversed(row)) for row in placement]
            packed.copy_(
                torch.tensor(
                    build_logical_to_physical_map(
                        rank_to_logic_expert_ids=placement,
                        num_logical_experts=64,
                        current_rank=world_size - 1,
                        node_world_size=max(1, world_size // 2),
                    ),
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            minimal.copy_(
                torch.tensor(
                    build_minimal_logical_to_physical_map(
                        rank_to_logic_expert_ids=placement,
                        num_logical_experts=64,
                    ),
                    dtype=torch.int32,
                    device="cuda",
                )
            )
        graph, logical, output = captured[step % 2]
        graph.replay()
        expected_counter[step % 3] = torch.bincount(logical.cpu().to(torch.long).flatten(), minlength=64)
        assert sample_index.tolist() == [step + 1, 0]
        assert torch.equal(counter.cpu(), expected_counter)
        if step % 2 == 0:
            expected = minimal[logical.to(torch.long)].to(dtype)
        else:
            expected = eplb_prefill_replica_hash_topk_ids(
                logical_topk_ids=logical,
                logical_to_physical_map=packed,
                prefill_route_counter=None,
                prefill_route_sample_index=None,
                update_prefill_route_counter=False,
                mode="global_first",
            )
        assert torch.equal(output, expected)
        assert addresses == (packed.data_ptr(), minimal.data_ptr())


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
def test_empty_input_preserves_prefill_sampling_state(mode):
    _, packed, minimal = _build_tables(8)
    counter = torch.zeros((3, 64), dtype=torch.int64, device="cuda")
    sample_index = torch.zeros(2, dtype=torch.int64, device="cuda")
    output = routing.eplb_repair_prefill_topk_ids(
        logical_topk_ids=torch.empty((0, 8), dtype=torch.int32, device="cuda"),
        logical_to_physical_map=packed.cuda(),
        minimal_logical_to_physical_map=minimal.cuda(),
        world_size=8,
        prefill_route_counter=counter,
        prefill_route_sample_index=sample_index,
        update_prefill_route_counter=True,
        mode=mode,
    )
    assert output.shape == (0, 8)
    assert sample_index.tolist() == [0, 0]
    assert torch.count_nonzero(counter).item() == 0


@pytest.mark.parametrize("tokens", [0, 1, 32])
def test_invalid_mode_is_rejected_on_both_paths(tokens):
    _, packed, minimal = _build_tables(8)
    with pytest.raises(AssertionError, match="unsupported EPLB dispatch mode"):
        routing.eplb_repair_prefill_topk_ids(
            logical_topk_ids=torch.zeros((tokens, 8), dtype=torch.int32),
            logical_to_physical_map=packed,
            minimal_logical_to_physical_map=minimal,
            world_size=8,
            prefill_route_counter=None,
            prefill_route_sample_index=None,
            update_prefill_route_counter=False,
            mode="invalid",
        )
