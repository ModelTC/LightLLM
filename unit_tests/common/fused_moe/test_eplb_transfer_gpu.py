"""Complete-layout EPLB migration across two, three and eight CUDA GPUs."""
import math
import os
import socket
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightllm.common.eplb_utils import get_eplb_staging_nbytes, get_eplb_staging_tensor_shape
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    NixlEPLBTransfer,
    align_target_placement,
    build_transfer_plan,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_redundant_expert_ids,
    expand_redundant_placement,
    build_logical_to_physical_maps_for_layers,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
    EPLBState,
    ExpertParallelState,
)

pytest.importorskip("nixl", reason="NIXL package is required")


class _Pack:
    def __init__(self, weight, weight_scale):
        self.weight = weight
        self.weight_scale = weight_scale
        self.weight_zero_point = None


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_ready_prefix(transfer, control_group):
    deadline = time.monotonic() + 30
    while True:
        pending = transfer.pending_layers()
        ready_count = torch.tensor([len(pending)], dtype=torch.int32)
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=control_group)
        if int(ready_count.item()) > 0:
            return pending[: int(ready_count.item())]
        if time.monotonic() >= deadline:
            raise TimeoutError("EPLB transfer worker did not publish a globally ready layer")
        time.sleep(0.001)


def _run_layers(
    transfer,
    control_group,
    layer_plans,
    callback=lambda layer_index: None,
    before_commit_callback=lambda layer_index: None,
):
    transfer.start(layer_plans, transfer.prepare_transfer(layer_plans))
    committed = 0
    while committed < len(layer_plans):
        pending = _wait_for_ready_prefix(transfer, control_group)
        assert len(pending) <= len(layer_plans) - committed
        for layer_index, buffer_index in pending:
            assert layer_index == layer_plans[committed][0]
            before_commit_callback(layer_index)
            transfer.commit(
                layer_index,
                buffer_index,
                lambda layer_index=layer_index: callback(layer_index),
            )
            committed += 1
    transfer.finish()


def _full_layout_worker(rank, port, world_size, redundant=0, inject_failure=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    control_group = dist.new_group(list(range(world_size)), backend="gloo")
    transfer_group = dist.new_group(list(range(world_size)), backend="gloo")
    experts = world_size * 2
    initial_extras = build_initial_redundant_expert_ids(experts, world_size, redundant)
    current = expand_redundant_placement(initial_extras, experts)
    capacity = current.shape[1]
    layers = 9 if world_size == 8 else 3
    weights = []
    for layer in range(layers):
        state = EPLBState(
            num_redundant_experts_per_rank=redundant,
            logical_to_physical_map=torch.empty((experts, world_size), device="cuda", dtype=torch.int32),
            logical_replica_count=torch.ones(experts, device="cuda", dtype=torch.int32),
            route_counter=torch.zeros((1, experts), device="cuda", dtype=torch.int64),
            physical_to_logical=current.clone(),
        )
        values = (current[rank].cuda() + 10 * layer).view(-1, 1)
        cols13 = torch.arange(17, device="cuda").view(1, -1)
        cols2 = torch.arange(9, device="cuda").view(1, -1)
        weights.append(
            SimpleNamespace(
                expert_parallel_state=ExpertParallelState(experts, world_size, state),
                w13=_Pack((values * 17 + cols13).to(torch.uint8), (values + 0.5).float()),
                w2=_Pack((values * 9 + cols2).to(torch.float16), (values + 0.25 + cols2[:, :3]).float()),
            )
        )
    for layer, weight in enumerate(weights):
        values = (current[rank].cuda() + 10 * layer).view(-1, 1)
        weight.w13.weight_zero_point = (values + torch.arange(5, device="cuda")).to(torch.int32)
    transfer = NixlEPLBTransfer(weights, transfer_group, rank, world_size)
    original_copy_part = transfer._copy_part
    delayed = [False]

    def delayed_copy_part(prepared):
        if rank == 0 and not delayed[0] and prepared.part_index == 0:
            delayed[0] = True
            time.sleep(0.05)
        return original_copy_part(prepared)

    transfer._copy_part = delayed_copy_part
    assert transfer.staging_depth == 1
    assert all(tensor.shape[0] == capacity for _, tensor in transfer.staging[0])
    actual = sum(tensor.nbytes for staging in transfer.staging for _, tensor in staging)
    expected = sum(capacity * ((tensor[0].numel() + 1) // 2) * tensor.element_size() for _, tensor in transfer.live[0])
    shared_expected = sum(
        math.prod(get_eplb_staging_tensor_shape(tensor, capacity)) * tensor.element_size()
        for _, tensor in transfer.live[0]
    )
    assert actual == expected == shared_expected == get_eplb_staging_nbytes(weights)

    def assert_values(layer, placement):
        values = (placement[rank].cuda() + 10 * layer).view(-1, 1)
        weight = weights[layer]
        torch.testing.assert_close(
            weight.w13.weight,
            (values * 17 + torch.arange(17, device="cuda")).to(torch.uint8),
        )
        torch.testing.assert_close(weight.w13.weight_scale, (values + 0.5).float())
        torch.testing.assert_close(
            weight.w13.weight_zero_point, (values + torch.arange(5, device="cuda")).to(torch.int32)
        )
        torch.testing.assert_close(
            weight.w2.weight,
            (values * 9 + torch.arange(9, device="cuda")).to(torch.float16),
        )
        torch.testing.assert_close(
            weight.w2.weight_scale,
            (values + 0.25 + torch.arange(3, device="cuda")).float(),
        )

    targets = [current.roll(1, dims=0)]
    second = targets[0].clone()
    second[[0, 1]] = second[[1, 0]]
    targets.append(second)
    for generation, target in enumerate(targets, start=1):
        target = align_target_placement(current, target)
        plan = build_transfer_plan(current, target, experts, world_size, world_size)
        if world_size == 3 and generation == 2:
            assert all(step.dst_rank != 2 for step in plan)
        maps, counts = build_logical_to_physical_maps_for_layers(
            target[None], experts, source_rank=rank, node_world_size=world_size
        )

        snapshots = {}

        def assert_old(layer):
            assert_values(layer, current)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1000000)
                snapshots[layer] = tuple(
                    t.clone()
                    for pack in (weights[layer].w13, weights[layer].w2)
                    for t in (pack.weight, pack.weight_scale, pack.weight_zero_point)
                    if t is not None
                )
            torch.cuda.current_stream().wait_stream(stream)

        def publish(layer):
            assert_values(layer, target)
            expected_old = []
            values = (current[rank].cuda() + 10 * layer).view(-1, 1)
            expected_old.extend(
                [
                    (values * 17 + torch.arange(17, device="cuda")).to(torch.uint8),
                    (values + 0.5).float(),
                    (values + torch.arange(5, device="cuda")).to(torch.int32),
                    (values * 9 + torch.arange(9, device="cuda")).to(torch.float16),
                    (values + 0.25 + torch.arange(3, device="cuda")).float(),
                ]
            )
            for snapshot, expected in zip(snapshots[layer], expected_old):
                torch.testing.assert_close(snapshot, expected)
            state = weights[layer].expert_parallel_state.eplb
            state.logical_to_physical_map.copy_(maps[0])
            state.logical_replica_count.copy_(counts[0])
            state.physical_to_logical = target.clone()
            state.placement_generation += 1

        _run_layers(transfer, control_group, [(layer, plan) for layer in reversed(range(layers))], publish, assert_old)
        torch.cuda.synchronize()
        for layer, weight in enumerate(weights):
            assert_values(layer, target)
            state = weight.expert_parallel_state.eplb
            assert state.placement_generation == generation
            for expert in range(experts):
                physical = state.logical_to_physical_map[expert, 0].item()
                dst, row = divmod(physical, capacity)
                assert target[dst, row] == expert
        current = target
        dist.barrier(group=control_group)
    if inject_failure:
        target = current.roll(1, dims=0)
        plan = build_transfer_plan(current, target, experts, world_size, world_size)
        original_copy = transfer._copy_part

        def fail_second(prepared):
            if rank == 0 and prepared.part_index == 1:
                raise RuntimeError("injected second part failure")
            return original_copy(prepared)

        transfer._copy_part = fail_second
        plans = [(layer, plan) for layer in (0, 1, 2)]
        transfer.start(plans, transfer.prepare_transfer(plans))
        layer, buffer_index = _wait_for_ready_prefix(transfer, control_group)[0]
        with pytest.raises(RuntimeError, match="full EPLB migration phase failed"):
            transfer.commit(layer, buffer_index, lambda: pytest.fail("must not publish"))
        assert all(release.is_set() for release in transfer._release)
        transfer._thread.join(timeout=5)
        assert not transfer._thread.is_alive()
        with pytest.raises(RuntimeError, match="migration worker failed"):
            transfer.finish()
    transfer.shutdown()
    dist.barrier(group=control_group)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 8])
@pytest.mark.parametrize("redundant", [0, 1])
def test_full_layout_gpu_overwrite_cycles_and_staging_reuse(world_size, redundant):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA GPUs")
    mp.spawn(_full_layout_worker, args=(_free_port(), world_size, redundant), nprocs=world_size, join=True)


def test_full_layout_gpu_second_part_failure_propagates():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    mp.spawn(_full_layout_worker, args=(_free_port(), 2, 0, True), nprocs=2, join=True)
