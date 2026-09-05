"""NIXL EPLB correctness tests and a two-GPU 512 MiB micro-performance test."""
import os
import random
import socket
import statistics
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    NixlEPLBTransfer,
    align_target_placement,
    build_transfer_plan,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_redundant_expert_ids,
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


class _FakeWeight:
    def __init__(self, rank, layer_index, row_elements):
        self.n_routed_experts = 32
        self.expert_parallel_state = ExpertParallelState(
            num_logical_experts=32,
            world_size=2,
            eplb=EPLBState(
                num_redundant_experts_per_rank=16,
                initial_redundant_expert_ids_by_rank=build_initial_redundant_expert_ids(32, 2, 16),
                logical_to_physical_map=torch.zeros((32, 2), dtype=torch.int32, device="cuda"),
                logical_replica_count=torch.ones(32, dtype=torch.int32, device="cuda"),
                route_counter=torch.zeros((1, 32), dtype=torch.int64, device="cuda"),
            ),
        )
        base = rank * 100 + layer_index * 100
        self.w13 = self._pack(base, row_elements)
        self.w2 = self._pack(base + 10, row_elements)

    @staticmethod
    def _pack(base, row_elements):
        weight = torch.empty((32, row_elements), dtype=torch.float16, device="cuda")
        for row in range(weight.shape[0]):
            weight[row].fill_(base + row)
        scale = torch.empty((32, 1), dtype=torch.float32, device="cuda")
        for row in range(scale.shape[0]):
            scale[row].fill_(base + row + 0.5)
        return _Pack(weight, scale)


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
    transfer.start(layer_plans)
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


def _assert_correctness(weights, rank, source_rows_by_dst_slot):
    if rank == 0:
        for layer_index in (0, len(weights) - 1):
            base = 100 + layer_index * 100
            for dst_slot, src_row in enumerate(source_rows_by_dst_slot):
                dst_row = 16 + dst_slot
                assert torch.all(weights[layer_index].w13.weight[dst_row] == base + src_row)
                assert torch.all(weights[layer_index].w13.weight_scale[dst_row] == base + src_row + 0.5)
                assert torch.all(weights[layer_index].w2.weight[dst_row] == base + src_row + 10)
                assert torch.all(weights[layer_index].w2.weight_scale[dst_row] == base + src_row + 10.5)


def _benchmark(transfer, control_group, layer_plans, payload):
    for _ in range(3):
        _run_layers(transfer, control_group, layer_plans)
    dist.barrier(group=control_group)
    started = time.perf_counter()
    for _ in range(8):
        _run_layers(transfer, control_group, layer_plans)
    torch.cuda.synchronize()
    return payload * 8 / (time.perf_counter() - started) / 1e9


def _benchmark_nixl_copy_batch(transfer, control_group, layer_plans, payload):
    batch = [
        (layer_index, plan, buffer_index, transfer.staging[buffer_index])
        for buffer_index, (layer_index, plan) in enumerate(layer_plans)
    ]
    # Measure the precompiled hot path only; planning and descriptor construction are off the timer.
    prepared_batch = transfer._prepare_batch(batch)
    for _ in range(3):
        transfer._copy_batch(batch, prepared_batch)
    dist.barrier(group=control_group)
    samples = []
    for _ in range(20):
        started = time.perf_counter()
        transfer._copy_batch(batch, prepared_batch)
        samples.append(payload / (time.perf_counter() - started) / 1e9)
    torch.cuda.synchronize()
    median = statistics.median(samples)
    print(
        f"NIXL _copy_batch payload={payload / 2**20:.1f} MiB; "
        f"min={min(samples):.2f} GB/s median={median:.2f} "
        f"mean={statistics.mean(samples):.2f} max={max(samples):.2f}",
        flush=True,
    )
    return median


def _eplb_worker(rank, port, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    control_group = dist.new_group([0, 1], backend="gloo")
    transfer_group = dist.new_group([0, 1], backend="gloo")
    # Eight layers × 16 changed experts × two 2 MiB rows = 512 MiB useful remote weight payload.
    row_elements = int(os.getenv("LIGHTLLM_EPLB_TEST_ROW_ELEMENTS", str(1024 * 1024)))
    current = torch.tensor([list(range(16)), list(range(16))])
    source_rows_by_dst_slot = list(range(16))
    random.Random(20260731).shuffle(source_rows_by_dst_slot)
    # Rank 0 receives the reverse logical-expert range in a deterministic random slot order.
    # Consequently every descriptor has a distinct source and destination row.
    target = torch.tensor([[16 + source_row for source_row in source_rows_by_dst_slot], list(range(16))])
    plan = build_transfer_plan(current, target, 32, 2, 2)
    assert [step.src_local_row for step in plan if step.dst_rank == 0] == source_rows_by_dst_slot
    benchmark_layer_count = 8
    layer_count = benchmark_layer_count + 1
    row_payload = (
        2 * row_elements * torch.empty((), dtype=torch.float16).element_size()
        + 2 * torch.empty((), dtype=torch.float32).element_size()
    )
    payload = benchmark_layer_count * 16 * row_payload
    weights = [_FakeWeight(rank, layer_index, row_elements) for layer_index in range(layer_count)]
    transfer = NixlEPLBTransfer(weights, transfer_group, rank, world_size=2)
    assert transfer.staging_depth == 8
    assert transfer._eplb_states[0] is weights[0].expert_parallel_state.eplb
    assert all(tensor.is_cuda for staging in transfer.staging for _, tensor in staging)

    wrap_layer_plans = [(layer_index, plan) for layer_index in range(layer_count)]

    def delay_rank_zero_first_commit(layer_index):
        if rank == 0 and layer_index == 0:
            time.sleep(0.1)

    _run_layers(
        transfer,
        control_group,
        wrap_layer_plans,
        before_commit_callback=delay_rank_zero_first_commit,
    )
    torch.cuda.synchronize()
    _assert_correctness(weights, rank, source_rows_by_dst_slot)
    layer_plans = wrap_layer_plans[:benchmark_layer_count]
    nixl_copy_batch = _benchmark_nixl_copy_batch(transfer, control_group, layer_plans, payload)
    nixl_bandwidth = _benchmark(transfer, control_group, layer_plans, payload)
    transfer.shutdown()

    gathered = [None, None]
    dist.all_gather_object(gathered, (nixl_bandwidth, nixl_copy_batch), group=control_group)
    if rank == 0:
        queue.put((payload, *gathered[0]))
    dist.barrier(group=control_group)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs",
)
def test_eplb_transfer_two_gpu_correctness_and_microperf():
    queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(_eplb_worker, args=(_free_port(), queue), nprocs=2, join=True)
    payload, nixl_gbps, nixl_copy_batch_gbps = queue.get()
    print(
        f"EPLB remote payload/round: {payload / 2**20:.1f} MiB; "
        f"NIXL={nixl_gbps:.2f} GB/s NIXL _copy_batch={nixl_copy_batch_gbps:.2f} GB/s"
    )
    assert nixl_gbps > 0


def _depth_value(expert, layer_index, offset):
    return expert // 32 * 100 + layer_index * 5 + expert % 32 + offset


class _DepthWeight:
    def __init__(self, rank, layer_index, initial_placement):
        self.n_routed_experts = 256
        self.expert_parallel_state = ExpertParallelState(
            num_logical_experts=256,
            world_size=8,
            eplb=EPLBState(
                num_redundant_experts_per_rank=4,
                initial_redundant_expert_ids_by_rank=initial_placement.clone(),
                logical_to_physical_map=torch.zeros((256, 8), dtype=torch.int32, device="cuda"),
                logical_replica_count=torch.ones(256, dtype=torch.int32, device="cuda"),
                route_counter=torch.zeros((1, 256), dtype=torch.int64, device="cuda"),
            ),
        )
        logical_ids = list(range(rank * 32, (rank + 1) * 32)) + initial_placement[rank].tolist()
        self.w13 = self._pack(logical_ids, layer_index, 0)
        self.w2 = self._pack(logical_ids, layer_index, 2)

    @staticmethod
    def _pack(logical_ids, layer_index, offset):
        weight = torch.empty((36, 64), dtype=torch.float16, device="cuda")
        scale = torch.empty((36, 1), dtype=torch.float32, device="cuda")
        for row, expert in enumerate(logical_ids):
            value = _depth_value(expert, layer_index, offset)
            weight[row].fill_(value)
            scale[row].fill_(value + 0.25)
        return _Pack(weight, scale)


def _depth_target(layer_index):
    return torch.tensor([[((dst + layer_index + slot + 1) % 8) * 32 + slot for slot in range(4)] for dst in range(8)])


def _wait_all_pending(transfer, group, expected_count):
    deadline = time.monotonic() + 30
    while True:
        pending = transfer.pending_layers()
        ready_count = torch.tensor([len(pending)], dtype=torch.int32)
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=group)
        if int(ready_count.item()) == expected_count:
            return pending
        if time.monotonic() > deadline:
            raise TimeoutError(f"expected {expected_count} pending layers, got {pending}")
        time.sleep(0.001)


def _clone_depth_live(weights):
    return [
        [tensor.detach().clone() for _, tensor in transfer_tensors]
        for transfer_tensors in [
            [
                ("w13.weight", weight.w13.weight),
                ("w13.scale", weight.w13.weight_scale),
                ("w2.weight", weight.w2.weight),
                ("w2.scale", weight.w2.weight_scale),
            ]
            for weight in weights
        ]
    ]


def _assert_depth_snapshot(weights, snapshot, layer_indices=None, primary_only=False):
    if layer_indices is None:
        layer_indices = range(len(weights))
    for layer_index in layer_indices:
        live_tensors = (
            weights[layer_index].w13.weight,
            weights[layer_index].w13.weight_scale,
            weights[layer_index].w2.weight,
            weights[layer_index].w2.weight_scale,
        )
        for live, expected in zip(live_tensors, snapshot[layer_index]):
            if primary_only:
                live = live[:32]
                expected = expected[:32]
            torch.testing.assert_close(live, expected)


def _assert_depth_staging(rank, layer_plans, target_placements, pending, transfer):
    for buffer_index, ((layer_index, plan), pending_item) in enumerate(zip(layer_plans, pending)):
        assert pending_item == (layer_index, buffer_index)
        expected = {step.dst_slot: step for step in plan if step.dst_rank == rank}
        for dst_slot, step in expected.items():
            expert = int(target_placements[layer_index][rank, dst_slot])
            base = _depth_value(expert, layer_index, 0)
            staging = transfer.staging[buffer_index]
            assert torch.all(staging[0][1][dst_slot] == base)
            assert torch.all(staging[1][1][dst_slot] == base + 0.25)
            assert torch.all(staging[2][1][dst_slot] == base + 2)
            assert torch.all(staging[3][1][dst_slot] == base + 2.25)


def _assert_depth_live(weights, rank, layer_plans, target_placements):
    for layer_index, plan in layer_plans:
        for step in plan:
            if step.dst_rank != rank:
                continue
            expert = int(target_placements[layer_index][rank, step.dst_slot])
            base = _depth_value(expert, layer_index, 0)
            assert torch.all(weights[layer_index].w13.weight[32 + step.dst_slot] == base)
            assert torch.all(weights[layer_index].w13.weight_scale[32 + step.dst_slot] == base + 0.25)
            assert torch.all(weights[layer_index].w2.weight[32 + step.dst_slot] == base + 2)
            assert torch.all(weights[layer_index].w2.weight_scale[32 + step.dst_slot] == base + 2.25)


def _assert_peer_coverage(layer_plans, require_redundant_source=False):
    steps = [step for _, plan in layer_plans for step in plan]
    assert {step.dst_rank for step in steps} == set(range(8))
    assert {step.src_rank for step in steps} == set(range(8))
    for source_rank in range(8):
        assert len({step.dst_rank for step in steps if step.src_rank == source_rank}) > 1
    if require_redundant_source:
        assert any(step.src_local_row >= 32 for step in steps)


def _depth_worker(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=8)
    control_group = dist.new_group(list(range(8)), backend="gloo")
    transfer_group = dist.new_group(list(range(8)), backend="gloo")
    initial_placement = build_initial_redundant_expert_ids(256, 8, 4)
    weights = [_DepthWeight(rank, layer_index, initial_placement) for layer_index in range(9)]
    transfer = NixlEPLBTransfer(weights, transfer_group, rank, world_size=8)
    assert transfer.staging_depth == 8
    staging_bytes = sum(tensor.nbytes for staging in transfer.staging for _, tensor in staging)
    one_layer_staging_bytes = sum(tensor[:4].nbytes for _, tensor in transfer.live[0])
    assert staging_bytes == 8 * one_layer_staging_bytes
    current = initial_placement
    first_order = [8, 0, 5, 1, 7, 3, 6, 2, 4]
    first_targets = {
        layer_index: align_target_placement(current, _depth_target(layer_index)) for layer_index in range(9)
    }
    first_plans = [
        (layer_index, build_transfer_plan(current, first_targets[layer_index], 256, 8, 8))
        for layer_index in first_order
    ]
    _assert_peer_coverage(first_plans)
    first_snapshot = _clone_depth_live(weights)
    transfer.start(first_plans, transfer.prepare_transfer(first_plans))
    committed = 0
    pending = _wait_all_pending(transfer, control_group, 8)
    _assert_depth_staging(rank, first_plans[:8], first_targets, pending, transfer)
    _assert_depth_snapshot(weights, first_snapshot)
    dist.barrier(group=control_group)
    if rank == 0:
        time.sleep(0.1)
    for layer_index, buffer_index in pending:
        _assert_depth_snapshot(weights, first_snapshot, [layer for layer, _ in first_plans[committed:]])
        transfer.commit(layer_index, buffer_index)
        committed += 1
    pending = _wait_all_pending(transfer, control_group, 1)
    _assert_depth_staging(rank, first_plans[8:], first_targets, pending, transfer)
    _assert_depth_snapshot(weights, first_snapshot, [first_plans[8][0]])
    dist.barrier(group=control_group)
    if rank == 0:
        time.sleep(0.1)
    for layer_index, buffer_index in pending:
        _assert_depth_snapshot(weights, first_snapshot, [layer for layer, _ in first_plans[committed:]])
        transfer.commit(layer_index, buffer_index)
        committed += 1
    transfer.finish()
    torch.cuda.synchronize()
    dist.barrier(group=control_group)
    _assert_depth_live(weights, rank, first_plans, first_targets)

    second_order = [7, 2, 4]
    second_targets = {
        layer_index: align_target_placement(
            first_targets[layer_index],
            torch.tensor([[((dst + layer_index + slot + 3) % 8) * 32 + slot for slot in range(4)] for dst in range(8)]),
        )
        for layer_index in second_order
    }
    second_plans = [
        (
            layer_index,
            build_transfer_plan(first_targets[layer_index], second_targets[layer_index], 256, 8, 8),
        )
        for layer_index in second_order
    ]
    _assert_peer_coverage(second_plans, require_redundant_source=True)
    second_snapshot = _clone_depth_live(weights)
    transfer.start(second_plans, transfer.prepare_transfer(second_plans))
    pending = _wait_all_pending(transfer, control_group, len(second_plans))
    _assert_depth_staging(rank, second_plans, second_targets, pending, transfer)
    _assert_depth_snapshot(weights, second_snapshot)
    dist.barrier(group=control_group)
    if rank == 0:
        time.sleep(0.1)
    for layer_index, buffer_index in pending:
        transfer.commit(layer_index, buffer_index)
    transfer.finish()
    torch.cuda.synchronize()
    dist.barrier(group=control_group)
    _assert_depth_live(weights, rank, second_plans, second_targets)
    _assert_depth_snapshot(weights, second_snapshot, set(range(9)) - set(second_order))
    _assert_depth_snapshot(weights, second_snapshot, primary_only=True)
    transfer.shutdown()
    dist.barrier(group=control_group)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 8,
    reason="requires eight CUDA GPUs",
)
def test_eplb_transfer_eight_gpu_bounded_staging_reuse():
    mp.spawn(_depth_worker, args=(_free_port(),), nprocs=8, join=True)
