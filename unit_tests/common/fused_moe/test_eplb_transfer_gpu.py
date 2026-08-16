"""Two-GPU NIXL EPLB correctness and 512 MiB micro-performance test."""
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
    build_transfer_plan,
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
            logical_experts=32,
            world_size=2,
            eplb=EPLBState(
                redundant_experts_per_rank=16,
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
    for _ in range(3):
        transfer._copy_batch(batch)
    dist.barrier(group=control_group)
    samples = []
    for _ in range(20):
        started = time.perf_counter()
        transfer._copy_batch(batch)
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
