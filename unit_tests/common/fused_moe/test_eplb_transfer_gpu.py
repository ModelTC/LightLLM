"""Multi-GPU correctness test for pinned-memory EPLB transfers."""

import os
import socket
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    PinnedMemoryEPLBTransfer,
    build_transfer_plan,
)


class _Pack:
    def __init__(self, weight, weight_scale):
        self.weight = weight
        self.weight_scale = weight_scale
        self.weight_zero_point = None


class _FakeWeight:
    def __init__(self, rank, layer_index):
        self.fuse_moe_impl = SimpleNamespace(
            num_primary_experts_per_rank=2,
            num_redundant_experts_per_rank=1,
        )
        logical_ids = ([0, 1, 2], [2, 3, 0])[rank]
        self.w13 = self._pack(logical_ids, layer_index, 0)
        self.w2 = self._pack(logical_ids, layer_index, 10)

    @staticmethod
    def _pack(logical_ids, layer_index, offset):
        values = torch.tensor(
            [[layer_index * 100 + expert + offset] for expert in logical_ids],
            dtype=torch.float16,
            device="cuda",
        )
        scales = values.to(torch.float32) + 0.5
        return _Pack(values, scales)


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_ready_layer(transfer, control_group):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ready_layer = transfer.ready_layer()
        ready_count = torch.tensor([int(ready_layer is not None)], dtype=torch.int32)
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=control_group)
        if int(ready_count.item()) == 1:
            return ready_layer
        time.sleep(0.001)
    raise TimeoutError("EPLB transfer worker did not publish a globally ready layer")


def _worker(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    control_group = dist.new_group([0, 1], backend="gloo")
    transfer_group = dist.new_group([0, 1], backend="gloo")

    weights = [_FakeWeight(rank, layer_index) for layer_index in range(2)]
    transfer = PinnedMemoryEPLBTransfer(weights, transfer_group, rank)
    assert all(row.is_pinned() for _, row in transfer.pinned_rows)
    current = torch.tensor([[2], [0]])
    target = torch.tensor([[3], [1]])
    plan = build_transfer_plan(current, target, num_logical_experts=4, world_size=2, node_world_size=2)
    layer_plans = [(0, plan), (1, plan)]
    transfer.start(layer_plans)

    for expected_layer in range(2):
        layer_index = _wait_for_ready_layer(transfer, control_group)
        assert layer_index == expected_layer
        expected_expert = 3 if rank == 0 else 1
        expected_w13 = expected_layer * 100 + expected_expert
        expected_w2 = expected_w13 + 10
        assert torch.all(transfer.staging[0][1][0] == expected_w13)
        assert torch.all(transfer.staging[1][1][0] == expected_w13 + 0.5)
        assert torch.all(transfer.staging[2][1][0] == expected_w2)
        assert torch.all(transfer.staging[3][1][0] == expected_w2 + 0.5)
        transfer.commit(layer_index)

    transfer.finish()
    torch.cuda.synchronize()
    for layer_index, weight in enumerate(weights):
        expected_expert = 3 if rank == 0 else 1
        expected_w13 = layer_index * 100 + expected_expert
        assert torch.all(weight.w13.weight[2] == expected_w13)
        assert torch.all(weight.w2.weight[2] == expected_w13 + 10)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs",
)
def test_eplb_pinned_memory_transfer_two_gpu_correctness():
    mp.spawn(_worker, args=(_free_port(),), nprocs=2, join=True)
