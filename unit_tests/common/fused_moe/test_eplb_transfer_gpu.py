"""Multi-GPU correctness test for pinned-memory EPLB transfers."""

import gc
import os
import socket
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightllm.server.router.model_infer.mode_backend.eplb.expert_transfer import (
    EPLBTransferInfo,
    PinnedMemoryEPLBTransfer,
    TransferStatus,
    build_transfer_plan,
)


class _Pack:
    def __init__(self, weight, weight_scale):
        self.weight = weight
        self.weight_scale = weight_scale
        self.weight_zero_point = None


class _FakeWeight:
    def __init__(self, rank, layer_index):
        self.layer_num_ = layer_index
        logical_ids = ([0, 1, 2], [2, 3, 0])[rank]
        self.fuse_moe_impl = SimpleNamespace(
            num_redundant_experts_per_rank=1,
            local_logics_expert_ids_list=list(logical_ids),
        )
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


class _StressFakeWeight:
    """为并发传输压力测试生成可识别 source rank 的专家权重。"""

    def __init__(self, rank, layer_index, num_logical_experts):
        self.layer_num_ = layer_index
        logical_ids = list(range(num_logical_experts))
        self.fuse_moe_impl = SimpleNamespace(local_logics_expert_ids_list=logical_ids)
        self.w13 = self._pack(rank, logical_ids, layer_index, 0)
        self.w2 = self._pack(rank, logical_ids, layer_index, 100)

    @staticmethod
    def _pack(rank, logical_ids, layer_index, offset):
        # rank、layer、expert 和 tensor 类型都编码进数值；任何 recv 串包都会
        # 在目标 rank 的逐任务校验中表现为数值不一致。
        values = torch.tensor(
            [[rank * 100_000 + layer_index * 1_000 + expert + offset] for expert in logical_ids],
            dtype=torch.float32,
            device="cuda",
        )
        scales = values + 0.5
        return _Pack(values, scales)


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_transfer(transfer, control_group):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ready_count = torch.tensor([int(transfer.status is TransferStatus.SUCCEEDED)], dtype=torch.int32)
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=control_group)
        if int(ready_count.item()) == 1:
            return
        time.sleep(0.001)
    raise TimeoutError("EPLB transfer worker did not finish globally")


def _wait_for_all_transfers(transfers, control_group):
    """等待每个 rank 参与的全部并发传输完成。"""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        local_finished = all(transfer.status is TransferStatus.SUCCEEDED for transfer in transfers)
        globally_finished = torch.tensor([int(local_finished)], dtype=torch.int32)
        dist.all_reduce(globally_finished, op=dist.ReduceOp.MIN, group=control_group)
        if int(globally_finished.item()) == 1:
            return
        time.sleep(0.001)
    raise TimeoutError("concurrent EPLB transfer workers did not finish globally")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pytorch_keeps_unreferenced_pinned_source_alive_until_async_copy_finishes():
    """验证 pinned allocator 不会提前复用仍被异步 H2D 读取的内存。

    copy stream 中先排入一个长任务，确保 H2D copy 在 Python 引用释放时仍未
    完成。随后删除 pinned tensor 的唯一引用，并立刻申请同尺寸 pinned 内存：

    * 如果旧地址尚未复用，新 buffer 可以立即覆盖而不影响 H2D；
    * 如果 allocator 返回了旧地址，对应 copy event 必须已经完成；
    * 最终 GPU 数据必须保持为源 buffer 的原始内容。

    该测试验证当前 PyTorch/CUDA 组合的运行时行为；allocator 的正确性契约
    仍由 PyTorch ``copy_`` 中的 host ``record_event`` 实现提供。
    """
    torch.cuda.synchronize()
    num_elements = 8 * 1024 * 1024
    source = torch.full((num_elements,), 7, dtype=torch.int32, pin_memory=True)
    source_ptr = source.data_ptr()
    destination = torch.empty_like(source, device="cuda")

    copy_stream = torch.cuda.Stream()
    copy_finished = torch.cuda.Event()
    with torch.cuda.stream(copy_stream):
        # copy 与 sleep 位于同一 stream，必须等 sleep 完成后才能开始。
        torch.cuda._sleep(1_000_000_000)
        destination.copy_(source, non_blocking=True)
        copy_finished.record()

    assert not copy_finished.query(), "test setup failed to leave the H2D copy pending"

    del source
    gc.collect()

    replacement = torch.empty((num_elements,), dtype=torch.int32, pin_memory=True)
    if replacement.data_ptr() == source_ptr:
        # 相同地址只有在原 copy 已经结束、allocator 确认可以复用后才合法。
        assert copy_finished.query()
    replacement.fill_(-3)

    copy_finished.synchronize()
    assert torch.all(destination == 7)


def _worker(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    control_group = dist.new_group([0, 1], backend="gloo")
    transfer_group = dist.new_group([0, 1], backend="gloo")

    weights = [_FakeWeight(rank, layer_index) for layer_index in range(2)]
    current = [[0, 1, 2], [2, 3, 0]]
    target = [[0, 1, 3], [2, 3, 1]]
    for expected_layer in range(2):
        transfer_batches = build_transfer_plan(
            current,
            target,
            expected_layer,
            num_logical_experts=4,
            world_size=2,
        )
        for transfer_batch in transfer_batches:
            transfers = [PinnedMemoryEPLBTransfer(weights, transfer_group, rank, info) for info in transfer_batch]
            for transfer in transfers:
                assert all(buffer.pinned_row.is_pinned() for buffer in transfer.tensor_buffers)
                transfer.start()
            for transfer in transfers:
                _wait_for_transfer(transfer, control_group)
                if transfer.transfer_info.dest_rank == rank:
                    expected_expert = 3 if rank == 0 else 1
                    expected_w13 = expected_layer * 100 + expected_expert
                    expected_w2 = expected_w13 + 10
                    assert [buffer.name for buffer in transfer.tensor_buffers] == [
                        "w13.weight",
                        "w13.weight_scale",
                        "w2.weight",
                        "w2.weight_scale",
                    ]
                    pinned_rows = [buffer.pinned_row for buffer in transfer.tensor_buffers]
                    assert torch.all(pinned_rows[0][0] == expected_w13)
                    assert torch.all(pinned_rows[1][0] == expected_w13 + 0.5)
                    assert torch.all(pinned_rows[2][0] == expected_w2)
                    assert torch.all(pinned_rows[3][0] == expected_w2 + 0.5)

    # 主槽位互换会形成覆盖环。两个方向必须同时完成 GPU -> pinned memory
    # 传输后才能 commit，验证同一 rank 上并发的 send/recv 任务可以正常结束。
    swap_target = [[0, 3, 2], [2, 1, 0]]
    swap_plan = build_transfer_plan(current, swap_target, 0, num_logical_experts=4, world_size=2)
    assert len(swap_plan) == 1
    swap_infos = swap_plan[0]
    assert len(swap_infos) == 2
    swap_transfers = [PinnedMemoryEPLBTransfer(weights, transfer_group, rank, info) for info in swap_infos]
    for transfer in swap_transfers:
        transfer.start()
    for transfer in swap_transfers:
        _wait_for_transfer(transfer, control_group)

    for transfer in swap_transfers:
        if transfer.transfer_info.dest_rank == rank:
            expected_expert = transfer.transfer_info.source_logical_expert_id
            assert torch.all(transfer.tensor_buffers[0].pinned_row[0] == expected_expert)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs",
)
def test_eplb_pinned_memory_transfer_two_gpu_correctness():
    mp.spawn(_worker, args=(_free_port(),), nprocs=2, join=True)


def _many_concurrent_p2p_worker(rank, port):
    """同时运行大量、重复 rank 对的 PinnedMemoryEPLBTransfer。"""
    world_size = 4
    num_layers = 8
    num_logical_experts = 32
    transfers_per_rank_pair_per_layer = 8

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    control_group = dist.new_group(list(range(world_size)), backend="gloo")
    transfer_group = dist.new_group(list(range(world_size)), backend="gloo")

    try:
        weights = [_StressFakeWeight(rank, layer_index, num_logical_experts) for layer_index in range(num_layers)]

        # 每层覆盖全部 12 个有向 rank 对，每个 rank 对重复 8 次。任务 identity
        # 中的 layer、expert 和目标槽位不同，因此应该获得独立的 Gloo tag；
        # source/destination rank 对则会被大量重复使用。
        transfer_infos = []
        for layer_index in range(num_layers):
            for source_rank in range(world_size):
                for dest_rank in range(world_size):
                    if source_rank == dest_rank:
                        continue
                    source_peers = [peer_rank for peer_rank in range(world_size) if peer_rank != dest_rank]
                    source_peer_index = source_peers.index(source_rank)
                    for repeat_index in range(transfers_per_rank_pair_per_layer):
                        expert_id = source_rank * transfers_per_rank_pair_per_layer + repeat_index
                        dest_local_expert_index = source_peer_index * transfers_per_rank_pair_per_layer + repeat_index
                        transfer_infos.append(
                            EPLBTransferInfo(
                                source_rank=source_rank,
                                layer_index=layer_index,
                                source_logical_expert_id=expert_id,
                                dest_rank=dest_rank,
                                dest_local_expert_index=dest_local_expert_index,
                            )
                        )

        local_transfer_infos = [
            transfer_info
            for transfer_info in transfer_infos
            if rank in (transfer_info.source_rank, transfer_info.dest_rank)
        ]
        transfers = [
            PinnedMemoryEPLBTransfer(weights, transfer_group, rank, transfer_info)
            for transfer_info in local_transfer_infos
        ]
        assert len(transfer_infos) == 768
        assert len(transfers) == 384

        # 同一个 source/destination 对上的并发消息必须具有不同 tag，否则不同
        # 专家或张量可能被错误匹配。不同 rank 对可以安全复用相同整数 tag。
        message_keys = []
        for transfer in transfers:
            transfer_info = transfer.transfer_info
            for tensor_buffer in transfer.tensor_buffers:
                message_keys.append(
                    (
                        transfer_info.source_rank,
                        transfer_info.dest_rank,
                        transfer._build_p2p_message_tag(tensor_buffer.name),
                    )
                )
        assert len(message_keys) == len(set(message_keys))

        dist.barrier(group=control_group)
        for transfer in transfers:
            transfer.start()
        _wait_for_all_transfers(transfers, control_group)

        destination_transfers = [transfer for transfer in transfers if transfer.transfer_info.dest_rank == rank]
        assert len(destination_transfers) == 192
        for transfer in destination_transfers:
            transfer_info = transfer.transfer_info
            expected_w13 = (
                transfer_info.source_rank * 100_000
                + transfer_info.layer_index * 1_000
                + transfer_info.source_logical_expert_id
            )
            expected_values = [expected_w13, expected_w13 + 0.5, expected_w13 + 100, expected_w13 + 100.5]
            assert [buffer.name for buffer in transfer.tensor_buffers] == [
                "w13.weight",
                "w13.weight_scale",
                "w2.weight",
                "w2.weight_scale",
            ]
            for tensor_buffer, expected_value in zip(transfer.tensor_buffers, expected_values):
                assert torch.all(tensor_buffer.pinned_row == expected_value)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="requires four CUDA GPUs",
)
def test_eplb_pinned_memory_transfer_four_gpu_many_concurrent_p2p():
    mp.spawn(_many_concurrent_p2p_worker, args=(_free_port(),), nprocs=4, join=True)
