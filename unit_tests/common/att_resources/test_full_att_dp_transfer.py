"""DP 直接索引复制：完整编码、真实 stride、非默认流和跨设备 IPC。"""

from __future__ import annotations

from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
import traceback

import pytest
import torch
import torch.multiprocessing as mp

from lightllm.common.attention_state.full_att import StaticFP8FullAttentionState
from lightllm.common.attention_state.full_att.dp_transfer import copy_gqa_tokens
from unit_tests.common.att_resources.model_helpers import full_state_test_buffers, rebuild_full_state
from unit_tests.common.att_resources.test_capacity import definition


SCHEMES = ("gqa", "int4", "int8", "static_head", "static_tensor", "mla", "packed", "packed_fp8", "packed_glm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", SCHEMES)
@pytest.mark.parametrize("source_device", [0, 1])
@pytest.mark.parametrize("token_count", [0, 1, 513])
def test_dp_copy_preserves_complete_rows_without_temporary_payload(
    kind: str,
    source_device: int,
    token_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if source_device >= torch.cuda.device_count():
        pytest.skip("two CUDA devices required")
    torch.cuda.set_device(0)
    if source_device != 0:
        if not torch.cuda.can_device_access_peer(0, source_device):
            pytest.skip("CUDA peer access required")
        # 仅测试直接跨卡 Tensor 的场景需要预先开 peer；生产通过 IPC 装配，不在复制时建映射。
        from cuda.bindings import runtime
        error, = runtime.cudaDeviceEnablePeerAccess(source_device, 0)
        assert error in (runtime.cudaError_t.cudaSuccess, runtime.cudaError_t.cudaErrorPeerAccessAlreadyEnabled)

    template = definition(kind)
    source = rebuild_full_state(template, token_capacity=1024, device=f"cuda:{source_device}")
    target = rebuild_full_state(template, token_capacity=1200, device="cuda:0")
    source_buffers = full_state_test_buffers(source)
    target_buffers = full_state_test_buffers(target)
    expected, original = {}, {}
    for name, tensor in source_buffers.items():
        tensor.view(torch.uint8).random_(0, 256)
        target_buffers[name].view(torch.uint8).fill_(173)
        original[name] = tensor.cpu().view(torch.uint8)
        expected[name] = target_buffers[name].cpu().view(torch.uint8)

    # 源索引允许重复，目标唯一；两端都带 storage offset 和非单位 stride。
    src_list = [(i * 29) % 257 for i in range(token_count)]
    dst_list = [(i * 7) % 1200 for i in range(token_count)]
    src_dtype = torch.int32 if source_device == 0 else torch.int64
    dst_dtype = torch.int64 if source_device == 0 else torch.int32
    src_ids = torch.empty(size=(token_count * 2 + 1,), dtype=src_dtype, device=f"cuda:{source_device}")[1::2]
    dst_ids = torch.empty(size=(token_count * 3 + 2,), dtype=dst_dtype, device="cuda:0")[2::3]
    src_ids.copy_(torch.tensor(src_list, dtype=src_dtype, device=f"cuda:{source_device}"))
    dst_ids.copy_(torch.tensor(dst_list, dtype=dst_dtype, device="cuda:0"))
    for name in expected:
        expected[name][:, dst_list] = original[name][:, src_list]
    calibration = []
    if isinstance(target, StaticFP8FullAttentionState):
        for tensor in (target.scales, target.q_scales):
            if tensor is not None:
                calibration.append((tensor, tensor.clone()))

    # 显式依赖事件，不让测试偶然依赖默认流；跨设备源索引也必须等生产完成。
    producer_done = torch.cuda.Event()
    producer_done.record(torch.cuda.current_stream(device=source_device))
    stream = torch.cuda.Stream(device=0)
    stream.wait_event(producer_done)
    stream.wait_stream(torch.cuda.current_stream(device=0))

    def reject_intermediate(*args: object, **kwargs: object) -> None:
        pytest.fail("DP copies must use supplied CUDA indices and storage directly")

    with monkeypatch.context() as patch:
        for method in ("index_select", "index_copy_", "to", "tolist", "cpu", "contiguous"):
            patch.setattr(torch.Tensor, method, reject_intermediate)
        with torch.cuda.stream(stream):
            target.copy_tokens_from_dp(
                src_token_indices=src_ids,
                dst_token_indices=dst_ids,
                remote_dp_full_att_state=source,
            )
    stream.synchronize()
    for name, tensor in target_buffers.items():
        # 比较全部存储：包含未选槽位、HOLD 页、packed padding 和任意浮点位模式。
        assert torch.equal(tensor.cpu().view(torch.uint8), expected[name])
        assert torch.equal(source_buffers[name].cpu().view(torch.uint8), original[name])
    for tensor, unchanged in calibration:
        assert torch.equal(tensor, unchanged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("column_stride", [1, 2])
@pytest.mark.parametrize("offset", [0, 1, 2, 4])
def test_dp_copy_uses_real_storage_strides(column_stride: int, offset: int) -> None:
    # 宽类型 view 的 8/4/2 字节对齐及不连续末轴回退；源目标层/token/head stride 不同。
    source_base = torch.empty(size=(2, 19, 3, 32), dtype=torch.bfloat16, device="cuda")
    target_base = torch.empty(size=(2, 23, 3, 48), dtype=torch.bfloat16, device="cuda")
    source_base.view(torch.uint8).random_(0, 256)
    target_base.view(torch.uint8).fill_(173)
    source = source_base[:, ::2, :, offset:offset + 8 * column_stride:column_stride]
    target = target_base[:, ::3, :, offset:offset + 8 * column_stride:column_stride]
    expected = target_base.clone()
    expected[:, ::3, :, offset:offset + 8 * column_stride:column_stride][:, [4, 0, 6]] = source[:, [7, 2, 7]]
    copy_gqa_tokens(
        src_k_storage=source,
        src_v_storage=source,
        dst_k_storage=target,
        dst_v_storage=torch.empty_like(target),
        src_token_indices=torch.tensor([7, 2, 7], dtype=torch.int32, device="cuda"),
        dst_token_indices=torch.tensor([4, 0, 6], dtype=torch.int32, device="cuda"),
    )
    torch.cuda.synchronize()
    assert torch.equal(target_base.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_dp_copy_through_production_peer_ipc(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightllm.server.router.model_infer.mode_backend.pd.p2p_fix import reduce_tensor

    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("CUDA peer access required")
    states = [rebuild_full_state(definition(kind), token_capacity=16, device="cuda:1") for kind in SCHEMES]
    for state in states:
        for tensor in full_state_test_buffers(state).values():
            tensor.view(torch.uint8).random_(0, 256)
    source_ids = torch.tensor([3, 99, 1, 99, 3], dtype=torch.int32, device="cuda:1")[::2]
    torch.cuda.synchronize(device=1)
    monkeypatch.setattr(mp.reductions.reduce_tensor, "__code__", reduce_tensor.__code__)
    payload = bytes(ForkingPickler.dumps((states, source_ids)))
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_ipc_copy_worker, args=(child, payload))
    process.start()
    child.close()
    try:
        assert parent.poll(60), "DP peer worker did not finish"
        error = parent.recv()
        assert error is None, error
        parent.send(None)
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()


def _ipc_copy_worker(connection: Connection, payload: bytes) -> None:
    try:
        torch.cuda.set_device(0)
        sources, src_ids = ForkingPickler.loads(payload)
        for kind, source in zip(SCHEMES, sources):
            target = rebuild_full_state(definition(kind), token_capacity=24, device="cuda:0")
            target_buffers = full_state_test_buffers(target)
            for tensor in target_buffers.values():
                tensor.view(torch.uint8).fill_(173)
            expected = {name: tensor.cpu().view(torch.uint8) for name, tensor in target_buffers.items()}
            for name, tensor in full_state_test_buffers(source).items():
                expected[name][:, [8, 2, 5]] = tensor.cpu().view(torch.uint8)[:, [3, 1, 3]]
            target.copy_tokens_from_dp(
                src_token_indices=src_ids,
                dst_token_indices=torch.tensor([8, 2, 5], dtype=torch.int64, device="cuda:0"),
                remote_dp_full_att_state=source,
            )
            torch.cuda.synchronize()
            for name, tensor in target_buffers.items():
                assert torch.equal(tensor.cpu().view(torch.uint8), expected[name])
        connection.send(None)
        connection.recv()
    except Exception:
        connection.send(traceback.format_exc())
    finally:
        connection.close()
