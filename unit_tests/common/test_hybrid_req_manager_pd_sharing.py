import multiprocessing
import traceback
from multiprocessing.reduction import ForkingPickler

import pytest
import torch

from lightllm.common.attention_state.bounded_att import GDNState, KDAState
from lightllm.server.router.model_infer.mode_backend.pd.p2p_fix import reduce_tensor
from unit_tests.common.test_linear_state import make_state


class _SerializationFailure:
    def __reduce_ex__(self, protocol):
        raise RuntimeError("injected serialization failure")


def _assert_received_manager_roundtrip(payload, state_name, mode, expected):
    resources = ForkingPickler.loads(payload)
    state = getattr(resources, state_name)
    assert state.big_pages_pool is state.small_pages_pool is None
    assert resources.full_att_allocator is None
    layout = resources.get_pd_operator().get_pd_bounded_att_page_layouts()[state_name]
    block = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    state.write_to_pd_page(1, block, 0, 8, layout)
    state.read_from_pd_page(block, 2, 0, 8, layout)
    width = state.config.conv_width
    ssm_index = 2 if mode == "replay" else 2 * (state.mtp_step + 1)
    actual = (state.conv_state[:, 2, :, :width], state.ssm_state[:, ssm_index])
    for tensor, values in zip(actual, expected):
        torch.testing.assert_close(tensor, torch.tensor(values, dtype=tensor.dtype, device="cuda"), rtol=0, atol=0)
    assert state.mtp_state_index[2] == 0
    assert state.mtp_state_index[1] == 1
    if mode == "replay":
        assert state.cursors[2] == 0
        assert state.cursors[1] == 1
    torch.cuda.synchronize()


def _receive_pd_managers(connection):
    torch.cuda.set_device(0)
    while True:
        message = connection.recv()
        if message is None:
            return
        try:
            _assert_received_manager_roundtrip(*message)
            connection.send(None)
        except Exception:
            connection.send(traceback.format_exc())


@pytest.fixture(scope="module")
def pd_receiver():
    context = multiprocessing.get_context("spawn")
    sender, receiver = context.Pipe()
    process = context.Process(target=_receive_pd_managers, args=(receiver,))
    process.start()
    receiver.close()
    try:
        yield sender, process
    finally:
        if process.is_alive():
            sender.send(None)
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        sender.close()
        assert process.exitcode == 0


@pytest.mark.parametrize("cls", [GDNState, KDAState])
@pytest.mark.parametrize("mode", ["native", "replay"])
def test_pd_export_copy_keeps_worker_pools_and_runtime_identity(cls, mode):
    state, resources, _ = make_state(cls=cls, mode=mode, device="meta")
    state.big_pages_pool, state.small_pages_pool = object(), object()
    before = vars(state).copy()
    exported = resources.get_pd_operator().create_ipc_copy()
    copy = getattr(exported, state.state_name)
    assert copy.big_pages_pool is copy.small_pages_pool is None
    assert copy.conv_state is state.conv_state
    assert copy.ssm_state is state.ssm_state
    if mode == "replay":
        assert copy.cursors is state.cursors
        assert copy.keys is state.keys
    assert copy.mtp_state_index is state.mtp_state_index
    assert all(vars(state)[name] is value for name, value in before.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC and pinned memory required")
@pytest.mark.parametrize("cls", [GDNState, KDAState])
@pytest.mark.parametrize("mode", ["native", "replay"])
@pytest.mark.parametrize("fail_serialization", [False, True])
def test_shared_attention_state_roundtrip_preserves_worker_resources(
    runtime_start_args, monkeypatch, pd_receiver, cls, mode, fail_serialization
):
    # Torch 2.11 uses pidfd_getfd for expandable segments, which is unavailable in
    # some containers. Disable expandable segments before allocating the exported state.
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    state, resources, _ = make_state(cls=cls, mode=mode)
    state.initialize_page_pools(big_page_capacity=2, small_page_capacity=2)
    state.conv_state.copy_(torch.arange(5, dtype=torch.bfloat16, device="cuda"))
    for slot in range(state.ssm_state.shape[1]):
        state.ssm_state[:, slot].fill_(slot + 1)
    state.mtp_state_index[1], state.mtp_state_index[2] = 1, 2
    if mode == "replay":
        state.raw_keys.zero_()
        state.raw_values.zero_()
        state.gates.fill_(-0.5)
        state.betas.zero_()
        state.cursors[1], state.cursors[2] = 1, 3
    width = state.config.conv_width
    if mode == "replay":
        expected = [
            state.conv_state[:, 1, :, :width].cpu().tolist(),
            state.ssm_state[:, 1].cpu().tolist(),
        ]
    else:
        expected = [
            state.conv_state[:, 1, :, :width].cpu().tolist(),
            state.ssm_state[:, state.mtp_step + 1].cpu().tolist(),
        ]
    before = vars(state).copy()
    source_slot = 1 if mode == "replay" else state.mtp_step + 1
    source_conv, source_ssm = state.conv_state[:, 1].clone(), state.ssm_state[:, source_slot].clone()
    snapshots = []
    for pool in (state.big_pages_pool, state.small_pages_pool):
        for tensor in (pool.conv_state, pool.ssm_state):
            tensor.fill_(3)
            snapshots.append((tensor, tensor.data_ptr(), tensor.clone()))
    monkeypatch.setitem(ForkingPickler._extra_reducers, torch.Tensor, reduce_tensor)
    exported = resources.get_pd_operator().create_ipc_copy()
    if fail_serialization:
        with pytest.raises(RuntimeError, match="injected serialization failure"):
            ForkingPickler.dumps((_SerializationFailure(), exported))
    else:
        sender, receiver = pd_receiver
        sender.send((ForkingPickler.dumps(exported).tobytes(), state.state_name, mode, expected))
        assert sender.poll(30), f"PD receiver did not respond; exit code: {receiver.exitcode}"
        error = sender.recv()
        assert error is None, error
    assert all(vars(state)[name] is value for name, value in before.items())
    torch.testing.assert_close(state.conv_state[:, 1], source_conv, rtol=0, atol=0)
    torch.testing.assert_close(state.ssm_state[:, source_slot], source_ssm, rtol=0, atol=0)
    for tensor, pointer, contents in snapshots:
        assert tensor.data_ptr() == pointer
        torch.testing.assert_close(tensor, contents, rtol=0, atol=0)
