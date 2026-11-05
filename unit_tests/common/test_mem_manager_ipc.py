from multiprocessing.reduction import ForkingPickler

import pytest
import torch
import torch.multiprocessing as mp

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.allocators import full_att_kv_allocator as allocator_module
from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState


class _Counter:
    def __init__(self, *args):
        pass

    def set_value(self, value):
        self.value = value


def _make_manager(monkeypatch, size, device="cpu"):
    monkeypatch.setattr(allocator_module, "SharedInt", _Counter)
    monkeypatch.setattr(allocator_module, "get_current_rank_in_node", lambda: 0)
    manager = AttentionStateManager(request_capacity=2)
    state = GQAFullAttentionState(
        GQAAttentionConfig(logical_layer_groups=((0,),), global_head_num=1, head_dim=4, dtype=torch.float32),
        device=device,
    )
    state.allocate_storage(4)
    manager.register_state("main", state)
    manager.finish_state_registration()
    manager.full_att_allocator = allocator_module.FullAttKVAllocator(size, pin_memory=False)
    mapping = torch.arange(4, dtype=torch.int32, device=device).reshape(1, 4)
    return manager, mapping


def _export(manager, mapping, kind):
    operator = manager.get_dp_shared_operator() if kind == "dp" else manager.get_pd_operator()
    return ForkingPickler.dumps((operator.create_ipc_copy(), mapping))


@pytest.mark.parametrize("kind", ["dp", "pd"])
def test_ipc_descriptor_size_does_not_scale_with_local_allocator(monkeypatch, kind):
    small, small_mapping = _make_manager(monkeypatch, 256)
    large, large_mapping = _make_manager(monkeypatch, 131072)
    small_payload = _export(small, small_mapping, kind)
    large_payload = _export(large, large_mapping, kind)
    assert abs(len(large_payload) - len(small_payload)) < 128


@pytest.mark.parametrize("kind", ["dp", "pd"])
def test_ipc_roundtrip_retains_shared_buffers_and_local_owner(monkeypatch, kind):
    source, mapping = _make_manager(monkeypatch, 256)
    allocator = source.full_att_allocator
    exported, exported_mapping = ForkingPickler.loads(_export(source, mapping, kind))
    assert source.full_att_allocator is allocator
    assert allocator.alloc(1).tolist() == [0]
    assert exported.full_att_allocator is None
    assert exported.get_dp_shared_operator().owner is exported
    assert exported.get_pd_operator().owner is exported
    state = exported.get_full_attention_states()[0]
    state.k_storage.fill_(17)
    state.v_storage.fill_(19)
    exported_mapping.fill_(23)
    source_state = source.get_full_attention_states()[0]
    assert torch.all(source_state.k_storage == 17)
    assert torch.all(source_state.v_storage == 19)
    assert torch.all(mapping == 23)


def _update_exported_buffers(payload, connection):
    try:
        exported, mapping = ForkingPickler.loads(payload)
        assert exported.full_att_allocator is None
        state = exported.get_full_attention_states()[0]
        state.k_storage.fill_(17)
        state.v_storage.fill_(19)
        mapping.fill_(23)
        if mapping.is_cuda:
            torch.cuda.synchronize()
        connection.send("complete")
    finally:
        connection.close()


@pytest.mark.parametrize("kind", ["dp", "pd"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ipc_buffers_remain_shared_between_processes(monkeypatch, device, kind):
    source, mapping = _make_manager(monkeypatch, 256, device)
    allocator = source.full_att_allocator
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_update_exported_buffers, args=(_export(source, mapping, kind).tobytes(), sender))
    try:
        process.start()
        sender.close()
        assert receiver.poll(60), "IPC consumer did not complete"
        assert receiver.recv() == "complete"
        process.join(10)
        assert process.exitcode == 0
        assert source.full_att_allocator is allocator
        state = source.get_full_attention_states()[0]
        assert torch.all(state.k_storage == 17)
        assert torch.all(state.v_storage == 19)
        assert torch.all(mapping == 23)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        receiver.close()
        sender.close()
