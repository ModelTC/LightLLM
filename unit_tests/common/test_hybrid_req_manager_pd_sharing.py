import multiprocessing
import traceback
from multiprocessing.reduction import ForkingPickler
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import Glm5NextMemManager, MemoryManager
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager
from lightllm.common.linear_state import LinearStateManager
from lightllm.common.req_manager import Glm5NextReqManager, ReqManagerForMamba
from lightllm.common.req_manager.linear_att_pd_state import LinearAttPDState
from lightllm.common.state_cache_manager import (
    Glm5NextCacheConfig,
    LayerCache,
    LinearAttCacheConfig,
    LinearAttCacheManager,
)
from lightllm.server.router.model_infer.mode_backend.pd.p2p_fix import reduce_tensor


_LOCAL_FIELDS = ("mem_manager", "small_page_buffers", "req_list", "req_sampling_params_manager")


class _SerializationFailure:
    def __reduce_ex__(self, protocol):
        raise RuntimeError("injected serialization failure")


def _assert_received_manager_roundtrip(payload, model, mode, expected):
    restored_mem = ForkingPickler.loads(payload)
    restored = restored_mem.req_manager
    assert type(restored) is (Glm5NextReqManager if model == "glm" else ReqManagerForMamba)
    assert restored_mem.big_page_buffers is None
    assert not set(_LOCAL_FIELDS).intersection(vars(restored))
    assert not {"linear_state_manager", "pd_state", "req_to_indexer_tail"}.intersection(vars(restored_mem))
    assert restored.pd_state.linear_state_manager is restored.linear_state_manager
    assert restored_mem.req_to_token_indexs is restored.req_to_token_indexs
    if model == "glm":
        assert restored.pd_state.indexer_tail is restored.req_to_indexer_tail
    assert restored.get_pd_state_size() == expected["size"]
    src, dst = 1, 2
    block = torch.empty(restored.get_pd_state_size(), dtype=torch.uint8, device="cuda")
    restored.get_pd_state(src, block, tp_rank=0)
    restored.load_pd_state(dst, block, tp_rank=0)
    torch.cuda.synchronize()
    state = restored.linear_state_manager
    actual = state.get_accepted_state(dst)
    torch.testing.assert_close(
        actual.conv, torch.tensor(expected["conv"], dtype=actual.conv.dtype, device="cuda"), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.ssm, torch.tensor(expected["ssm"], dtype=actual.ssm.dtype, device="cuda"), rtol=0, atol=0
    )
    assert state.accepted_state_index[dst] == 0
    assert state.accepted_state_index[src] == 1
    if mode == "replay":
        assert state.ssm_backend.cursors[dst] == 0
        assert state.ssm_backend.cursors[src] == 1
    if model == "glm":
        tail = restored.req_to_indexer_tail.buffer[:, dst]
        torch.testing.assert_close(
            tail, torch.tensor(expected["tail"], dtype=tail.dtype, device="cuda"), rtol=0, atol=0
        )
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


@pytest.mark.parametrize("raise_error", [False, True])
def test_pd_sharing_context_preserves_absent_fields_and_restores_original_objects(raise_error):
    manager = object.__new__(ReqManagerForMamba)
    manager.mem_manager = object()
    manager.small_page_buffers = None
    manager.retained = object()
    before = vars(manager).copy()

    def share():
        with manager.pd_sharing_context():
            assert set(vars(manager)) == {"retained"}
            assert manager.retained is before["retained"]
            if raise_error:
                raise RuntimeError("injected context failure")

    if raise_error:
        with pytest.raises(RuntimeError, match="injected context failure"):
            share()
    else:
        share()
    assert set(vars(manager)) == set(before)
    for field, value in before.items():
        assert vars(manager)[field] is value


def _make_sharing_pair(model, mode):
    config_type = Glm5NextCacheConfig if model == "glm" else LinearAttCacheConfig
    config = config_type(
        tp_world_size=1,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=32,
        global_linear_k_heads=1,
        global_linear_v_heads=1,
        num_linear_k_heads=1,
        num_linear_v_heads=1,
        head_linear_k_dim=32,
        head_linear_v_dim=32,
        conv_kernel_size=4,
        linear_layer_num=2,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=3,
        all_layer_num=3,
        draft_full_att_kv_layer_num=1,
    )
    args = SimpleNamespace(mtp_step=2, ssm_state_mode=mode, ssm_replay_config={"cache_len": 4})
    state = LinearStateManager(3, config, args, kda=model == "glm")
    manager_type = Glm5NextReqManager if model == "glm" else ReqManagerForMamba
    mem_type = Glm5NextMemManager if model == "glm" else Qwen3NextMemManager
    # Omit unrelated allocators and service SHM setup; use the real sharing path.
    manager, mem = object.__new__(manager_type), object.__new__(mem_type)
    manager.linear_config = mem.linear_config = config
    manager.mtp_step = 2
    manager.max_request_num = 3
    manager.HOLD_REQUEST_ID = 3
    manager.req_to_token_indexs = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
    manager.linear_state_manager = state
    manager.mem_manager = mem
    manager.small_page_buffers = LinearAttCacheManager(2, config)
    manager.req_list = Lock()  # A leaked local resource must make real pickling fail.
    manager.req_sampling_params_manager = SimpleNamespace(pinned=torch.empty(8, pin_memory=True))
    if model == "glm":
        manager.req_to_indexer_tail = LayerCache(
            4,
            config.full_att_dtype,
            (config.index_kpool + manager.mtp_step, 2 * config.index_head_dim),
            config.get_full_att_kv_layer_num_with_draft_model(),
            "cuda",
        )
        manager.req_to_indexer_tail.buffer.normal_()
    manager.pd_state = LinearAttPDState(state, getattr(manager, "req_to_indexer_tail", None))
    mem.big_page_buffers = LinearAttCacheManager(2, config)
    mem.kv_buffer = torch.zeros((2, 8, 1, 32), dtype=torch.bfloat16, device="cuda")
    return manager, mem


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC and pinned memory required")
@pytest.mark.parametrize("model", ["qwen", "glm"])
@pytest.mark.parametrize("mode", ["native", "replay"])
@pytest.mark.parametrize("fail_serialization", [False, True])
def test_shared_req_manager_roundtrip_preserves_worker_resources(
    monkeypatch, pd_receiver, model, mode, fail_serialization
):
    manager, mem = _make_sharing_pair(model, mode)
    state = manager.linear_state_manager
    src, dst = 1, 2
    state.conv_state.buffer.copy_(torch.arange(5, dtype=torch.bfloat16, device="cuda"))
    for slot in range(state.ssm_state.buffer.shape[1]):
        state.ssm_state.buffer[:, slot].fill_(slot + 1)
    state.accepted_state_index[src] = 1
    state.accepted_state_index[dst] = 2
    if mode == "replay":
        state.ssm_backend.raw_keys.zero_()
        state.ssm_backend.raw_values.zero_()
        state.ssm_backend.gates.fill_(-0.5)
        state.ssm_backend.betas.zero_()
        state.ssm_backend.cursors[src] = 1
        state.ssm_backend.cursors[dst] = 3
    expected = state.get_accepted_state(src)
    expected_conv, expected_ssm = expected.conv.clone(), expected.ssm.clone()
    source_conv = state.conv_state.buffer[:, src].clone()
    source_ssm = state.ssm_state.buffer[:, src * state.ssm_slots_per_req].clone()
    expected_tail = manager.req_to_indexer_tail.buffer[:, src].clone() if model == "glm" else None
    resources = vars(manager).copy()
    big_pool = mem.big_page_buffers
    pool_tensors = [
        tensor
        for pool in (big_pool, manager.small_page_buffers)
        for tensor in (pool.conv_state_cache.buffer, pool.ssm_state_cache.buffer)
    ] + [manager.req_sampling_params_manager.pinned]
    snapshots = []
    for index, tensor in enumerate(pool_tensors):
        tensor.fill_(index + 1)
        assert tensor.is_pinned()
        snapshots.append((tensor, tensor.data_ptr(), tensor.clone()))

    def serialize(shared_mem, shared_req):
        assert shared_mem.req_manager is shared_req is manager
        assert shared_mem.big_page_buffers is None
        assert not set(_LOCAL_FIELDS).intersection(vars(shared_req))
        assert not {"linear_state_manager", "pd_state", "req_to_indexer_tail"}.intersection(vars(shared_mem))
        shared_mem.req_to_token_indexs = shared_req.req_to_token_indexs
        # Fail before exporting CUDA handles that no receiver could release.
        payload = (_SerializationFailure(), shared_mem) if fail_serialization else shared_mem
        return ForkingPickler.dumps(payload).tobytes()

    monkeypatch.setattr(MemoryManager, "write_to_shm", serialize)
    monkeypatch.setitem(ForkingPickler._extra_reducers, torch.Tensor, reduce_tensor)
    if fail_serialization:
        with pytest.raises(RuntimeError, match="injected serialization failure"):
            mem.write_to_shm(manager)
    else:
        expected_values = {
            "size": manager.get_pd_state_size(),
            "conv": expected_conv.cpu().tolist(),
            "ssm": expected_ssm.cpu().tolist(),
            "tail": expected_tail.cpu().tolist() if expected_tail is not None else None,
        }
        sender, receiver_process = pd_receiver
        sender.send((mem.write_to_shm(manager), model, mode, expected_values))
        assert sender.poll(30), f"PD receiver did not respond; exit code: {receiver_process.exitcode}"
        error = sender.recv()
        assert error is None, error

    assert set(vars(manager)) == set(resources)
    for field, value in resources.items():
        assert vars(manager)[field] is value
    assert mem.big_page_buffers is big_pool
    torch.testing.assert_close(state.conv_state.buffer[:, src], source_conv, rtol=0, atol=0)
    torch.testing.assert_close(state.ssm_state.buffer[:, src * state.ssm_slots_per_req], source_ssm, rtol=0, atol=0)
    for tensor, pointer, contents in snapshots:
        assert tensor.data_ptr() == pointer
        assert tensor.is_pinned()
        torch.testing.assert_close(tensor, contents, rtol=0, atol=0)
