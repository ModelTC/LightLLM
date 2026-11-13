import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unit_tests.common.dsv4_test_utils import DeepseekV4MemoryManager, DeepseekV4CacheConfig, DeepseekV4ReqManager
from unit_tests.common.req_token_index_test_utils import attach_req_token_index
from lightllm.models.deepseek_v4.triton_kernel.build_compressed_kv_indices import build_compressed_kv_indices
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.deepseek_v4.triton_kernel.swa_cache import get_swa_token_indices as _get_swa_token_indices


def _compressed_kv_pools(manager):
    c4 = next(iter(manager.cache_config.layer_to_c4_idx))
    c128 = next(iter(manager.cache_config.layer_to_c128_idx))
    return (
        manager.full_states[c4].get_layer_view(c4).kv.unsqueeze(0),
        manager.full_states[c4].get_layer_view(c4).indexer_k.unsqueeze(0),
        manager.full_states[c128].get_layer_view(c128).kv.unsqueeze(0),
    )


def _randomize_compressed_kv(manager):
    for buffer in manager.kv_buffer:
        buffer.random_(0, 256)


@pytest.fixture
def cache(monkeypatch, tmp_path, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the packed cache kernels")
    param = getattr(request, "param", 256)
    page_size, mtp_step = param if isinstance(param, tuple) else (param, 3)
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 100}))
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_dsv4_paged_cache")
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "page_size": page_size,
                "run_mode": "normal",
                "linear_att_hash_page_size": page_size,
                "linear_att_page_block_num": 8,
                "cpu_cache_token_page_size": page_size * 8,
                "chunked_prefill_size": 2048,
                "max_req_total_len": 4096,
                "disable_chunked_prefill": False,
                "model_dir": str(tmp_path),
                "penalty_counter_mode": "cpu_counter",
                "output_constraint_mode": "none",
                "mtp_step": mtp_step,
                "mtp_dynamic_verify": False,
                "enable_ep_moe": False,
                "enable_dp_prompt_cache_fetch": False,
            }
        ),
    )
    get_env_start_args.cache_clear()
    manager = DeepseekV4MemoryManager(
        8192,
        torch.bfloat16,
        1,
        512,
        3,
        DeepseekV4CacheConfig.from_model_config(
            dict(
                num_hidden_layers=3,
                compress_ratios=[4, 128, 0],
                head_dim=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
                sliding_window=128,
            ),
            SimpleNamespace(mtp_mode=None),
        ),
    )
    requests = DeepseekV4ReqManager(2, 4096, None, manager.cache_config, mtp_step=mtp_step)
    requests.bind_mem_manager(manager)
    yield manager, requests
    torch.cuda.synchronize()
    get_env_start_args.cache_clear()


def test_copy_mem_to_mem_preserves_complete_block_indices(cache):
    manager, _ = cache
    # 压缩块内的 KV/scale/indexer 不可拆分；允许按非连续、乱序的完整块复制。
    source = torch.cat([torch.arange(start, start + 256, device="cuda") for start in (0, 512, 256)])
    destination = torch.cat([torch.arange(start, start + 256, device="cuda") for start in (1024, 1536, 1280)])
    for buffer in manager.kv_buffer:
        buffer.zero_()
        values = torch.arange(1, 4, device="cuda", dtype=torch.uint8).repeat_interleave(256)
        buffer[source] = values[:, None, None]
    manager.operator.copy_mem_to_mem(source, destination)
    for buffer in manager.kv_buffer:
        torch.testing.assert_close(buffer[destination], buffer[source], rtol=0, atol=0)


def test_cache_io_copies_individual_token_slots(cache):
    manager, _ = cache
    source = torch.tensor([3, 17, 257, 509], device="cuda")
    destination = torch.tensor([1025, 1028, 1407, 2047], device="cuda")
    for buffer in manager.kv_buffer:
        buffer.zero_()
        buffer[source] = torch.arange(1, 5, device="cuda", dtype=torch.uint8)[:, None, None]
    layouts = manager.manager.get_cpu_cache_operator().get_page_layouts()
    pages = torch.empty(
        (1, manager.manager.get_cpu_cache_operator().get_page_bytes()), dtype=torch.uint8, pin_memory=True
    )
    for state in manager.full_states.values():
        state.write_to_cpu(source, pages, [0], layout=layouts[state.state_name])
        state.read_from_cpu(pages, [0], destination, layout=layouts[state.state_name])
    torch.cuda.synchronize()
    for buffer in manager.kv_buffer:
        torch.testing.assert_close(buffer[destination], buffer[source], rtol=0, atol=0)
        assert buffer[destination - 1].eq(0).all()
        untouched = torch.tensor([1024, 1026, 1406, 1408], device="cuda")
        assert buffer[untouched].eq(0).all()
    for buffer in manager.kv_buffer:
        buffer[destination] = 0
    # 同一 CPU 页只读前两个 token，不能覆盖目标中剩余槽位。
    for state in manager.full_states.values():
        state.read_from_cpu(pages, [0], destination[:2], layout=layouts[state.state_name])
    for buffer in manager.kv_buffer:
        torch.testing.assert_close(buffer[destination[:2]], buffer[source[:2]], rtol=0, atol=0)
        assert buffer[destination[2:]].eq(0).all()


def _packed_swa(buffer):
    return buffer.view(buffer.shape[0], -1, 74880) if buffer.shape[-1] == 585 else buffer


def _state_buffer(requests, kind):
    return getattr(requests, "req_to_" + kind + "_state").flatten(1, 2)


def _swa_blocks(requests, req_idx, blocks):
    return _swa_slots(requests, req_idx, blocks * 128) // 128


def _packed_geometry(buffer):
    # Independent reference for FlashMLA's four fixed packed formats.
    return {74880: (128, 576, 8), 37440: (64, 576, 8), 8448: (64, 128, 4), 1728: (2, 576, 8)}[buffer.shape[-1]]


def _read_packed_rows(buffer, layer_index, slots):
    buffer = _packed_swa(buffer)
    page_size, data_bytes, scale_bytes = _packed_geometry(buffer)
    slots = slots.reshape(-1).long()
    pages = (slots // page_size)[:, None]
    data_offsets = (slots % page_size) * data_bytes
    scale_offsets = page_size * data_bytes + (slots % page_size) * scale_bytes
    layer = buffer[layer_index]
    data = layer[pages, data_offsets[:, None] + torch.arange(data_bytes, device=slots.device)]
    scales = layer[pages, scale_offsets[:, None] + torch.arange(scale_bytes, device=slots.device)]
    return torch.cat((data, scales), dim=1)


def _write_packed_rows(buffer, layer_index, slots, packed):
    buffer = _packed_swa(buffer)
    page_size, data_bytes, scale_bytes = _packed_geometry(buffer)
    slots = slots.reshape(-1).long()
    packed = packed.reshape(-1, data_bytes + scale_bytes)
    pages = (slots // page_size)[:, None]
    data_offsets = (slots % page_size) * data_bytes
    scale_offsets = page_size * data_bytes + (slots % page_size) * scale_bytes
    layer = buffer[layer_index]
    layer[pages, data_offsets[:, None] + torch.arange(data_bytes, device=slots.device)] = packed[:, :data_bytes]
    layer[pages, scale_offsets[:, None] + torch.arange(scale_bytes, device=slots.device)] = packed[:, data_bytes:]


def _test_backend(requests, token_capacity=8192):
    from lightllm.common.basemodel.attention.nsa.dsv4_fp8_flashmla_sparse import DeepseekV4FlashMlaFp8SparseAttBackend

    model = type("TestModel", (), {})()
    model.req_manager = requests
    model.att_state_manager = requests.memory.manager
    model.layers_infer = [SimpleNamespace(layer_num_=i) for i in range(requests.cache_config.layer_num)]
    model.tp_rank_ = 0
    model.mtp_previous_draft_models = []
    model.__dict__["att_state_role"] = "main"
    model.config = {
        "num_attention_heads": 64,
        "compress_ratios": requests.cache_config.compress_ratios,
        "index_topk": 512,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "sliding_window": requests.sliding_window,
    }
    model.tp_world_size_ = 1
    model.max_req_num = requests.max_request_num
    model.batch_max_tokens = token_capacity
    model.max_seq_length = requests.req_to_token_indexs.shape[1]
    model.has_vision = False
    model.args = SimpleNamespace(
        run_mode="normal",
        enable_prefill_microbatch_overlap=True,
        enable_decode_microbatch_overlap=False,
        mtp_mode=None,
        mtp_step=0,
    )
    return DeepseekV4FlashMlaFp8SparseAttBackend(model)


def _prepare_prefill(requests, req_ids, starts, ends, microbatch_index=0):
    req_ids, starts, ends = (value.to(device="cuda", dtype=torch.int32) for value in (req_ids, starts, ends))
    if not hasattr(requests, "_test_backend"):
        requests._test_backend = _test_backend(requests)
        requests._test_prefill_states = {}
    state = requests._test_backend.create_att_prefill_state(None)
    state.prepare_swa_workspace(
        SimpleNamespace(
            req_manager=requests,
            swa_state=requests.swa_state,
            get_compressor_state=requests.get_compressor_state,
            get_compressed_kv_state=requests.get_compressed_kv_state,
            get_compressed_kv_view=requests.get_compressed_kv_view,
            b_req_idx=req_ids,
            b_ready_cache_len=starts,
            b_seq_len=ends,
            microbatch_index=microbatch_index,
        )
    )
    requests._test_prefill_states.update((req, state) for req in req_ids.tolist())
    if not hasattr(requests, "_test_prefill_ends"):
        requests._test_prefill_ends = {}
    requests._test_prefill_ends.update(zip(req_ids.tolist(), ends.tolist()))


def _prepare_swa(requests, req_idx, start, end, microbatch_index=0):
    _prepare_prefill(requests, torch.tensor([req_idx]), torch.tensor([start]), torch.tensor([end]), microbatch_index)


def _checkpoint_req(req_idx, length):
    return SimpleNamespace(req_idx=req_idx, get_chuncked_input_token_len=lambda: length)


def _finish_prefill(requests, req_ids):
    for req in req_ids:
        requests._test_prefill_states.pop(req, None)


def _prefill_state(requests, req_idx, prefill=True):
    if not prefill or not hasattr(requests, "_test_prefill_states") or not isinstance(req_idx, int):
        return None
    return requests._test_prefill_states.get(req_idx)


def _swa_slots(requests, req_idx, positions, *, prefill=True):
    return _get_swa_token_indices(
        requests, req_idx, positions, prefill_state=_prefill_state(requests, req_idx, prefill)
    )


def _swa_buffer(requests, req_idx, *, prefill=True):
    state = _prefill_state(requests, req_idx, prefill)
    return state.swa_buffer if state is not None else requests.req_to_swa.flatten(1, 2)


def _read_swa(requests, layer, req_idx, positions, *, prefill=True):
    return _read_packed_rows(
        _swa_buffer(requests, req_idx, prefill=prefill),
        layer,
        _swa_slots(requests, req_idx, positions, prefill=prefill),
    )


def _write_swa_rows(requests, layer, req_idx, positions, packed, *, prefill=True):
    _write_packed_rows(
        _swa_buffer(requests, req_idx, prefill=prefill),
        layer,
        _swa_slots(requests, req_idx, positions, prefill=prefill),
        packed,
    )


def _write_swa(requests, layer, req_idx, positions, packed):
    _write_swa_rows(requests, layer, req_idx, positions, packed)
    end = requests._test_prefill_ends.get(req_idx, int(positions.max()) + 1)
    keep = positions >= end - requests.swa_capacity_tokens
    _write_swa_rows(requests, layer, req_idx, positions[keep], packed[keep], prefill=False)


def _request_pages(requests, req):
    count = requests.swa_capacity_tokens // 128
    return range(req * count, (req + 1) * count)


def _reserve(manager, requests, length):
    req_idx = requests.alloc()
    held = (length + manager.page_size - 1) // manager.page_size * manager.page_size
    req = SimpleNamespace(req_idx=req_idx, hold_kv_len=held, cur_kv_len=0, shm_req=SimpleNamespace(input_len=held))
    attach_req_token_index(req, requests, requests.memory.allocator.alloc(held), mode="pin_mem_page_cache")
    # Publish the request table before kernels consume it; keep an owned snapshot
    # because callers may hold slots after the request row has been reused.
    slots = req.token_index.get_gpu_token_indexes(0, held).clone()
    return req_idx, slots


def test_non_aligned_chunks_preserve_window_and_checkpoint_history(cache):
    manager, requests = cache
    req, slots = _reserve(manager, requests, 3072)
    packed = torch.randint(0, 256, (3072, 584), dtype=torch.uint8, device="cuda")
    boundaries = (0, 127, 129, 257, 511, 777, 1025, 1537, 2051, 3072)
    for start, end in zip(boundaries, boundaries[1:]):
        _prepare_swa(requests, req, start, end)
        positions = torch.arange(start, end, device="cuda")
        _write_swa(requests, 0, req, positions, packed[start:end])
        # Check early and late queries, plus every checkpoint crossed by this chunk.
        reads = [(max(0, p - 127), p + 1) for p in (start, (start + end) // 2, end - 1)]
        reads += [(b - requests.sliding_window, b) for b in range((start // 256 + 1) * 256, end + 1, 256)]
        for left, right in reads:
            logical = torch.arange(left, right, device="cuda")
            torch.testing.assert_close(
                _read_swa(requests, 0, req, logical),
                packed[left:right],
                rtol=0,
                atol=0,
            )
        _finish_prefill(requests, [req])
    requests.free([req], slots)


@pytest.mark.parametrize("length", [256, 512, 2048])
def test_shared_compressed_kv_restores_private_continuation(cache, length):
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, length)
    _prepare_swa(requests, req_idx, 0, length)
    requests.req_to_swa.flatten(1, 2).random_(0, 256)
    _state_buffer(requests, "c4").uniform_()
    _state_buffer(requests, "c4_indexer").uniform_()
    states = requests.create_small_page_cache_manager(1)
    state_idx = states.alloc_one_state_cache()
    requests.save_state(_checkpoint_req(req_idx, length), state_idx, states)
    torch.cuda.synchronize()
    expected_swa, expected_c4, expected_indexer = [value.cuda() for value in states.get_state_cache(state_idx)]
    _finish_prefill(requests, [req_idx])
    requests.free_req(req_idx)
    reqs = [requests.alloc(), requests.alloc()]
    for req in reqs:
        requests.req_to_token_indexs[req, :length] = slots
        requests.restore_state(SimpleNamespace(req_idx=req, cur_kv_len=length), states, state_idx)
        positions = torch.arange(length - requests.sliding_window, length, device="cuda")
        actual = torch.stack([_read_swa(requests, layer, req, positions) for layer in range(3)])
        torch.testing.assert_close(actual, expected_swa)
        state_slots = torch.arange(length - 4, length, device="cuda") % requests.c4_state_tokens
        torch.testing.assert_close(requests.req_to_c4_state[:, req, state_slots], expected_c4)
        torch.testing.assert_close(requests.req_to_c4_indexer_state[:, req, state_slots], expected_indexer)
    assert set(_request_pages(requests, reqs[0])).isdisjoint(_request_pages(requests, reqs[1]))
    # Advancing one fork releases only its own window; the shared token page survives.
    _prepare_swa(requests, reqs[0], length + 512, length + 768)
    positions = torch.arange(length - requests.sliding_window, length, device="cuda")
    actual = torch.stack([_read_swa(requests, layer, reqs[1], positions) for layer in range(3)])
    torch.testing.assert_close(actual, expected_swa)
    requests.free(reqs, slots)
    states.free_state_cache([state_idx])
    assert manager.allocator.can_use_mem_size == manager.size
    assert states.get_free_cache_num() == 1


@pytest.mark.parametrize("cache", [(page, step) for page in (256, 512, 768, 1024) for step in (0, 3)], indirect=True)
@pytest.mark.parametrize("length", [3, 127, 128, 255, 256, 257, 511, 512, 513, 2051])
def test_pd_kv_and_att_state_roundtrip(cache, length):
    import copy

    manager, requests = cache
    prefill = DeepseekV4ReqManager(2, 4096, None, manager.cache_config, mtp_step=get_env_start_args().mtp_step)
    prefill.bind_mem_manager(manager)
    source, src = _reserve(manager, prefill, length)
    destination, dst = _reserve(manager, requests, length)
    _randomize_compressed_kv(manager)
    prefill.req_to_swa.flatten(1, 2).random_(0, 256)
    for buffer in (_state_buffer(prefill, "c4"), _state_buffer(prefill, "c4_indexer"), _state_buffer(prefill, "c128")):
        buffer.uniform_()
    source_memory = copy.copy(manager)
    for memory, runtime in ((source_memory, prefill), (manager, requests)):
        for name in (
            "req_to_swa",
            "req_to_c4_state",
            "req_to_c4_indexer_state",
            "req_to_c128_state",
        ):
            setattr(memory, name, getattr(runtime, name))
        memory.state_token_len = length
        memory.alloc_paged_kv_move_buffer(1, manager.page_size)
    for start in range(0, length, manager.page_size):
        end = min(length, start + manager.page_size)
        source_memory.write_mem_to_page_kv_move_buffer(src[start:end].tolist(), 0, 0, [source_memory], 1)
        manager.kv_move_buffer.copy_(source_memory.kv_move_buffer)
        manager.read_page_kv_move_buffer_to_mem(dst[start:end].tolist(), 0, 0, [manager], 1)
    for buffer, ratio in zip(_compressed_kv_pools(manager), (4, 4, 128)):
        torch.testing.assert_close(
            _read_packed_rows(buffer, 0, src[ratio - 1 : length : ratio].long() // ratio),
            _read_packed_rows(buffer, 0, dst[ratio - 1 : length : ratio].long() // ratio),
            rtol=0,
            atol=0,
        )

    # PD 传递与 MTP 容量无关的逻辑窗口，不再传递无效候选槽位或 padding。
    source_memory.write_mem_to_page_kv_move_buffer([], 0, 0, [source_memory], 1, "att_state", source)
    manager.kv_move_buffer.copy_(source_memory.kv_move_buffer)
    manager.read_page_kv_move_buffer_to_mem([], 0, 0, [manager], 1, "att_state", destination)
    positions = torch.arange(max(0, length - requests.sliding_window), length, device="cuda")
    for layer in range(requests.cache_config.layer_num):
        torch.testing.assert_close(
            _read_swa(requests, layer, destination, positions, prefill=False),
            _read_swa(prefill, layer, source, positions, prefill=False),
            rtol=0,
            atol=0,
        )
    ring_size = requests.swa_capacity_tokens
    assert requests.req_to_swa.flatten(1, 2)[:, : destination * ring_size].eq(0).all()
    assert requests.req_to_swa.flatten(1, 2)[:, (destination + 1) * ring_size :].eq(0).all()
    for source_buffer, target_buffer, width in (
        (prefill.req_to_c4_state, requests.req_to_c4_state, 8),
        (prefill.req_to_c4_indexer_state, requests.req_to_c4_indexer_state, 8),
        (prefill.req_to_c128_state, requests.req_to_c128_state, 128),
    ):
        positions = torch.arange(max(0, length - width), length, device="cuda")
        torch.testing.assert_close(
            source_buffer[:, source, positions % source_buffer.shape[2]],
            target_buffer[:, destination, positions % target_buffer.shape[2]],
            rtol=0,
            atol=0,
        )
    prefill.free([source], src)
    requests.free([destination], dst)
    assert manager.allocator.can_use_mem_size == manager.size
