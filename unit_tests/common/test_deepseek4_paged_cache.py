import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
from lightllm.common.state_cache_manager.deepseek4 import (
    DeepseekV4CacheConfig,
)
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.models.deepseek_v4.triton_kernel.build_compress_index_dsv4 import build_compress_index
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.deepseek_v4.triton_kernel.cache_io import get_compressed_kv_buffer, get_indexer_k_buffer


def _history_pools(manager):
    c4 = next(iter(manager.cache_config.layer_to_c4_idx))
    c128 = next(iter(manager.cache_config.layer_to_c128_idx))
    return (
        get_compressed_kv_buffer(manager, c4).unsqueeze(0),
        get_indexer_k_buffer(manager, c4).unsqueeze(0),
        get_compressed_kv_buffer(manager, c128).unsqueeze(0),
    )


def _randomize_history(manager):
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
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_dsv4_paged_cache")
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "page_size": page_size,
                "run_mode": "normal",
                "linear_att_hash_page_size": page_size,
                "linear_att_page_block_num": 8,
                "chunked_prefill_size": 2048,
                "max_req_total_len": 4096,
                "disable_chunked_prefill": False,
                "model_dir": str(tmp_path),
                "penalty_counter_mode": "cpu_counter",
                "output_constraint_mode": "none",
                "mtp_step": mtp_step,
                "mtp_dynamic_verify": False,
                "enable_ep_moe": False,
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
        DeepseekV4CacheConfig([4, 128, 0], (2048 + page_size - 1) // page_size * page_size),
    )
    requests = DeepseekV4ReqManager(
        2, 4096, None, manager.cache_config, max_forward_tokens=8192, mtp_step=mtp_step, prefill_regions=2
    )
    requests.bind_mem_manager(manager)
    requests.prefill_c4_checkpoints.zero_()
    requests.prefill_indexer_checkpoints.zero_()
    yield manager, requests
    torch.cuda.synchronize()
    get_env_start_args.cache_clear()


def test_attention_input_params_select_token_buffers(cache):
    manager, requests = cache
    for layer_index, slot in ((0, 0), (1, 1), (2, None)):
        swa, history = requests.get_att_input_params(layer_index)
        assert swa.data_ptr() == requests.swa_buffer[layer_index].data_ptr()
        if slot is None:
            assert history is None
        else:
            token_buffer = manager.get_att_input_params(layer_index)
            assert token_buffer is manager.kv_buffer[slot]
            assert token_buffer.shape[:2] == (manager.size + manager.page_size, 1)
            assert history.untyped_storage().data_ptr() == token_buffer.data_ptr()


def test_big_page_pool_reserves_cpu_cache_staging_slots(cache):
    manager, _ = cache
    pool = manager.big_page_buffers
    args = get_env_start_args()
    page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
    capacity = (manager.size + page_tokens - 1) // page_tokens
    assert pool.size == capacity + 2 and pool.keep_num == 2
    assert manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID == capacity
    assert manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID == capacity + 1
    slots = pool.alloc_state_cache(capacity)
    assert slots == list(range(capacity))
    assert pool.alloc_one_state_cache() is None
    pool.free_state_cache(slots)
    assert pool.get_free_cache_num() == capacity


def test_copy_mem_to_mem_preserves_token_slot_indices(cache):
    manager, _ = cache
    source = torch.tensor([3, 257, 509], device="cuda")
    destination = torch.tensor([1025, 1028, 1407], device="cuda")
    for buffer in manager.kv_buffer:
        buffer.zero_()
        buffer[source] = torch.arange(1, 4, device="cuda", dtype=torch.uint8)[:, None, None]
    manager.operator.copy_mem_to_mem(source, destination)
    for buffer in manager.kv_buffer:
        torch.testing.assert_close(buffer[destination], buffer[source], rtol=0, atol=0)


def test_cache_io_copies_individual_token_slots(cache):
    from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_history_pages

    manager, _ = cache
    source = torch.tensor([3, 17, 257, 509], device="cuda")
    destination = torch.tensor([1025, 1028, 1407, 2047], device="cuda")
    for buffer in manager.kv_buffer:
        buffer.zero_()
        buffer[source] = torch.arange(1, 5, device="cuda", dtype=torch.uint8)[:, None, None]
    pages = torch.empty((1, 4 * manager.get_cell_size()), dtype=torch.uint8, pin_memory=True)
    page_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    copy_history_pages(manager, source, page_ids, pages, to_pages=True, page_tokens=4)
    copy_history_pages(manager, destination, page_ids, pages, to_pages=False, page_tokens=4)
    torch.cuda.synchronize()
    for buffer in manager.kv_buffer:
        torch.testing.assert_close(buffer[destination], buffer[source], rtol=0, atol=0)
        assert buffer[destination - 1].eq(0).all()
        untouched = torch.tensor([1024, 1026, 1406, 1408], device="cuda")
        assert buffer[untouched].eq(0).all()
    snapshot = manager.get_index_kv_buffer(source)
    assert all(isinstance(value, torch.Tensor) for value in snapshot.values())
    # The shared preload path truncates dictionary values along their token axis.
    snapshot = {name: value[:, :2].cpu() for name, value in snapshot.items()}
    for buffer in manager.kv_buffer:
        buffer[destination] = 0
    manager.load_index_kv_buffer(destination[:2], snapshot)
    for buffer in manager.kv_buffer:
        torch.testing.assert_close(buffer[destination[:2]], buffer[source[:2]], rtol=0, atol=0)
        assert buffer[destination[2:]].eq(0).all()


def test_request_runtime_is_independent_of_history_capacity(cache, monkeypatch):
    from lightllm.common.kv_cache_mem_manager import MemoryManager

    original, requests = cache
    observed = []

    def profile(manager, mem_fraction):
        observed.append(requests.swa_buffer.nbytes)
        assert _state_buffer(requests, "c4").is_cuda and _state_buffer(requests, "c128").is_cuda
        assert not hasattr(manager, "swa_buffer")
        manager.size = 8192

    monkeypatch.setattr(MemoryManager, "profile_size", profile)
    manager = DeepseekV4MemoryManager(None, torch.bfloat16, 1, 512, 3, original.cache_config)
    assert observed == [requests.swa_buffer.nbytes]
    assert manager.size == 8192 - manager.page_size
    for buffer in manager.kv_buffer:
        assert buffer.shape[0] == manager.size + manager.page_size
    assert sum(buffer.nbytes for buffer in manager.kv_buffer) == original.get_cell_size() * 8192


@pytest.mark.parametrize("page_size", [256, 1024, 4096])
def test_pd_memory_reservation_matches_allocated_bytes(cache, monkeypatch, page_size):
    manager, _ = cache
    args = get_env_start_args()
    monkeypatch.setattr(args, "run_mode", "decode")
    monkeypatch.setattr(args, "pd_kv_page_num", 3, raising=False)
    monkeypatch.setattr(args, "pd_kv_page_size", page_size, raising=False)
    reserved = manager.get_pd_kv_move_buffer_size()
    buffer = manager.alloc_paged_kv_move_buffer(3, page_size)
    assert reserved == buffer.nbytes
    assert buffer.shape == manager.get_paged_kv_move_buffer_shape(3, page_size)


def _packed_swa(buffer):
    return buffer.view(buffer.shape[0], -1, 74880) if buffer.shape[-1] == 585 else buffer


def _state_buffer(requests, kind):
    return getattr(requests, "req_to_" + kind + "_state").flatten(1, 2)


def _swa_blocks(requests, req_idx, blocks):
    return requests.get_swa_token_indices(req_idx, blocks * 128) // 128


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


def _prepare_prefill(requests, req_ids, starts, ends, microbatch_index=0):
    req_ids, starts, ends = (value.to(device="cuda", dtype=torch.int32) for value in (req_ids, starts, ends))
    q_lens = ends - starts
    requests.init_forward_state(
        SimpleNamespace(
            is_draft_model=False,
            is_prefill=True,
            microbatch_index=microbatch_index,
            b_req_idx=req_ids,
            b_ready_cache_len=starts,
            b_seq_len=ends,
            b_q_start_loc=q_lens.cumsum(0, dtype=torch.int32) - q_lens,
        )
    )
    if not hasattr(requests, "_test_prefill_ends"):
        requests._test_prefill_ends = {}
    requests._test_prefill_ends.update(zip(req_ids.tolist(), ends.tolist()))


def _prepare_swa(requests, req_idx, start, end, microbatch_index=0):
    _prepare_prefill(requests, torch.tensor([req_idx]), torch.tensor([start]), torch.tensor([end]), microbatch_index)


def _finish_prefill(requests, req_ids):
    for req in req_ids:
        requests.clear_runtime_state(req)


def _write_swa(requests, layer, req_idx, positions, packed):
    slots = requests.get_swa_token_indices(req_idx, positions)
    _write_packed_rows(requests.swa_buffer, layer, slots, packed)
    end = requests._test_prefill_ends.get(req_idx, int(positions.max()) + 1)
    retain = requests.swa_capacity_tokens
    keep = positions >= end - retain
    _write_packed_rows(
        requests.swa_buffer,
        layer,
        requests.get_swa_token_indices(req_idx, positions[keep], prefill=False),
        packed[keep],
    )


def _request_pages(requests, req):
    count = requests.swa_capacity_tokens // 128
    return range(req * count, (req + 1) * count)


def _reserve(manager, requests, length):
    req_idx = requests.alloc()
    held = (length + manager.page_size - 1) // manager.page_size * manager.page_size
    slots = manager.alloc(held).cuda()
    requests.req_to_token_indexs[req_idx, :held] = slots
    return req_idx, slots


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cache", [256, 512, 768, 1024], indirect=True)
def test_noncontiguous_pages_obey_logical_compression_boundaries(cache, ratio):
    manager, requests = cache
    page_size = manager.page_size
    req_idx, allocated = _reserve(manager, requests, 3 * page_size)
    slots = allocated.view(3, page_size)[torch.tensor([2, 0, 1], device="cuda")].reshape(-1)
    requests.req_to_token_indexs[req_idx, : slots.numel()] = slots
    positions = torch.tensor(
        [0, 2, 3, 126, 127, 254, 255, page_size - 1, page_size, 2 * page_size - 1, 2 * page_size], device="cuda"
    )
    indexes = torch.empty((len(positions), slots.numel() // ratio), dtype=torch.int32, device="cuda")
    lengths = torch.empty_like(positions, dtype=torch.int32)
    build_compress_index(
        torch.full_like(positions, req_idx), positions, requests.req_to_token_indexs, ratio, indexes, lengths
    )
    group_slots = slots[ratio - 1 :: ratio] // ratio
    for row, position in enumerate(positions.tolist()):
        count = (position + 1) // ratio
        torch.testing.assert_close(indexes[row, :count], group_slots[:count])
        assert indexes[row, count:].eq(-1).all()
        assert lengths[row].item() == max(1, count)
    manager.free(allocated)
    assert manager.allocator.can_use_mem_size == manager.size


def test_indexer_writer_only_writes_closed_groups(cache):
    manager, requests = cache
    _, slots = _reserve(manager, requests, 512)
    slots = slots.view(2, 256).flip(0).reshape(-1)
    keys = torch.randn((259, 128), dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(259, dtype=torch.int32, device="cuda")
    _history_pools(manager)[1].zero_()
    from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k_dsv4 import destindex_copy_indexer_k_dsv4

    destindex_copy_indexer_k_dsv4(keys, slots[:259], positions, get_indexer_k_buffer(manager, 0), 64)
    closed = torch.arange(3, 259, 4, device="cuda")
    actual = _read_packed_rows(_history_pools(manager)[1], 0, slots[closed] // 4)
    # Reference quantization is independent of the cache writer and stays in the test.
    selected_keys = keys[closed].float()
    scale = selected_keys.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    quantized = (selected_keys / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    expected = torch.cat((quantized.view(torch.uint8), scale.view(torch.uint8)), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Positions 256..258 cannot publish their reserved next compressed slot.
    assert _read_packed_rows(_history_pools(manager)[1], 0, slots[256:257] // 4).eq(0).all()


def test_speculative_retry_reuses_swa_without_freeing_token_pages(cache):
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, 512)

    def cpu(data):
        return torch.tensor(data, dtype=torch.int32)

    _prepare_prefill(requests, cpu([req_idx]), cpu([0]), cpu([254]))
    for sequences in ([255, 256, 257, 258], [255, 256, 257, 258], [256, 257, 258, 259]):
        swa_slots = requests.get_swa_token_indices(req_idx, torch.tensor(sequences, device="cuda") - 1)
        assert swa_slots.unique().numel() == len(sequences)
        assert manager.allocator.can_use_mem_size == manager.size - 512
    requests.free([req_idx], slots)
    assert manager.allocator.can_use_mem_size == manager.size


@pytest.mark.parametrize("model_kind", ["target", "mtp", "dspark"])
def test_model_selects_held_slots_without_allocating_swa(cache, model_kind):
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
    from lightllm.models.deepseek_v4_mtp.model import DeepseekV4MTPModel
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, 512)
    _prepare_swa(requests, req_idx, 0, 256)
    model_cls = {"target": DeepseekV4TpPartModel, "mtp": DeepseekV4MTPModel, "dspark": DeepseekV4DSparkModel}[
        model_kind
    ]
    model = model_cls.__new__(model_cls)
    model.req_manager = requests
    model_input = ModelInput(
        batch_size=1,
        total_token_num=257,
        max_q_seq_len=1,
        max_kv_seq_len=257,
        b_req_idx=torch.tensor([req_idx], dtype=torch.int32),
        b_seq_len=torch.tensor([257], dtype=torch.int32),
        b_mtp_index=torch.zeros(1, dtype=torch.int32),
        b_position_delta=torch.zeros(1, dtype=torch.int32),
        b_shared_seq_len=torch.zeros(1, dtype=torch.int32),
        b_shared_radix_node_id=torch.full((1,), -1, dtype=torch.int64),
        multimodal_params=[{"images": [], "audios": []}],
    )
    model_input.to_cuda()
    for _ in range(2):
        torch.testing.assert_close(model._select_mem_indexes(model_input), slots[256:257])
        assert manager.allocator.can_use_mem_size == manager.size - 512
    requests.free([req_idx], slots)


@pytest.mark.parametrize("cache", [(256, 0), (256, 5)], indirect=True)
def test_prefill_chunk_keeps_all_new_swa_rows(cache):
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, 2048)
    _prepare_prefill(requests, torch.tensor([req_idx]), torch.tensor([0]), torch.tensor([2048]))
    positions = torch.arange(2048, device="cuda")
    swa_slots = requests.get_swa_token_indices(req_idx, positions)
    assert swa_slots.unique().numel() == 2048
    packed = torch.randint(0, 256, (2048, 584), dtype=torch.uint8, device="cuda")
    _write_swa(requests, 0, req_idx, positions, packed)
    # Checkpoints can still read any boundary of the complete chunk here.
    torch.testing.assert_close(_read_packed_rows(requests.swa_buffer, 0, swa_slots), packed)
    _finish_prefill(requests, [req_idx])
    tail = positions[-requests.sliding_window :]
    torch.testing.assert_close(
        _read_packed_rows(requests.swa_buffer, 0, requests.get_swa_token_indices(req_idx, tail)), packed[tail]
    )
    requests.free([req_idx], slots)


def test_prefill_checkpoint_reads_precede_window_retention(cache):
    """Intermediate checkpoints and late draft KV must survive a long prefill."""
    manager, requests = cache
    req, slots = _reserve(manager, requests, 2048)
    _prepare_swa(requests, req, 0, 2048)
    positions = torch.arange(2048, device="cuda")
    packed = torch.randint(0, 256, (2048, 584), dtype=torch.uint8, device="cuda")
    _write_swa(requests, 0, req, positions, packed)
    _state_buffer(requests, "c4").uniform_()
    _state_buffer(requests, "c4_indexer").uniform_()
    live_tail = requests.req_to_c4_state.clone()
    states = requests.create_small_page_cache_manager(1)
    index = states.alloc_one_state_cache()
    for boundary in range(256, 2049, 256):
        requests.save_state(req, index, states, boundary)
        window, _, _ = states.get_state_cache(index)
        torch.testing.assert_close(
            window[0].cuda(), packed[boundary - requests.sliding_window : boundary], rtol=0, atol=0
        )
        torch.testing.assert_close(
            _read_packed_rows(
                requests.swa_buffer,
                0,
                requests.get_swa_token_indices(req, positions[boundary - requests.sliding_window : boundary]),
            ),
            packed[boundary - requests.sliding_window : boundary],
            rtol=0,
            atol=0,
        )

    # A DSpark context commit writes KV after the target model has returned.
    _write_swa(requests, 2, req, positions, packed.flip(0))
    _finish_prefill(requests, [req])
    # Reuse the scratch: only fixed request windows and CPU snapshots remain.
    requests.swa_buffer[:, requests.swa_runtime_tokens :].zero_()
    tail = positions[-requests.sliding_window :]
    for layer, expected in ((0, packed), (2, packed.flip(0))):
        torch.testing.assert_close(
            _read_packed_rows(requests.swa_buffer, layer, requests.get_swa_token_indices(req, tail)),
            expected[tail],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(requests.req_to_c4_state, live_tail)
    requests.free([req], slots)
    states.free_state_cache([index])


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
                _read_packed_rows(requests.swa_buffer, 0, requests.get_swa_token_indices(req, logical)),
                packed[left:right],
                rtol=0,
                atol=0,
            )
        _finish_prefill(requests, [req])
    requests.free([req], slots)


@pytest.mark.parametrize("cache", [256, 512, 768, 1024], indirect=True)
def test_history_copy_preserves_packed_data_and_scale_regions(cache):
    manager, requests = cache
    source, src = _reserve(manager, requests, 2 * manager.page_size)
    destination, dst = _reserve(manager, requests, 2 * manager.page_size)
    swa_mapping = requests.prefill_token_ranges.clone()
    src = src.view(2, manager.page_size).flip(0).reshape(-1)
    for pool in _history_pools(manager):
        pool.random_(0, 256)
    manager.operator.copy_mem_to_mem(src_mem_index=src, dst_mem_index=dst)
    for pool in _history_pools(manager):
        torch.testing.assert_close(pool[:, src[::256].long() // 256], pool[:, dst[::256].long() // 256], rtol=0, atol=0)
    torch.testing.assert_close(requests.prefill_token_ranges, swa_mapping)
    requests.free([source, destination], torch.cat([src, dst]))


@pytest.mark.parametrize("cache", [256, 512, 768, 1024], indirect=True)
def test_hold_page_is_separate_from_request_rings(cache):
    manager, requests = cache
    hold = requests.req_to_token_indexs[requests.HOLD_REQUEST_ID]
    torch.testing.assert_close(
        hold[: manager.page_size],
        torch.arange(manager.size, manager.size + manager.page_size, dtype=torch.int32, device="cuda"),
    )
    assert (
        _swa_blocks(requests, requests.HOLD_REQUEST_ID, torch.arange(32, device="cuda"))
        .eq(requests.HOLD_SWA_SLOT // 128)
        .all()
    )
    assert (hold // 4).lt(_history_pools(manager)[0].shape[1] * 64).all()
    assert (hold // 128).lt(_history_pools(manager)[2].shape[1] * 2).all()
    free_tokens = manager.allocator.can_use_mem_size
    reqs = [requests.alloc(), requests.alloc()]
    assert set(_request_pages(requests, reqs[0])).isdisjoint(_request_pages(requests, reqs[1]))
    assert (
        _swa_blocks(requests, torch.tensor(reqs, device="cuda")[:, None], torch.arange(32, device="cuda")[None, :])
        .lt(requests.HOLD_SWA_SLOT // 128)
        .all()
    )
    assert manager.allocator.can_use_mem_size == free_tokens
    for req in reqs:
        requests.free_req(req)


def test_swa_checkpoint_and_prefix_copy_use_64bit_layer_offsets(cache):
    from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager
    from lightllm.models.deepseek_v4.triton_kernel.swa_cache import prepare_prefill_workspace

    config = DeepseekV4CacheConfig((0,) * 40)
    requests = DeepseekV4ReqManager(256, 2048, None, config, max_forward_tokens=8208)
    assert requests.swa_buffer.numel() > 2 ** 31
    for layer in range(config.layer_num):
        requests.swa_buffer[layer, :128].fill_(layer + 1)
    checkpoints = DeepseekV4StateCacheManager(1, config)
    requests.save_state(0, 0, checkpoints, checkpoint_len=512)
    torch.cuda.synchronize()
    expected = torch.arange(1, 41, dtype=torch.uint8).view(40, 1, 1).expand(40, 128, config.kv_bytes)
    torch.testing.assert_close(checkpoints.get_state_cache(0)[0], expected, rtol=0, atol=0)

    requests.restore_state(SimpleNamespace(req_idx=1, cur_kv_len=512), checkpoints, 0)
    prepare_prefill_workspace(
        requests,
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        torch.tensor([512], dtype=torch.int32, device="cuda"),
        torch.tensor([768], dtype=torch.int32, device="cuda"),
        microbatch_index=0,
    )
    checkpoints.buffer.zero_()
    requests.save_state(1, 0, checkpoints, checkpoint_len=512)
    torch.cuda.synchronize()
    torch.testing.assert_close(checkpoints.get_state_cache(0)[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [256, 512, 2048])
def test_shared_history_restores_private_continuation(cache, length):
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, length)
    _prepare_swa(requests, req_idx, 0, length)
    requests.swa_buffer.random_(0, 256)
    _state_buffer(requests, "c4").uniform_()
    _state_buffer(requests, "c4_indexer").uniform_()
    states = requests.create_small_page_cache_manager(1)
    state_idx = states.alloc_one_state_cache()
    requests.save_state(req_idx, state_idx, states, checkpoint_len=length)
    torch.cuda.synchronize()
    expected_swa, expected_c4, expected_indexer = [value.cuda() for value in states.get_state_cache(state_idx)]
    requests.free_req(req_idx)
    reqs = [requests.alloc(), requests.alloc()]
    for req in reqs:
        requests.req_to_token_indexs[req, :length] = slots
        requests.restore_state(SimpleNamespace(req_idx=req, cur_kv_len=length), states, state_idx)
        positions = torch.arange(length - requests.sliding_window, length, device="cuda")
        actual = torch.stack(
            [
                _read_packed_rows(requests.swa_buffer, layer, requests.get_swa_token_indices(req, positions))
                for layer in range(3)
            ]
        )
        torch.testing.assert_close(actual, expected_swa)
        state_slots = torch.arange(length - 4, length, device="cuda") % requests.c4_state_tokens
        torch.testing.assert_close(requests.req_to_c4_state[:, req, state_slots], expected_c4)
        torch.testing.assert_close(requests.req_to_c4_indexer_state[:, req, state_slots], expected_indexer)
    assert set(_request_pages(requests, reqs[0])).isdisjoint(_request_pages(requests, reqs[1]))
    # Advancing one fork releases only its own window; the shared token page survives.
    _prepare_swa(requests, reqs[0], length + 512, length + 768)
    positions = torch.arange(length - requests.sliding_window, length, device="cuda")
    actual = torch.stack(
        [
            _read_packed_rows(requests.swa_buffer, layer, requests.get_swa_token_indices(reqs[1], positions))
            for layer in range(3)
        ]
    )
    torch.testing.assert_close(actual, expected_swa)
    requests.free(reqs, slots)
    states.free_state_cache([state_idx])
    assert manager.allocator.can_use_mem_size == manager.size
    assert states.get_free_cache_num() == 1


@pytest.mark.parametrize("cache", [256, 512, 768, 1024], indirect=True)
@pytest.mark.parametrize("length", [3, 127, 128, 255, 256, 257, 511, 512, 513, 2051])
@pytest.mark.parametrize("prefill_mtp", [0, 3])
def test_pd_kv_and_att_state_roundtrip_with_different_rings(cache, length, prefill_mtp):
    import copy
    from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import create_cache_page_io

    manager, requests = cache
    prefill = DeepseekV4ReqManager(2, 4096, None, manager.cache_config, max_forward_tokens=8192, mtp_step=prefill_mtp)
    prefill.bind_mem_manager(manager)
    source, src = _reserve(manager, prefill, length)
    destination, dst = _reserve(manager, requests, length)
    _randomize_history(manager)
    prefill.swa_buffer.random_(0, 256)
    for buffer in (_state_buffer(prefill, "c4"), _state_buffer(prefill, "c4_indexer"), _state_buffer(prefill, "c128")):
        buffer.uniform_()
    source_memory = copy.copy(manager)
    for memory, runtime in ((source_memory, prefill), (manager, requests)):
        for name in (
            "swa_buffer",
            "swa_capacity_tokens",
            "HOLD_REQUEST_ID",
            "HOLD_SWA_SLOT",
            "req_to_c4_state",
            "req_to_c4_indexer_state",
            "req_to_c128_state",
            "req_to_token_indexs",
        ):
            setattr(memory, name, getattr(runtime, name))
        memory.alloc_paged_kv_move_buffer(1, manager.page_size)
    source_io, destination_io = create_cache_page_io([source_memory], 0, 1), create_cache_page_io([manager], 0, 1)
    task = SimpleNamespace(
        src_page_index=0, dst_page_index=0, prefill_dp_index=0, decode_dp_index=0, req_idx=source, page_kind="kv"
    )
    for start in range(0, length, manager.page_size):
        end = min(length, start + manager.page_size)
        task.mem_indexes = src[start:end].tolist()
        task.start_kv_index = start
        moved = source_io.pack(task)
        assert 0 < moved <= source_memory.kv_move_buffer[0].numel()
        manager.kv_move_buffer.copy_(source_memory.kv_move_buffer)
        task.mem_indexes = dst[start:end].tolist()
        destination_io.unpack(task)
    for buffer, ratio in zip(_history_pools(manager), (4, 4, 128)):
        torch.testing.assert_close(
            _read_packed_rows(buffer, 0, src[ratio - 1 : length : ratio].long() // ratio),
            _read_packed_rows(buffer, 0, dst[ratio - 1 : length : ratio].long() // ratio),
            rtol=0,
            atol=0,
        )

    # The final att_state task carries the window and only the unfinished compressor tails.
    task.mem_indexes, task.page_kind, task.start_kv_index = [], "att_state", length
    assert 0 < source_io.pack(task) <= source_memory.kv_move_buffer[0].numel()
    manager.kv_move_buffer.copy_(source_memory.kv_move_buffer)
    task.req_idx = destination
    destination_io.unpack(task)
    positions = torch.arange(max(0, length - requests.sliding_window), length, device="cuda")
    for layer in range(manager.layer_num):
        torch.testing.assert_close(
            _read_packed_rows(prefill.swa_buffer, layer, prefill.get_swa_token_indices(source, positions)),
            _read_packed_rows(requests.swa_buffer, layer, requests.get_swa_token_indices(destination, positions)),
            rtol=0,
            atol=0,
        )
    for source_buffer, target_buffer, source_ring, target_ring, first in (
        (
            _state_buffer(prefill, "c4"),
            _state_buffer(requests, "c4"),
            prefill.c4_state_tokens,
            requests.c4_state_tokens,
            max(0, length - 4 - length % 4),
        ),
        (
            _state_buffer(prefill, "c4_indexer"),
            _state_buffer(requests, "c4_indexer"),
            prefill.c4_state_tokens,
            requests.c4_state_tokens,
            max(0, length - 4 - length % 4),
        ),
        (
            _state_buffer(prefill, "c128"),
            _state_buffer(requests, "c128"),
            prefill.c128_state_tokens,
            requests.c128_state_tokens,
            length - length % 128,
        ),
    ):
        positions = torch.arange(first, length, device="cuda")
        torch.testing.assert_close(
            source_buffer[:, source * source_ring + positions % source_ring],
            target_buffer[:, destination * target_ring + positions % target_ring],
            rtol=0,
            atol=0,
        )
    prefill.free([source], src)
    requests.free([destination], dst)
    assert manager.allocator.can_use_mem_size == manager.size


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("length", [257, 2051])
def test_pd_transport_completion_covers_peer_gpu_state(cache, length):
    import ctypes
    from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import create_cache_page_io

    transport, _ = cache
    with torch.cuda.device(1):
        peer = DeepseekV4MemoryManager(8192, torch.bfloat16, 1, 512, 3, transport.cache_config)
        requests = DeepseekV4ReqManager(2, 4096, None, peer.cache_config, max_forward_tokens=4096, mtp_step=0)
        requests.bind_mem_manager(peer)
        source, src = _reserve(peer, requests, length)
        destination, dst = _reserve(peer, requests, length)
        _randomize_history(peer)
        requests.swa_buffer.random_(0, 256)
        for buffer in (
            _state_buffer(requests, "c4"),
            _state_buffer(requests, "c4_indexer"),
            _state_buffer(requests, "c128"),
        ):
            buffer.uniform_()
        for name in (
            "swa_buffer",
            "swa_capacity_tokens",
            "HOLD_REQUEST_ID",
            "HOLD_SWA_SLOT",
            "req_to_c4_state",
            "req_to_c4_indexer_state",
            "req_to_c128_state",
            "req_to_token_indexs",
        ):
            setattr(peer, name, getattr(requests, name))
        torch.cuda.synchronize()
        peer_stream = torch.cuda.Stream()
    # CUDA IPC enables peer access in production. A same-process test needs
    # to establish that access before Triton dereferences the peer pointer.
    cudart = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
    assert cudart.cudaDeviceEnablePeerAccess(1, 0) in (0, 704)
    cudart.cudaGetLastError()
    transport.alloc_paged_kv_move_buffer(1, transport.page_size)
    page_io = create_cache_page_io([transport, peer], 0, 1)
    task = SimpleNamespace(
        src_page_index=0, dst_page_index=0, prefill_dp_index=1, decode_dp_index=1, req_idx=source, page_kind="kv"
    )
    for start in range(0, length, transport.page_size):
        task.mem_indexes = src[start : start + transport.page_size].tolist()
        page_io.pack(task)
        torch.cuda.current_stream().synchronize()  # The network consumes the completed packed page.
        task.mem_indexes = dst[start : start + transport.page_size].tolist()
        page_io.unpack(task)
        torch.cuda.current_stream().synchronize()  # Return the page to the pool only after completion.
    task.mem_indexes, task.page_kind, task.start_kv_index = [], "att_state", length
    with torch.cuda.stream(peer_stream), torch.cuda.device(0):
        page_io.pack(task)
        with torch.cuda.device(1):
            torch.cuda._sleep(20_000_000)
        task.req_idx = destination
        page_io.unpack(task)
        completed = torch.cuda.Event()
        completed.record()
        completed.synchronize()
        assert peer_stream.query(), "transport completion must include the peer's index_copy kernels"
    with torch.cuda.device(1):
        for buffer in peer.kv_buffer:
            torch.testing.assert_close(buffer[src.long()], buffer[dst.long()], rtol=0, atol=0)
        positions = torch.arange(max(0, length - requests.sliding_window), length, device="cuda")
        torch.testing.assert_close(
            _read_packed_rows(requests.swa_buffer, 0, requests.get_swa_token_indices(source, positions)),
            _read_packed_rows(requests.swa_buffer, 0, requests.get_swa_token_indices(destination, positions)),
            rtol=0,
            atol=0,
        )
        for buffer, ring, first in (
            (_state_buffer(requests, "c4"), requests.c4_state_tokens, length - 4 - length % 4),
            (_state_buffer(requests, "c4_indexer"), requests.c4_state_tokens, length - 4 - length % 4),
            (_state_buffer(requests, "c128"), requests.c128_state_tokens, length - length % 128),
        ):
            positions = torch.arange(first, length, device="cuda") % ring
            torch.testing.assert_close(
                buffer[:, source * ring + positions], buffer[:, destination * ring + positions], rtol=0, atol=0
            )
        requests.free([source, destination], torch.cat([src, dst]))


@pytest.mark.parametrize("ratio,layer", [(4, 0), (128, 1)])
@pytest.mark.parametrize("length", [512, 2048])
@pytest.mark.parametrize("cache", [(256, 0), (256, 1), (256, 5)], indirect=True)
def test_compressor_checkpoint_continuation_matches_full_prefill(cache, ratio, layer, length):
    from lightllm.models.deepseek_v4.layer_infer.compressor import fused_compress

    manager, requests = cache
    source, src = _reserve(manager, requests, length)
    destination, dst = _reserve(manager, requests, length)
    _prepare_swa(requests, source, 0, length)
    scores = torch.randn((length, 512 * (4 if ratio == 4 else 2)), dtype=torch.float32, device="cuda")
    cos = torch.ones((length, 32), dtype=torch.float32, device="cuda")
    sin = torch.zeros_like(cos)
    weight = torch.ones(512, dtype=torch.float32, device="cuda")

    def compress(req_idx, slots, start, end, decode=False):
        positions = torch.arange(start, end, dtype=torch.int32, device="cuda")
        width = end - start
        state = SimpleNamespace(
            mem_manager=manager,
            req_manager=requests,
            mem_index=slots[start:end],
            position_ids=positions,
            is_prefill=not decode,
            _dsv4_token_to_batch_idx=torch.zeros(width, dtype=torch.int32, device="cuda"),
            b_req_idx=torch.full((width if decode else 1,), req_idx, dtype=torch.int32, device="cuda"),
            b_mtp_index=torch.arange(width, dtype=torch.int32, device="cuda")
            if decode
            else torch.zeros(1, dtype=torch.int32, device="cuda"),
            b_seq_len=positions + 1 if decode else torch.tensor([end], dtype=torch.int32, device="cuda"),
            b_ready_cache_len=None if decode else torch.tensor([start], dtype=torch.int32, device="cuda"),
            b_q_start_loc=None if decode else torch.zeros(1, dtype=torch.int32, device="cuda"),
            dsv4_swa_write_slots=requests.get_swa_token_indices(req_idx, positions),
            dsv4_swa_runtime_write_slots=torch.where(
                positions >= end - requests.swa_capacity_tokens,
                requests.get_swa_token_indices(req_idx, positions, prefill=False),
                -1,
            )
            if not decode
            else None,
        )
        fused_compress(
            kv_score=scores[start:end],
            infer_state=state,
            layer_idx=layer,
            norm_weight=weight,
            eps=1e-6,
            head_dim=512,
            qk_rope_head_dim=64,
            compress_ratio=ratio,
            cos_table=cos,
            sin_table=sin,
        )

    compress(source, src, 0, length)
    states = requests.create_small_page_cache_manager(1)
    index = states.alloc_one_state_cache()
    requests.save_state(source, index, states, checkpoint_len=256)
    requests.restore_state(SimpleNamespace(req_idx=destination, cur_kv_len=256), states, index)
    verify_width = requests.c4_state_tokens - 8 + 1
    for start in range(256, length - verify_width, 256):
        end = min(start + 256, length - verify_width)
        _prepare_swa(requests, destination, start, end)
        compress(destination, dst, start, end)
        _finish_prefill(requests, [destination])
    # Rejected speculative rows are written again in the same held token page.
    compress(destination, dst, length - verify_width, length, decode=True)
    compress(destination, dst, length - verify_width, length, decode=True)
    pool = _history_pools(manager)[0] if ratio == 4 else _history_pools(manager)[2]
    torch.testing.assert_close(
        _read_packed_rows(pool, 0, src[256 + ratio - 1 : length : ratio].long() // ratio),
        _read_packed_rows(pool, 0, dst[256 + ratio - 1 : length : ratio].long() // ratio),
        rtol=0,
        atol=0,
    )
    requests.free([source, destination], torch.cat([src, dst]))
    states.free_state_cache([index])


def test_swa_index_cuda_graph_reads_updated_private_pages(cache):
    from lightllm.models.deepseek_v4.triton_kernel.build_swa_index_dsv4 import build_swa_index

    manager, requests = cache
    source, src = _reserve(manager, requests, 512)
    destination, dst = _reserve(manager, requests, 512)
    _prepare_swa(requests, source, 0, 512)
    _prepare_swa(requests, destination, 0, 512)
    req_ids = torch.tensor([source, requests.HOLD_REQUEST_ID], dtype=torch.int32, device="cuda")
    positions = torch.tensor([255, 1], dtype=torch.int32, device="cuda")
    indexes = torch.empty((2, 128), dtype=torch.int32, device="cuda")
    lengths = torch.empty(2, dtype=torch.int32, device="cuda")
    write_slots = torch.empty_like(lengths)

    def build():
        build_swa_index(req_ids, positions, requests, indexes, lengths, write_slots, is_prefill=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        build()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        build()
    req_ids[0] = destination
    positions[0] = 256
    graph.replay()
    expected = requests.get_swa_token_indices(destination, torch.arange(256, 128, -1, dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(indexes[0], expected)
    assert write_slots[0].item() == expected[0].item()
    assert write_slots[1].item() == requests.HOLD_SWA_SLOT + 1
    requests.free([source, destination], torch.cat([src, dst]))


@pytest.mark.parametrize(
    "dspark,cache",
    [(False, (256, 0)), (False, (256, 1)), (False, (256, 5)), (True, (256, 1)), (True, (256, 5))],
    indirect=["cache"],
)
def test_swa_ring_attention_matches_full_cache_after_speculative_retries(cache, dspark):
    from lightllm.common.basemodel.attention.nsa import dsv4_fp8_flashmla_sparse as attention
    from lightllm.models.deepseek_v4.triton_kernel.build_swa_index_dsv4 import build_swa_index
    from lightllm.models.deepseek_v4.triton_kernel.build_dspark_swa_index import build_dspark_swa_index

    torch.manual_seed(1593)
    manager, requests = cache
    source, src = _reserve(manager, requests, 4096)
    destination, dst = _reserve(manager, requests, 4096)
    _prepare_swa(requests, destination, 0, 512)
    # The reference retains every position in the other forward workspace.
    _prepare_swa(requests, source, 0, 4096, microbatch_index=1)
    values = torch.randn((4096, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    packed = torch.zeros((4096, 584), dtype=torch.uint8, device="cuda")
    packed[:, :448] = values[:, :448].to(torch.float8_e4m3fn).view(torch.uint8)
    packed[:, 448:576] = values[:, 448:].contiguous().view(torch.uint8)
    packed[:, 576:583] = 127
    positions = torch.arange(4096, device="cuda")
    _write_packed_rows(requests.swa_buffer, 2, requests.get_swa_token_indices(source, positions), packed)
    _write_swa(requests, 2, destination, positions[:512], packed[:512])
    _finish_prefill(requests, [destination])

    mtp_step = requests.c4_state_tokens - 8
    token_num = mtp_step if dspark else mtp_step + 1
    width = 256 if dspark else 128
    req_idx = torch.full((token_num,), destination, dtype=torch.int32, device="cuda")
    query_positions = torch.arange(512, 512 + token_num, dtype=torch.int32, device="cuda")
    indices = torch.empty((token_num, width), dtype=torch.int32, device="cuda")
    lengths = torch.empty(token_num, dtype=torch.int32, device="cuda")
    write_slots = torch.empty_like(lengths)

    def build():
        if dspark:
            build_dspark_swa_index(
                req_idx,
                query_positions,
                requests,
                indices,
                lengths,
                write_slots,
                window=128,
                block_size=token_num,
                hold_req_id=requests.HOLD_REQUEST_ID,
                hold_swa_slot=requests.HOLD_SWA_SLOT,
            )
        else:
            build_swa_index(req_idx, query_positions, requests, indices, lengths, write_slots)

    backend = attention.DeepseekV4FlashMlaFp8SparseAttBackend(
        type(
            "Model",
            (),
            {
                "config": {"num_attention_heads": 64, "compress_ratios": [0]},
                "tp_world_size_": 1,
                "req_manager": requests,
            },
        )()
    )
    state_info = SimpleNamespace(mem_manager=manager)
    state = backend.create_att_decode_state(state_info)
    state.init_state()
    params = dict(
        compress_ratio=0,
        head_dim_v=512,
        softmax_scale=512 ** -0.5,
        swa_indices=indices.unsqueeze(1),
        swa_lengths=lengths,
        attn_sink=torch.zeros(64, device="cuda"),
    )
    control = attention.AttControl(nsa_decode=True, nsa_decode_dict=params)
    q = torch.randn((token_num, 64, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    packed_kv = requests.get_att_input_params(2)

    def forward():
        build()
        return state.decode_att(q, packed_kv, None, att_control=control)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = forward()

    reference_state = backend.create_att_decode_state(state_info)
    reference_state.init_state()
    committed_end = 512
    # Both sides of page boundaries, repeated retries, and several complete ring wraps.
    for start in (512, 513, 638, 639, 640, 640, 641, 767, 768, 1023, 1279, 1280, 1919, 1920, 2047):
        # Advance intervening committed KV in small steps. A large parallel
        # prefill would instead borrow pages before writing the whole chunk.
        for gap_start in range(committed_end, start, 128):
            gap = positions[gap_start : min(start, gap_start + 128)]
            _write_packed_rows(requests.swa_buffer, 2, requests.get_swa_token_indices(source, gap), packed[gap])
            _write_packed_rows(requests.swa_buffer, 2, requests.get_swa_token_indices(destination, gap), packed[gap])
        query_positions.copy_(torch.arange(start, start + token_num, dtype=torch.int32))
        current = packed[start : start + token_num].clone()
        current[:, :448] = (
            (values[start : start + token_num, :448] * (1 + start % 7)).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        logical_positions = query_positions.long()
        _write_packed_rows(requests.swa_buffer, 2, requests.get_swa_token_indices(source, logical_positions), current)
        _write_packed_rows(
            requests.swa_buffer, 2, requests.get_swa_token_indices(destination, logical_positions), current
        )
        history = torch.arange(max(0, start - 128), start, device="cuda")
        graph.replay()
        if dspark:
            visible = torch.cat((history.flip(0), logical_positions))
            reference_indices = requests.get_swa_token_indices(source, visible).expand(token_num, -1).contiguous()
            reference_lengths = torch.full_like(lengths, visible.numel())
        else:
            visible = logical_positions[:, None] - torch.arange(128, device="cuda")
            reference_indices = requests.get_swa_token_indices(source, visible)
            reference_lengths = torch.full_like(lengths, 128)
        reference_indices = torch.nn.functional.pad(
            reference_indices, (0, width - reference_indices.shape[1]), value=-1
        )
        reference_indices = reference_indices.to(torch.int32)
        reference_params = dict(params, swa_indices=reference_indices.unsqueeze(1), swa_lengths=reference_lengths)
        expected = reference_state.decode_att(
            q, packed_kv, None, att_control=attention.AttControl(nsa_decode=True, nsa_decode_dict=reference_params)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"SWA mismatch at position {start}")
        torch.testing.assert_close(
            write_slots, requests.get_swa_token_indices(destination, logical_positions).to(torch.int32)
        )
        # Accept one position; speculative future rows stay dirty until the next write.
        committed_end = start + 1
    requests.free([source, destination], torch.cat([src, dst]))


@pytest.mark.parametrize(
    "cache,hash_page_size", [(256, 256), (256, 512), (512, 512), (768, 768), (1024, 1024)], indirect=["cache"]
)
@pytest.mark.parametrize("hit_extra_page", [False, True])
@pytest.mark.parametrize("chunked", [True, False])
def test_hybrid_radix_hit_fork_pause_abort_and_eviction(cache, monkeypatch, hash_page_size, hit_extra_page, chunked):
    from sortedcontainers import SortedDict
    from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context, CacheTier
    from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
    from lightllm.utils.kv_cache_utils import compute_token_list_hash

    manager, requests = cache
    small = requests.create_small_page_cache_manager(2)
    block_num = 2048 // hash_page_size
    big_page_tokens = block_num * hash_page_size
    radix = HybridAttPagedRadixCache(manager.size, 0, hash_page_size, block_num, manager, small)
    args = get_env_start_args()
    args.linear_att_hash_page_size = hash_page_size
    args.linear_att_page_block_num = block_num
    args.disable_chunked_prefill = not chunked
    tail_len = big_page_tokens + hash_page_size
    hit_len = tail_len if hit_extra_page else big_page_tokens
    if not chunked:
        args.chunked_prefill_size = args.max_req_total_len
    for name, value in {
        "req_manager": requests,
        "radix_cache": radix,
        "args": args,
        "is_hybrid_att_model": True,
    }.items():
        monkeypatch.setattr(g_infer_context, name, value, raising=False)

    def request(req_idx, total_len):
        req = InferReq.__new__(InferReq)
        req.req_idx, req.args = req_idx, args
        req.cur_kv_len, req.hold_kv_len, req.cur_output_len = 0, 0, 0
        req.shared_kv_node = None
        req.tail_small_page_buffer_id = None
        req.hybrid_len_to_big_page_id = SortedDict()
        req.hybrid_cache_len = (total_len - 1) // hash_page_size * hash_page_size
        req.mm_input_spans = []
        req.cache_tiers = {CacheTier.GPU}
        req.sampling_param = SimpleNamespace(disable_prompt_cache=False)
        req.prompt_selected_logprobs = SimpleNamespace(copy_capture_slots_if_needed=lambda **kwargs: None)
        tokens = list(range(total_len))
        hashes = compute_token_list_hash(tokens, hash_page_size)
        req.shm_req = SimpleNamespace(
            input_len=total_len,
            shm_prompt_ids=SimpleNamespace(arr=tokens),
            hybrid_token_hash_list=SimpleNamespace(size=len(hashes), get_all=lambda: hashes),
            shm_cur_kv_len=0,
            prompt_cache_len=0,
        )
        req.get_chuncked_input_token_len = req.get_chuncked_input_token_len_for_hybrid_att
        return req

    source, held = _reserve(manager, requests, tail_len + 1)
    req = request(source, tail_len + 1)
    req.hold_kv_len = held.numel()
    for end in (big_page_tokens, tail_len) if chunked else (tail_len + 1,):
        assert req.get_chuncked_input_token_len() == end
        _prepare_swa(requests, source, req.cur_kv_len, end)
        for pool in (requests.swa_buffer, *_history_pools(manager)):
            pool.random_(0, 256)
        _state_buffer(requests, "c4").uniform_()
        _state_buffer(requests, "c4_indexer").uniform_()
        g_infer_context.save_hybrid_state_to_cache(torch.tensor([source], device="cuda"), [req])
        req.cur_kv_len = end
    _prepare_swa(requests, source, req.cur_kv_len, tail_len + 1)
    req.cur_kv_len = tail_len + 1
    freed = []
    g_infer_context.free_a_req_mem(freed, req)
    manager.free(torch.cat(freed))
    requests.free_req(source)
    assert radix.get_tree_total_tokens_num() == tail_len

    forks = [request(requests.alloc(), hit_len + 1) for _ in range(2)]
    for fork in forks:
        fork._hybrid_match_radix_cache()
        assert fork.cur_kv_len == hit_len
        assert fork.hold_kv_len == hit_len
        assert fork.shared_kv_node.node_prefix_total_len == big_page_tokens
        if hit_extra_page:
            assert requests.req_to_token_indexs[fork.req_idx, big_page_tokens].item() != held[big_page_tokens].item()
        restored = requests.req_to_token_indexs[fork.req_idx, :hit_len]
        for pool, ratio in zip(_history_pools(manager), (4, 4, 128)):
            torch.testing.assert_close(
                _read_packed_rows(pool, 0, held[ratio - 1 : hit_len : ratio].long() // ratio),
                _read_packed_rows(pool, 0, restored[ratio - 1 : hit_len : ratio].long() // ratio),
            )
    assert set(_request_pages(requests, forks[0].req_idx)).isdisjoint(_request_pages(requests, forks[1].req_idx))
    # Pause can publish a reusable prefix; abort with GPU placement disabled only dereferences it.
    forks[1].cache_tiers.clear()
    for fork in forks:
        freed = []
        g_infer_context.free_a_req_mem(freed, fork)
        manager.free(torch.cat(freed))
        requests.free_req(fork.req_idx)
    radix.free_radix_cache_to_get_enough_token(manager.size)
    assert manager.allocator.can_use_mem_size == manager.size
    assert (
        manager.big_page_buffers.get_free_cache_num()
        == manager.big_page_buffers.size - manager.big_page_buffers.keep_num
    )
    assert small.get_free_cache_num() == small.size


@pytest.mark.parametrize("mode", ["decode", "prefill", "prefill_aux"])
def test_c4_indexer_metadata_scoring_and_graph_replay(cache, monkeypatch, mode):
    from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
    from lightllm.models.deepseek_v4.layer_infer import indexer as indexer_module
    from lightllm.models.deepseek_v4.workspace import DeepseekV4C4PrefillWorkspace, DeepseekV4Workspace

    torch.manual_seed(1593)
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    args = get_env_start_args()
    monkeypatch.setitem(args, "mtp_mode", "none")
    monkeypatch.setitem(args, "enable_decode_microbatch_overlap", False)
    monkeypatch.setitem(args, "enable_prefill_microbatch_overlap", False)
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, 4096)
    heads, tokens = 64, 5
    config = dict(
        compress_ratios=[4, 128, 0],
        index_topk=512,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_n_heads=heads,
        sliding_window=128,
        head_dim=512,
        rms_norm_eps=1e-6,
    )
    indexer = indexer_module.DeepseekV4IndexInfer(0, config, 1)
    model = SimpleNamespace(batch_max_tokens=tokens, config=config, max_seq_length=4096)
    state = DeepseekV4InferStateInfo()
    state.mem_manager, state.req_manager = manager, requests
    state.dsv4_workspace = DeepseekV4Workspace(model)
    state.is_prefill = mode != "decode"
    state.microbatch_index, state.max_kv_seq_len = 0, 4096
    state.b_req_idx = torch.tensor([req_idx] if state.is_prefill else [req_idx] * tokens, device="cuda")
    state.b_seq_len = torch.full_like(state.b_req_idx, 4096)
    state.position_ids = torch.arange(3068, 3068 + tokens, device="cuda")
    state._dsv4_token_to_batch_idx = torch.zeros(tokens, dtype=torch.int64, device="cuda")
    # Exercise chunk scheduling without requiring a long prompt or large logits allocation.
    monkeypatch.setattr(indexer_module, "C4_PREFILL_LOGITS_BUDGET_BYTES", 2 * 1024 * 4)
    aux = (
        DeepseekV4C4PrefillWorkspace(tokens, 4096, 1, heads, 128, requests.cache_config)
        if mode == "prefill_aux"
        else None
    )
    if aux is not None:
        monkeypatch.setattr(aux, "rows_per_logits_chunk", lambda capacity: 2)
    keys = (torch.randn((1024, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    scales = torch.ones((1024, 1), dtype=torch.float32, device="cuda")
    packed = torch.cat((keys.view(torch.uint8), scales.view(torch.uint8)), dim=1)
    c4_slots = slots[3::4].long() // 4
    _write_packed_rows(_history_pools(manager)[1], 0, c4_slots, packed)
    q = (torch.randn((tokens, heads, 128), device="cuda") * 0.125).to(torch.float8_e4m3fn)
    weights = torch.rand((tokens, heads), device="cuda")

    def select():
        return indexer._c4_indices(state, q, weights, state.position_ids, c4_aux_workspace=aux)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        select()
        state.c4_indexer_metadata = None
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, captured_lengths = select()
    # Change both causal lengths and Q inputs; graph metadata must be recomputed.
    state.position_ids.add_(256)
    state.b_seq_len.fill_(3584)
    q.copy_((q.float() * 1.25).to(q.dtype))
    graph.replay()
    actual = captured.clone()
    actual_lengths = captured_lengths.clone()
    state.c4_indexer_metadata = None
    eager, eager_lengths = select()
    torch.testing.assert_close(actual.sort(dim=-1).values, eager.sort(dim=-1).values, rtol=0, atol=0)
    torch.testing.assert_close(actual_lengths, eager_lengths, rtol=0, atol=0)
    torch.testing.assert_close(actual_lengths, torch.full_like(actual_lengths, 512))
    inverse_slots = torch.full((_history_pools(manager)[1].shape[1] * 64,), -1, device="cuda", dtype=torch.long)
    inverse_slots[c4_slots] = torch.arange(1024, device="cuda")
    logical_indices = inverse_slots[actual[:, 0].long()]
    scores = ((q.float() @ keys.float().T).relu() * weights[:, :, None]).sum(1)
    valid_lengths = (state.position_ids + 1) // 4
    for row in range(tokens):
        selected = logical_indices[row]
        assert ((selected >= 0) & (selected < valid_lengths[row])).all()
        selected_scores = scores[row, selected].sort(descending=True).values
        reference_scores = scores[row, : valid_lengths[row]].topk(512).values
        torch.testing.assert_close(selected_scores, reference_scores, rtol=2e-3, atol=2e-3)
    requests.free([req_idx], slots)


def test_empty_dspark_commit_is_local_but_proposal_keeps_common_padding_path(monkeypatch):
    from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    model_input = SimpleNamespace(batch_size=0, mtp_draft_input_hiddens=torch.empty((0, 8), device="cuda"))
    # A KV-only empty commit is local; a full proposal still takes the common
    # model path and participates in DP collectives through HOLD padding.
    output = model._decode(model_input)
    assert output.logits.shape == (0, 1)
    assert output.logits.device == model_input.mtp_draft_input_hiddens.device

    proposal_calls = []
    monkeypatch.setattr(DeepseekV4TpPartModel, "_decode", lambda self, inputs: proposal_calls.append(inputs))
    model_input.mtp_draft_input_hiddens = None
    model._decode(model_input)
    assert proposal_calls == [model_input]


@pytest.mark.parametrize(
    "length,prefix", [(768, 0), (768, 256), (2816, 0), (2816, 2304), (4096, 0), (4096, 256), (4096, 2304)]
)
@pytest.mark.parametrize("exhaust_small_pool", [False, True])
def test_standard_cpu_cache_offload_and_load_use_saved_checkpoints(
    cache, monkeypatch, length, prefix, exhaust_small_pool
):
    """Offload after the runtime ring is overwritten, then resume through the common loader."""
    from collections import deque
    from lightllm.server.multi_level_kv_cache import CacheTier
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common
    from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager

    manager, requests = cache
    source, src = _reserve(manager, requests, length)
    requests.create_small_page_cache_manager(4)
    page_tokens = manager.cache_config.cpu_cache_token_page_size
    ends = list(range(page_tokens, length + 1, page_tokens))
    if length % page_tokens:
        ends.append(length)
    source_checkpoints = {}
    tail = None
    expected_checkpoints = []
    for end in ends:
        value = 11 + end // 256
        requests.swa_buffer.fill_(value)
        _state_buffer(requests, "c4").fill_(value)
        _state_buffer(requests, "c4_indexer").fill_(value)
        state_pool = requests.small_page_buffers if end % page_tokens else manager.big_page_buffers
        slot = state_pool.alloc_one_state_cache()
        requests.save_state(source, slot, state_pool, end)
        torch.cuda.synchronize()
        expected_checkpoints.append(state_pool.buffer[slot].clone())
        if end % page_tokens:
            tail = slot
        else:
            source_checkpoints[end] = slot
    # The live ring no longer contains the prompt's continuation checkpoints.
    requests.swa_buffer.fill_(99)
    _state_buffer(requests, "c4").fill_(88)
    _state_buffer(requests, "c4_indexer").fill_(77)
    _randomize_history(manager)
    pages = torch.full(
        (len(ends) + 2, 1, 1, 1, manager.cache_config.get_cpu_cache_big_page_bytes()),
        221,
        dtype=torch.uint8,
        pin_memory=True,
    )
    published, released = [], []
    client = SimpleNamespace(
        cpu_kv_cache_tensor=pages,
        lock=SimpleNamespace(acquire_sleep1ms=lambda: None, release=lambda: None),
        allocate_pages=lambda hashes, **kwargs: (list(range(1, len(hashes) + 1)), [False] * len(hashes)),
        update_pages_status_to_ready=lambda **kwargs: published.append(kwargs),
        deref_pages=lambda page_list: released.extend(page_list),
    )
    req = SimpleNamespace(
        req_idx=source,
        req_id=42,
        cur_kv_len=length,
        hold_kv_len=src.numel(),
        shared_kv_node=None,
        hybrid_len_to_big_page_id=source_checkpoints,
        hybrid_cache_len=length,
        tail_small_page_buffer_id=tail,
        cpu_cache_task_status=InferReq._CpuCacheTaskStatus.NOT_STARTED,
        cache_tiers=(CacheTier.CPU,),
        shm_req=SimpleNamespace(
            group_req_id=42,
            request_id=42,
            input_len=length + 1,
            token_hash_list=SimpleNamespace(get_all=lambda: list(range(len(ends)))),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: ends),
        ),
    )
    producer = torch.cuda.current_stream()
    context = common.g_infer_context
    monkeypatch.setattr(context, "req_manager", requests)
    monkeypatch.setattr(
        context,
        "radix_cache",
        SimpleNamespace(get_big_page_ids_by_node=lambda node: [], free_one_small_page_buffer=lambda: None),
    )
    monkeypatch.setattr(context, "is_hybrid_att_model", True)
    monkeypatch.setattr(context, "get_overlap_stream", lambda: producer)
    monkeypatch.setattr(context, "get_cpu_kv_cache_stream", torch.cuda.Stream)
    monkeypatch.setattr(context, "get_can_alloc_token_num", lambda: manager.allocator.can_use_mem_size)
    monkeypatch.setattr(common.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(common.dist, "barrier", lambda **kwargs: None)
    module = common.MultiLevelKvCacheModule.__new__(common.MultiLevelKvCacheModule)
    module.args = SimpleNamespace(
        page_size=manager.page_size,
        cpu_cache_token_page_size=page_tokens,
        linear_att_hash_page_size=256,
        disable_linear_att_small_page_cpu_cache=False,
    )
    module.backend = SimpleNamespace(
        is_master_in_dp=True, dp_world_size=1, model=SimpleNamespace(mem_manager=manager, req_manager=requests)
    )
    module.cpu_cache_client = client
    module.cpu_cache_handle_queue = deque()
    module.page_index_buffer = torch.empty(8, dtype=torch.int32, device="cuda")
    module.page_ready_buffer = torch.empty(8, dtype=torch.bool, device="cuda")
    module.gloo_group = module.filter_group = module.init_sync_group = None
    module.need_sync_compute_stream = lambda: False
    assert module.offload_finished_reqs_to_cpu_cache([req]) == []
    assert req.cpu_cache_task_status.is_running()
    torch.cuda.synchronize()
    module.update_cpu_cache_task_states()
    assert module.offload_finished_reqs_to_cpu_cache([req]) == [req]
    assert published == [
        dict(
            page_list=list(range(1, len(ends) + 1)),
            deref=True,
            disk_offload_enable=False,
            token_num_in_page_list=length,
        )
    ]
    flat = pages.view(pages.shape[0], -1)
    for i, expected in enumerate(expected_checkpoints, 1):
        torch.testing.assert_close(
            flat[i, manager.cache_config.get_cpu_cache_full_att_bytes() :], expected, rtol=0, atol=0
        )
    if tail is not None:
        torch.testing.assert_close(
            manager.big_page_buffers.buffer[manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID],
            expected_checkpoints[-1],
            rtol=0,
            atol=0,
        )

    destination = requests.alloc()
    dst = torch.empty(0, dtype=torch.int32, device="cuda")
    if prefix:
        dst = manager.alloc(prefix).cuda()
        requests.req_to_token_indexs[destination, :prefix] = dst
        manager.operator.copy_mem_to_mem(src[:prefix], dst)
    restored = SimpleNamespace(
        req_idx=destination,
        cur_kv_len=prefix,
        hold_kv_len=prefix,
        hybrid_cache_len=length,
        hybrid_len_to_big_page_id={},
        tail_small_page_buffer_id=None,
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
        shm_req=SimpleNamespace(
            input_len=length + 1,
            disk_prompt_cache_len=0,
            cpu_prompt_cache_len=0,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: list(range(1, len(ends) + 1))),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: ends),
        ),
    )
    if prefix >= page_tokens:
        slot = manager.big_page_buffers.alloc_one_state_cache()
        manager.big_page_buffers.buffer[slot].copy_(expected_checkpoints[0])
        restored.hybrid_len_to_big_page_id[page_tokens] = slot

    def allocate(req, count):
        slots = manager.alloc(count)
        requests.req_to_token_indexs[req.req_idx, req.hold_kv_len : req.hold_kv_len + count].copy_(slots)
        req.hold_kv_len += count
        return slots

    module.backend._alloc_req_kv_mem = allocate
    busy_small_slots = []
    if exhaust_small_pool:
        busy_small_slots = requests.small_page_buffers.alloc_state_cache(
            requests.small_page_buffers.get_free_cache_num()
        )
    module.load_cpu_cache_to_reqs([restored])
    torch.cuda.synchronize()
    assert restored.cur_kv_len == restored.hold_kv_len == length
    assert released == list(range(1, len(ends) + 1))
    if length % page_tokens:
        torch.testing.assert_close(
            manager.big_page_buffers.buffer[manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID],
            expected_checkpoints[-1],
            rtol=0,
            atol=0,
        )
        if exhaust_small_pool:
            assert restored.tail_small_page_buffer_id is None
        else:
            assert restored.tail_small_page_buffer_id is not None
            torch.testing.assert_close(
                requests.small_page_buffers.buffer[restored.tail_small_page_buffer_id],
                expected_checkpoints[-1],
                rtol=0,
                atol=0,
            )
    dst = requests.req_to_token_indexs[destination, :length]
    for pool in _history_pools(manager):
        torch.testing.assert_close(
            pool[:, (src[::256] // 256).long()], pool[:, (dst[::256] // 256).long()], rtol=0, atol=0
        )
    check = DeepseekV4StateCacheManager(1, manager.cache_config)
    requests.save_state(destination, 0, check, length)
    torch.cuda.synchronize()
    torch.testing.assert_close(check.buffer[0], expected_checkpoints[-1], rtol=0, atol=0)
    requests.free([source, destination], torch.cat((src, dst)))
    manager.big_page_buffers.free_state_cache(
        list(source_checkpoints.values()) + list(restored.hybrid_len_to_big_page_id.values())
    )
    requests.small_page_buffers.free_state_cache(
        [x for x in (tail, restored.tail_small_page_buffer_id) if x is not None] + busy_small_slots
    )
    assert manager.allocator.can_use_mem_size == manager.size
    assert (
        manager.big_page_buffers.get_free_cache_num()
        == manager.big_page_buffers.size - manager.big_page_buffers.keep_num
    )


def test_prefill_fused_writer_preserves_old_prefix_and_runtime_tail(cache):
    from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_k_norm_rope_flashmla
    from lightllm.models.deepseek_v4.triton_kernel.build_swa_index_dsv4 import build_swa_index

    manager, requests = cache
    req, held = _reserve(manager, requests, 2048)
    reference = torch.zeros((1, 17, 74880), dtype=torch.uint8, device="cuda")
    kv = torch.randn((2048, 512), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    angles = torch.randn((2048, 32), device="cuda")
    freqs = torch.complex(angles.cos(), angles.sin())
    for start, end in ((0, 255), (255, 641), (641, 1100), (1100, 2048)):
        _prepare_swa(requests, req, start, end)
        positions = torch.arange(start, end, device="cuda", dtype=torch.int32)
        indices = torch.empty((positions.numel(), 128), dtype=torch.int32, device="cuda")
        lengths = torch.empty(positions.numel(), dtype=torch.int32, device="cuda")
        writes = torch.empty_like(lengths)
        runtime = torch.empty_like(lengths)
        build_swa_index(
            torch.full_like(positions, req),
            positions,
            requests,
            indices,
            lengths,
            writes,
            is_prefill=True,
            runtime_write_slots=runtime,
            token_to_batch=torch.zeros_like(positions),
            seq_lens=torch.tensor([end], device="cuda"),
        )
        slots = requests.get_swa_token_indices(req, positions)
        requests.pack_mla_kv_to_cache_fused_norm_rope(2, slots, kv[start:end], weight, 1e-6, freqs, positions, runtime)
        fused_k_norm_rope_flashmla(kv[start:end], weight, 1e-6, freqs, positions, positions, reference[0], 128)
        visible = torch.arange(max(0, start - 127), end, device="cuda")
        torch.testing.assert_close(
            _read_packed_rows(requests.swa_buffer, 2, requests.get_swa_token_indices(req, visible)),
            _read_packed_rows(reference, 0, visible),
            rtol=0,
            atol=0,
        )
        logical = positions[:, None] - torch.arange(128, device="cuda")[None, :]
        valid = logical >= 0
        expected_slots = requests.get_swa_token_indices(req, logical.clamp_min(0))
        torch.testing.assert_close(indices[valid], expected_slots[valid].to(torch.int32))
        _finish_prefill(requests, [req])
        tail = torch.arange(max(0, end - requests.sliding_window), end, device="cuda")
        torch.testing.assert_close(
            _read_packed_rows(requests.swa_buffer, 2, requests.get_swa_token_indices(req, tail)),
            _read_packed_rows(reference, 0, tail),
            rtol=0,
            atol=0,
        )
    requests.free([req], held)


def test_cpu_cache_preserves_already_published_pages(cache):
    from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_history_pages, copy_checkpoint_pages

    manager, requests = cache
    req, slots = _reserve(manager, requests, 2048)
    _randomize_history(manager)
    layout = manager.cache_config
    pages = torch.full((3, layout.get_cpu_cache_big_page_bytes()), 221, dtype=torch.uint8, pin_memory=True)
    page_ids = torch.tensor([1], dtype=torch.int32, device="cuda")
    readies = torch.ones(1, dtype=torch.bool, device="cuda")
    state_id = manager.big_page_buffers.alloc_one_state_cache()
    manager.big_page_buffers.buffer[state_id].fill_(7)
    checkpoint_ids = torch.tensor([state_id], dtype=torch.int64, device="cuda")
    copy_history_pages(manager, slots, page_ids, pages, to_pages=True, readies=readies)
    copy_checkpoint_pages(manager, checkpoint_ids, page_ids, pages, to_pages=True, readies=readies)
    torch.cuda.synchronize()
    assert pages.eq(221).all()
    readies.zero_()
    copy_history_pages(manager, slots, page_ids, pages, to_pages=True, readies=readies)
    copy_checkpoint_pages(manager, checkpoint_ids, page_ids, pages, to_pages=True, readies=readies)
    torch.cuda.synchronize()
    offset = 0
    for pool in manager.kv_buffer:
        expected = pool[slots.long()].cpu().reshape(-1)
        torch.testing.assert_close(pages[1, offset : offset + expected.numel()], expected, rtol=0, atol=0)
        offset += expected.numel()
    assert pages[1, layout.get_cpu_cache_full_att_bytes() :].eq(7).all()
    assert pages[0].eq(221).all() and pages[2].eq(221).all()
    manager.big_page_buffers.free_state_cache([state_id])
    requests.free([req], slots)


@pytest.mark.parametrize("window", [64, 128, 256, 512])
@pytest.mark.parametrize("mtp_step", [0, 5])
def test_configured_window_survives_chunks_and_checkpoint_restore(cache, window, mtp_step):
    from dataclasses import replace

    manager, _ = cache
    config = replace(manager.cache_config, sliding_window=window)
    requests = DeepseekV4ReqManager(2, 4096, manager, config, max_forward_tokens=2048, mtp_step=mtp_step)
    requests.prefill_c4_checkpoints.zero_()
    requests.prefill_indexer_checkpoints.zero_()
    source, destination = requests.alloc(), requests.alloc()
    packed = torch.randint(0, 256, (2048, config.kv_bytes), dtype=torch.uint8, device="cuda")
    states = requests.create_small_page_cache_manager(1)
    index = states.alloc_one_state_cache()
    for start, end in ((0, 257), (257, 513), (513, 2048)):
        _prepare_swa(requests, source, start, end)
        positions = torch.arange(start, end, device="cuda")
        _write_swa(requests, 0, source, positions, packed[start:end])
        for boundary in range((start // 256 + 1) * 256, end + 1, 256):
            requests.save_state(source, index, states, boundary)
            requests.restore_state(SimpleNamespace(req_idx=destination, cur_kv_len=boundary), states, index)
            positions = torch.arange(max(0, boundary - window), boundary, device="cuda")
            actual = _read_packed_rows(requests.swa_buffer, 0, requests.get_swa_token_indices(destination, positions))
            torch.testing.assert_close(actual, packed[positions], rtol=0, atol=0)
        _finish_prefill(requests, [source])
        positions = torch.arange(max(0, end - window), end, device="cuda")
        actual = _read_packed_rows(requests.swa_buffer, 0, requests.get_swa_token_indices(source, positions))
        torch.testing.assert_close(actual, packed[positions], rtol=0, atol=0)
    requests.free_req(source)
    requests.free_req(destination)
    states.free_state_cache([index])


@pytest.mark.parametrize("mtp_step", [0, 3])
def test_full_batch_unaligned_prefill_workspace_is_bounded_and_private(cache, mtp_step):
    manager, _ = cache
    request_count, chunk_tokens, ready = 8, 64, 382
    requests = DeepseekV4ReqManager(
        request_count,
        4096,
        manager,
        manager.cache_config,
        max_forward_tokens=request_count * chunk_tokens,
        mtp_step=mtp_step,
        prefill_regions=2,
    )
    reqs = [requests.alloc() for _ in range(request_count)]
    alignment = manager.cache_config.swa_token_alignment
    first = (ready - requests.sliding_window + 1) // alignment * alignment
    end = ready + chunk_tokens
    required = request_count * ((end + alignment - 1) // alignment * alignment - first)
    assert requests.prefill_capacity_tokens >= required
    next_region = requests.swa_runtime_tokens + requests.prefill_capacity_tokens
    requests.swa_buffer[:, next_region:].fill_(173)
    old_positions = torch.arange(ready - requests.sliding_window + 1, ready, device="cuda")
    new_positions = torch.arange(ready, end, device="cuda")
    for req in reqs:
        _write_packed_rows(
            requests.swa_buffer,
            0,
            requests.get_swa_token_indices(req, old_positions, prefill=False),
            torch.full((old_positions.numel(), 584), req + 1, dtype=torch.uint8, device="cuda"),
        )
    _prepare_prefill(
        requests, torch.tensor(reqs), torch.full((request_count,), ready), torch.full((request_count,), end)
    )
    for req in reqs:
        _write_swa(
            requests,
            0,
            req,
            new_positions,
            torch.full((chunk_tokens, 584), req + 17, dtype=torch.uint8, device="cuda"),
        )
    for req in reqs:
        positions = torch.arange(ready - requests.sliding_window + 1, end, device="cuda")
        slots = requests.get_swa_token_indices(req, positions)
        assert slots.unique().numel() == positions.numel()
        actual = _read_packed_rows(requests.swa_buffer, 0, slots)
        expected = torch.where(positions[:, None] < ready, req + 1, req + 17).expand_as(actual).to(torch.uint8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert requests.swa_buffer[:, next_region:].eq(173).all()
    _finish_prefill(requests, reqs)
    for req in reqs:
        positions = torch.arange(end - requests.sliding_window, end, device="cuda")
        actual = _read_packed_rows(requests.swa_buffer, 0, requests.get_swa_token_indices(req, positions))
        expected = torch.where(positions[:, None] < ready, req + 1, req + 17).expand_as(actual).to(torch.uint8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        requests.free_req(req)
