import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.models.deepseek_v4.triton_kernel.build_compress_index_dsv4 import build_compress_index
from lightllm.utils.envs_utils import get_env_start_args


@pytest.fixture
def cache(monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the packed cache kernels")
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 100}))
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_dsv4_paged_cache")
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "page_size": 256,
                "linear_att_hash_page_size": 256,
                "linear_att_page_block_num": 8,
                "chunked_prefill_size": 2048,
                "max_req_total_len": 4096,
                "disable_chunked_prefill": False,
                "model_dir": str(tmp_path),
                "penalty_counter_mode": "cpu_counter",
                "mtp_step": 3,
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
        compress_rates=[4, 128, 0],
        max_request_num=2,
        mtp_step=3,
        swa_full_tokens_ratio=1.0,
    )
    requests = DeepseekV4ReqManager(2, 4096, manager, sliding_window=128)
    yield manager, requests
    torch.cuda.synchronize()
    get_env_start_args.cache_clear()


def _reserve(manager, requests, length):
    req_idx = requests.alloc()
    held = (length + 255) // 256 * 256
    slots = manager.alloc(held).cuda()
    requests.req_to_token_indexs[req_idx, :held] = slots
    return req_idx, slots


@pytest.mark.parametrize("ratio", [4, 128])
def test_noncontiguous_pages_obey_logical_compression_boundaries(cache, ratio):
    manager, requests = cache
    req_idx, allocated = _reserve(manager, requests, 768)
    slots = allocated.view(3, 256)[torch.tensor([2, 0, 1], device="cuda")].reshape(-1)
    requests.req_to_token_indexs[req_idx, :768] = slots
    positions = torch.tensor([0, 2, 3, 126, 127, 254, 255, 256, 511, 512], device="cuda")
    indexes = torch.empty((len(positions), 768 // ratio), dtype=torch.int32, device="cuda")
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
    manager.c4_indexer_pool.buffer.zero_()
    manager.pack_indexer_k_to_cache(0, slots[:259], positions, keys)
    closed = torch.arange(3, 259, 4, device="cuda")
    actual = manager.c4_indexer_pool.read(0, slots[closed] // 4)
    expected = manager._pack_indexer_k(keys[closed])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Positions 256..258 cannot publish their reserved next compressed slot.
    assert manager.c4_indexer_pool.read(0, slots[256:257] // 4).eq(0).all()


def test_speculative_retry_reuses_swa_without_freeing_token_pages(cache):
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, 512)

    def cpu(data):
        return torch.tensor(data, dtype=torch.int32)

    requests.prepare_prefill(cpu([req_idx]), cpu([0]), cpu([254]))
    for sequences in ([255, 256, 257, 258], [255, 256, 257, 258], [256, 257, 258, 259]):
        requests.prepare_decode(cpu([req_idx] * 4), cpu(sequences), cpu([0, 1, 2, 3]))
        assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages - 3
        assert len(requests._swa_pages[req_idx]) == 3
        assert manager.allocator.can_use_mem_size == manager.size - 512
    requests.free([req_idx], slots)
    assert manager.allocator.can_use_mem_size == manager.size
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages
    assert requests.req_to_swa_pages[req_idx].eq(-1).all()


@pytest.mark.parametrize("model_kind", ["target", "mtp", "dspark"])
def test_model_selects_held_slots_and_prepares_only_owned_swa(cache, model_kind):
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
    from lightllm.models.deepseek_v4_mtp.model import DeepseekV4MTPModel
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, 512)
    requests.prepare_swa(req_idx, 0, 256)
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
        assert len(requests._swa_pages[req_idx]) == (2 if model_kind == "dspark" else 3)
        assert manager.allocator.can_use_mem_size == manager.size - 512
    requests.free([req_idx], slots)


def test_prefill_chunk_keeps_all_new_swa_rows(cache):
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, 2048)
    requests.prepare_prefill(torch.tensor([req_idx]), torch.tensor([0]), torch.tensor([2048]))
    assert requests.get_swa_slots(req_idx, torch.arange(2048, device="cuda")).ge(0).all()
    assert len(requests._swa_pages[req_idx]) == 16
    requests.free([req_idx], slots)
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages


def test_history_copy_preserves_packed_data_and_scale_regions(cache):
    manager, requests = cache
    _, src = _reserve(manager, requests, 512)
    _, dst = _reserve(manager, requests, 512)
    src = src.view(2, 256).flip(0).reshape(-1)
    for pool in (manager.c4_pool, manager.c4_indexer_pool, manager.c128_pool):
        pool.buffer.random_(0, 256)
    manager.operator.copy_mem_to_mem(src_mem_index=src, dst_mem_index=dst)
    for pool in (manager.c4_pool, manager.c4_indexer_pool, manager.c128_pool):
        torch.testing.assert_close(
            pool.buffer[:, src[::256].long() // 256], pool.buffer[:, dst[::256].long() // 256], rtol=0, atol=0
        )
    assert requests.req_to_swa_pages[: requests.HOLD_REQUEST_ID].eq(-1).all()


def test_hold_page_and_dspark_scratch_have_separate_capacity(cache):
    manager, requests = cache
    hold = requests.req_to_token_indexs[requests.HOLD_REQUEST_ID]
    torch.testing.assert_close(
        hold[:256], torch.arange(manager.size, manager.size + 256, dtype=torch.int32, device="cuda")
    )
    assert requests.req_to_swa_pages[requests.HOLD_REQUEST_ID].eq(manager.swa_num_pages).all()
    assert (hold // 4).lt(manager.c4_pool.num_pages * 64).all()
    assert (hold // 128).lt(manager.c128_pool.num_pages * 2).all()
    free_tokens = manager.allocator.can_use_mem_size
    for _ in range(3):
        pages_cpu, pages = manager.alloc_dspark_swa_block(16, 8)
        assert pages.numel() == 2
        assert manager.allocator.can_use_mem_size == free_tokens
        manager.free_dspark_swa_block(pages_cpu)
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages


def test_cpu_cache_roundtrip_uses_derived_history_slots(cache):
    manager, requests = cache
    req_idx, src = _reserve(manager, requests, 2048)
    requests.prepare_prefill(torch.tensor([req_idx]), torch.tensor([0]), torch.tensor([2048]))
    for pool in (manager.c4_pool, manager.c4_indexer_pool, manager.c128_pool, manager.swa_pool):
        pool.buffer.random_(0, 256)
    manager.c4_state_buffer.uniform_()
    manager.c4_indexer_state_buffer.uniform_()
    staging = torch.empty((1, manager.cpu_cache_layout.page_nbytes), dtype=torch.uint8, device="cuda")
    manager.operator.pack_cpu_cache_pages(src.view(1, -1), torch.tensor([[req_idx, 2048]], device="cuda"), staging)
    cpu_page = torch.empty(staging.shape, dtype=torch.uint8, pin_memory=True)
    cpu_page.copy_(staging)
    dst_req = requests.alloc()
    requests.prepare_swa(dst_req, 1792, 2048)
    resume_slots = requests.get_swa_slots(dst_req, torch.arange(1792, 2048, device="cuda"))
    plan = manager.prepare_cpu_cache_load(2048, 2048, resume_slots)
    manager.operator.load_cpu_cache_pages(
        plan, torch.tensor([0], dtype=torch.int32, device="cuda"), SimpleNamespace(cpu_kv_cache_tensor=cpu_page)
    )
    for pool, ratio in ((manager.c4_pool, 4), (manager.c4_indexer_pool, 4), (manager.c128_pool, 128)):
        torch.testing.assert_close(
            pool.read(0, src[ratio - 1 :: ratio].long() // ratio),
            pool.read(0, plan.mem_indexes[ratio - 1 :: ratio].long() // ratio),
            rtol=0,
            atol=0,
        )
    src_swa = requests.get_swa_slots(req_idx, torch.arange(1792, 2048, device="cuda")).long()
    dst_swa = resume_slots.long()
    for layer in range(manager.layer_num):
        torch.testing.assert_close(
            manager.swa_pool.read(layer, src_swa), manager.swa_pool.read(layer, dst_swa), rtol=0, atol=0
        )
    requests.free([req_idx, dst_req], torch.cat([src, plan.mem_indexes]))
    assert manager.allocator.can_use_mem_size == manager.size
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages


@pytest.mark.parametrize("length", [256, 512, 2048])
def test_shared_history_restores_private_continuation(cache, length):
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, length)
    requests.prepare_swa(req_idx, 0, length)
    manager.swa_pool.buffer.random_(0, 256)
    manager.c4_state_buffer.uniform_()
    manager.c4_indexer_state_buffer.uniform_()
    states = requests.create_small_page_cache_manager(1)
    state_idx = states.alloc_one_state_cache()
    requests.save_state(req_idx, state_idx, states, checkpoint_len=length)
    torch.cuda.synchronize()
    expected_swa, expected_c4, expected_indexer = [value.cuda() for value in states.get_state_cache(state_idx)]
    requests.free_req(req_idx)
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages
    reqs = [requests.alloc(), requests.alloc()]
    for req in reqs:
        requests.req_to_token_indexs[req, :length] = slots
        requests.restore_state(SimpleNamespace(req_idx=req, cur_kv_len=0), states, state_idx, checkpoint_len=length)
        pages = requests.req_to_swa_pages[req, length // 128 - 2 : length // 128].long()
        torch.testing.assert_close(manager.swa_pool.buffer[:, pages], expected_swa)
        tail = pages[-1] * 128 + torch.arange(124, 128, device="cuda")
        rows = tail // 128 * manager.c4_state_ring + tail % manager.c4_state_ring
        torch.testing.assert_close(manager.c4_state_buffer[:, rows], expected_c4)
        torch.testing.assert_close(manager.c4_indexer_state_buffer[:, rows], expected_indexer)
    assert set(requests._swa_pages[reqs[0]].values()).isdisjoint(requests._swa_pages[reqs[1]].values())
    # Advancing one fork releases only its own window; the shared token page survives.
    requests.prepare_swa(reqs[0], length + 512, length + 768)
    pages = requests.req_to_swa_pages[reqs[1], length // 128 - 2 : length // 128].long()
    torch.testing.assert_close(manager.swa_pool.buffer[:, pages], expected_swa)
    requests.free(reqs, slots)
    states.free_state_cache([state_idx])
    assert manager.allocator.can_use_mem_size == manager.size
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages
    assert states.get_free_cache_num() == 1


@pytest.mark.parametrize("length", [3, 127, 128, 255, 256, 257, 511, 512, 513, 2051])
def test_pd_roundtrip_preserves_live_and_aligned_continuations(cache, length):
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4PDCacheLayout
    from lightllm.models.deepseek_v4.triton_kernel.pd_cache_io import pack_pd_cache_page, unpack_pd_cache_page

    manager, requests = cache
    source, src = _reserve(manager, requests, length)
    destination, dst = _reserve(manager, requests, length)
    requests.prepare_swa(source, 0, length)
    requests.prepare_pd_decode_cache([destination], [length])
    for pool in (manager.swa_pool, manager.c4_pool, manager.c4_indexer_pool, manager.c128_pool):
        pool.buffer.random_(0, 256)
    for buffer in (manager.c4_state_buffer, manager.c4_indexer_state_buffer, manager.c128_state_buffer):
        buffer.uniform_()
    layout = DeepseekV4PDCacheLayout.from_compress_rates(manager.compress_rates, token_page_size=256)
    staging = torch.empty((layout.page_nbytes,), dtype=torch.uint8, device="cuda")
    for start in range(0, length, 256):
        end = min(length, start + 256)
        pack_pd_cache_page(manager, layout, src[start:end], staging, start, length, source)
        unpack_pd_cache_page(manager, layout, dst[start:end], staging, start, length, destination)
    for pool, ratio in ((manager.c4_pool, 4), (manager.c4_indexer_pool, 4), (manager.c128_pool, 128)):
        torch.testing.assert_close(
            pool.read(0, src[ratio - 1 : length : ratio].long() // ratio),
            pool.read(0, dst[ratio - 1 : length : ratio].long() // ratio),
            rtol=0,
            atol=0,
        )
    checkpoint = (length - 1) // 256 * 256
    positions = torch.arange(max(0, checkpoint - 256), length, device="cuda")
    src_swa = requests.get_swa_slots(source, positions).long()
    dst_swa = requests.get_swa_slots(destination, positions).long()
    for layer in range(manager.layer_num):
        torch.testing.assert_close(manager.swa_pool.read(layer, src_swa), manager.swa_pool.read(layer, dst_swa))
    state_positions = list(range(max(0, length - 4 - length % 4), length))
    if checkpoint:
        state_positions += list(range(checkpoint - 4, checkpoint))
    state_positions = torch.tensor(state_positions, device="cuda")
    src_swa = requests.get_swa_slots(source, state_positions).long()
    dst_swa = requests.get_swa_slots(destination, state_positions).long()
    src_rows = src_swa // 128 * manager.c4_state_ring + src_swa % manager.c4_state_ring
    dst_rows = dst_swa // 128 * manager.c4_state_ring + dst_swa % manager.c4_state_ring
    for buffer in (manager.c4_state_buffer, manager.c4_indexer_state_buffer):
        torch.testing.assert_close(buffer[:, src_rows], buffer[:, dst_rows])
    positions = torch.arange(length - length % 128, length, device="cuda")
    torch.testing.assert_close(
        manager.c128_state_buffer[:, source * manager.c128_state_ring + positions % manager.c128_state_ring],
        manager.c128_state_buffer[:, destination * manager.c128_state_ring + positions % manager.c128_state_ring],
    )
    requests.free([source, destination], torch.cat([src, dst]))
    assert manager.allocator.can_use_mem_size == manager.size
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages


def test_dp_transfer_uses_request_page_tables(cache):
    from lightllm.models.deepseek_v4.triton_kernel.dp_cache_io import copy_dsv4_dp_caches

    manager, requests = cache
    source, src = _reserve(manager, requests, 512)
    destination, dst = _reserve(manager, requests, 512)
    requests.prepare_swa(source, 0, 512)
    requests.prepare_swa(destination, 256, 512)
    for pool in (manager.swa_pool, manager.c4_pool, manager.c4_indexer_pool, manager.c128_pool):
        pool.buffer.random_(0, 256)
    manager.c4_state_buffer.uniform_()
    manager.c4_indexer_state_buffer.uniform_()
    pointers = torch.tensor(
        [
            [
                buffer.data_ptr()
                for buffer in (
                    manager.c4_pool.buffer,
                    manager.c4_indexer_pool.buffer,
                    manager.c128_pool.buffer,
                    requests.req_to_swa_pages,
                    manager.swa_pool.buffer,
                    manager.c4_state_buffer,
                    manager.c4_indexer_state_buffer,
                )
            ]
        ],
        dtype=torch.uint64,
        device="cuda",
    )
    meta = torch.tensor(
        [0, src.data_ptr(), dst.data_ptr(), source, destination, 512], dtype=torch.uint64, device="cuda"
    )
    history = torch.tensor([0, 0, 0, 1], dtype=torch.uint64, device="cuda")
    copy_dsv4_dp_caches(pointers, manager, meta, history)
    for pool, ratio in ((manager.c4_pool, 4), (manager.c4_indexer_pool, 4), (manager.c128_pool, 128)):
        torch.testing.assert_close(
            pool.read(0, src[ratio - 1 :: ratio].long() // ratio), pool.read(0, dst[ratio - 1 :: ratio].long() // ratio)
        )
    src_swa = requests.get_swa_slots(source, torch.arange(256, 512, device="cuda")).long()
    dst_swa = requests.get_swa_slots(destination, torch.arange(256, 512, device="cuda")).long()
    for layer in range(manager.layer_num):
        torch.testing.assert_close(manager.swa_pool.read(layer, src_swa), manager.swa_pool.read(layer, dst_swa))
    src_rows = src_swa[-4:] // 128 * manager.c4_state_ring + src_swa[-4:] % manager.c4_state_ring
    dst_rows = dst_swa[-4:] // 128 * manager.c4_state_ring + dst_swa[-4:] % manager.c4_state_ring
    for buffer in (manager.c4_state_buffer, manager.c4_indexer_state_buffer):
        torch.testing.assert_close(buffer[:, src_rows], buffer[:, dst_rows])
    requests.free([source, destination], torch.cat([src, dst]))
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages


@pytest.mark.parametrize("ratio,layer", [(4, 0), (128, 1)])
def test_compressor_checkpoint_continuation_matches_full_prefill(cache, ratio, layer):
    from lightllm.models.deepseek_v4.layer_infer.compressor import fused_compress

    manager, requests = cache
    source, src = _reserve(manager, requests, 512)
    destination, dst = _reserve(manager, requests, 512)
    requests.prepare_swa(source, 0, 512)
    scores = torch.randn((512, 512 * (4 if ratio == 4 else 2)), dtype=torch.float32, device="cuda")
    cos = torch.ones((512, 32), dtype=torch.float32, device="cuda")
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
            dsv4_swa_write_slots=requests.get_swa_slots(req_idx, positions),
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

    compress(source, src, 0, 512)
    states = requests.create_small_page_cache_manager(1)
    index = states.alloc_one_state_cache()
    requests.save_state(source, index, states, checkpoint_len=256)
    requests.restore_state(SimpleNamespace(req_idx=destination), states, index, checkpoint_len=256)
    requests.prepare_swa(destination, 256, 512)
    compress(destination, dst, 256, 508)
    # Rejected speculative rows are written again in the same held token page.
    compress(destination, dst, 508, 512, decode=True)
    compress(destination, dst, 508, 512, decode=True)
    pool = manager.c4_pool if ratio == 4 else manager.c128_pool
    torch.testing.assert_close(
        pool.read(0, src[256 + ratio - 1 : 512 : ratio].long() // ratio),
        pool.read(0, dst[256 + ratio - 1 : 512 : ratio].long() // ratio),
        rtol=0,
        atol=0,
    )
    requests.free([source, destination], torch.cat([src, dst]))
    states.free_state_cache([index])
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages


def test_swa_index_cuda_graph_reads_updated_private_pages(cache):
    from lightllm.models.deepseek_v4.triton_kernel.build_swa_index_dsv4 import build_swa_index

    manager, requests = cache
    source, src = _reserve(manager, requests, 512)
    destination, dst = _reserve(manager, requests, 512)
    requests.prepare_swa(source, 0, 512)
    requests.prepare_swa(destination, 0, 512)
    req_ids = torch.tensor([source, requests.HOLD_REQUEST_ID], dtype=torch.int32, device="cuda")
    positions = torch.tensor([255, 1], dtype=torch.int32, device="cuda")
    indexes = torch.empty((2, 128), dtype=torch.int32, device="cuda")
    lengths = torch.empty(2, dtype=torch.int32, device="cuda")
    write_slots = torch.empty_like(lengths)

    def build():
        build_swa_index(req_ids, positions, requests.req_to_swa_pages, indexes, lengths, write_slots)

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
    expected = requests.get_swa_slots(destination, torch.arange(256, 128, -1, dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(indexes[0], expected)
    assert write_slots[0].item() == expected[0].item()
    assert write_slots[1].item() == manager.swa_size + 1
    requests.free([source, destination], torch.cat([src, dst]))


@pytest.mark.parametrize("hit_len", [2048, 2304])
@pytest.mark.parametrize("chunked", [True, False])
def test_hybrid_radix_hit_fork_pause_abort_and_eviction(cache, monkeypatch, hit_len, chunked):
    from sortedcontainers import SortedDict
    from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context, CacheTier
    from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
    from lightllm.utils.kv_cache_utils import compute_token_list_hash

    manager, requests = cache
    small = requests.create_small_page_cache_manager(2)
    radix = HybridAttPagedRadixCache(manager.size, 0, 256, 8, manager, small)
    args = get_env_start_args()
    args.disable_chunked_prefill = not chunked
    if not chunked:
        args.chunked_prefill_size = args.max_req_total_len
    for name, value in {
        "req_manager": requests,
        "radix_cache": radix,
        "args": args,
        "is_hybrid_att_model": True,
        "is_deepseek_v4": True,
    }.items():
        monkeypatch.setattr(g_infer_context, name, value, raising=False)

    def request(req_idx, total_len):
        req = InferReq.__new__(InferReq)
        req.req_idx, req.args = req_idx, args
        req.cur_kv_len, req.hold_kv_len, req.cur_output_len = 0, 0, 0
        req.shared_kv_node = None
        req.tail_small_page_buffer_id = None
        req.hybrid_len_to_big_page_id = SortedDict()
        req.hybrid_cache_len = (total_len - 1) // 256 * 256
        req.image_block_spans = []
        req.cache_tiers = {CacheTier.GPU}
        req.sampling_param = SimpleNamespace(disable_prompt_cache=False)
        req.prompt_selected_logprobs = SimpleNamespace(copy_capture_slots_if_needed=lambda **kwargs: None)
        tokens = list(range(total_len))
        hashes = compute_token_list_hash(tokens, 256)
        req.shm_req = SimpleNamespace(
            input_len=total_len,
            shm_prompt_ids=SimpleNamespace(arr=tokens),
            hybrid_token_hash_list=SimpleNamespace(size=len(hashes), get_all=lambda: hashes),
            shm_cur_kv_len=0,
            prompt_cache_len=0,
        )
        req.get_chuncked_input_token_len = req.get_chuncked_input_token_len_for_hybrid_att
        return req

    source, held = _reserve(manager, requests, 2305)
    req = request(source, 2305)
    req.hold_kv_len = held.numel()
    for end in (2048, 2304) if chunked else (2305,):
        assert req.get_chuncked_input_token_len() == end
        requests.prepare_swa(source, req.cur_kv_len, end)
        for pool in (manager.swa_pool, manager.c4_pool, manager.c4_indexer_pool, manager.c128_pool):
            pool.buffer.random_(0, 256)
        manager.c4_state_buffer.uniform_()
        manager.c4_indexer_state_buffer.uniform_()
        g_infer_context.save_hybrid_state_to_cache(torch.tensor([source], device="cuda"), [req])
        req.cur_kv_len = end
    requests.prepare_swa(source, req.cur_kv_len, 2305)
    req.cur_kv_len = 2305
    freed = []
    g_infer_context.free_a_req_mem(freed, req)
    manager.free(torch.cat(freed))
    requests.free_req(source)
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages
    assert radix.get_tree_total_tokens_num() == 2304

    forks = [request(requests.alloc(), hit_len + 1) for _ in range(2)]
    for fork in forks:
        fork._hybrid_match_radix_cache()
        assert fork.cur_kv_len == hit_len
        assert fork.hold_kv_len == hit_len
        assert fork.shared_kv_node.node_prefix_total_len == 2048
        if hit_len == 2304:
            assert requests.req_to_token_indexs[fork.req_idx, 2048].item() != held[2048].item()
        restored = requests.req_to_token_indexs[fork.req_idx, :hit_len]
        for pool, ratio in ((manager.c4_pool, 4), (manager.c4_indexer_pool, 4), (manager.c128_pool, 128)):
            torch.testing.assert_close(
                pool.read(0, held[ratio - 1 : hit_len : ratio].long() // ratio),
                pool.read(0, restored[ratio - 1 : hit_len : ratio].long() // ratio),
            )
    assert set(requests._swa_pages[forks[0].req_idx].values()).isdisjoint(
        requests._swa_pages[forks[1].req_idx].values()
    )
    # Pause can publish a reusable prefix; abort with GPU placement disabled only dereferences it.
    forks[1].cache_tiers.clear()
    for fork in forks:
        freed = []
        g_infer_context.free_a_req_mem(freed, fork)
        manager.free(torch.cat(freed))
        requests.free_req(fork.req_idx)
    radix.free_radix_cache_to_get_enough_token(manager.size)
    assert manager.allocator.can_use_mem_size == manager.size
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages
    assert manager.big_page_buffers.get_free_cache_num() == manager.big_page_buffers.size
    assert small.get_free_cache_num() == small.size


def test_cpu_load_failure_releases_reserved_history_and_private_swa(cache, monkeypatch):
    from lightllm.server.router.model_infer.mode_backend import dsv4_multi_level_kv_cache as module

    manager, requests = cache
    req_idx = requests.alloc()
    cache_module = module.Dsv4MultiLevelKvCacheModule.__new__(module.Dsv4MultiLevelKvCacheModule)
    cache_module.backend = SimpleNamespace(
        is_master_in_dp=False, radix_cache=None, model=SimpleNamespace(mem_manager=manager, req_manager=requests)
    )
    cache_module.cpu_cache_client = SimpleNamespace()
    cache_module.init_sync_group = None
    req = SimpleNamespace(
        req_idx=req_idx,
        req_id=1,
        cur_kv_len=0,
        hold_kv_len=0,
        image_block_spans=[],
        shm_req=SimpleNamespace(
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: [0]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [2048]),
            disk_prompt_cache_len=0,
        ),
    )

    def fail(**kwargs):
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(manager.operator, "load_cpu_cache_pages", fail)
    monkeypatch.setattr(module.g_infer_context, "get_can_alloc_token_num", lambda: manager.allocator.can_use_mem_size)
    monkeypatch.setattr(
        module.g_infer_context, "get_can_alloc_dsv4_swa_page_num", lambda: manager.swa_page_allocator.can_use_mem_size
    )
    with pytest.raises(RuntimeError, match="injected copy failure"):
        cache_module.load_cpu_cache_to_reqs([req])
    assert req.cur_kv_len == req.hold_kv_len == 0
    assert manager.allocator.can_use_mem_size == manager.size
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages
    requests.free_req(req_idx)


def _checkpoint_transfer_worker(rank, rendezvous):
    from datetime import timedelta
    from sortedcontainers import SortedDict
    import torch.distributed as dist
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4CpuCacheLayout
    from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager
    from lightllm.server.router.model_infer.mode_backend.dp_backend.dp_shared_kv_trans import DPKVSharedMoudle
    from lightllm.server.router.model_infer.infer_batch import g_infer_context

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        buffers = DeepseekV4StateCacheManager(2, DeepseekV4CpuCacheLayout.from_compress_rates([4, 128]))
        if rank == 0:
            for value in (31, 32):
                buffers.buffer[buffers.alloc_one_state_cache()].fill_(value)
        module = DPKVSharedMoudle.__new__(DPKVSharedMoudle)
        module.backend = SimpleNamespace(
            args=SimpleNamespace(linear_att_hash_page_size=256, linear_att_page_block_num=8, max_req_total_len=8192),
            node_nccl_group=dist.group.WORLD,
            model=SimpleNamespace(mem_manager=SimpleNamespace(big_page_buffers=buffers)),
            radix_cache=SimpleNamespace(get_big_page_ids_by_node=lambda node: [0, 1]),
        )
        for start in (0, 2048):
            req = SimpleNamespace(
                req_id=42,
                cur_kv_len=4096 if rank == 0 else start,
                shared_kv_node=object(),
                hybrid_len_to_big_page_id=SortedDict(),
            )
            if rank == 1 and start:
                index = buffers.alloc_one_state_cache()
                buffers.buffer[index].fill_(31)
                req.hybrid_len_to_big_page_id[2048] = index
            g_infer_context.requests_mapping = {42: req}
            tasks = (
                []
                if rank == 0
                else [SimpleNamespace(max_kv_len_mem_manager_index=0, req=req, mem_indexes=range(4096 - start))]
            )
            module._transfer_dsv4_checkpoints(tasks)
            torch.cuda.synchronize()
            if rank == 1:
                assert list(req.hybrid_len_to_big_page_id) == [2048, 4096]
                for length, value in ((2048, 31), (4096, 32)):
                    assert buffers.buffer[req.hybrid_len_to_big_page_id[length]].eq(value).all()
                buffers.free_state_cache(list(req.hybrid_len_to_big_page_id.values()))
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices for NCCL")
def test_dp_checkpoint_transfer_between_processes(tmp_path):
    torch.multiprocessing.spawn(_checkpoint_transfer_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
