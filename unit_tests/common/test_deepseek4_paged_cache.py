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

    requests.prepare_prefill(cpu([req_idx]), cpu([0]), cpu([254]), slots[:254])
    for sequences in ([255, 256, 257, 258], [255, 256, 257, 258], [256, 257, 258, 259]):
        indexes = slots[cpu(sequences).cuda().long() - 1]
        requests.prepare_decode(cpu([req_idx] * 4), cpu(sequences), cpu([0, 1, 2, 3]), indexes)
        assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages - 3
        assert manager.swa_page_live_count.sum().item() == max(sequences)
        assert manager.allocator.can_use_mem_size == manager.size - 512
    requests.free([req_idx], slots)
    assert manager.allocator.can_use_mem_size == manager.size
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages
    assert manager.swa_page_live_count.eq(0).all()


def test_prefill_chunk_keeps_all_new_swa_rows(cache):
    manager, requests = cache
    req_idx, slots = _reserve(manager, requests, 2048)
    requests.prepare_prefill(torch.tensor([req_idx]), torch.tensor([0]), torch.tensor([2048]), slots)
    assert manager.full_to_swa_indexs[slots].ge(0).all()
    assert manager.swa_page_live_count.sum().item() == 2048
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
    assert manager.full_to_swa_indexs[dst].eq(-1).all()


def test_hold_page_and_dspark_scratch_have_separate_capacity(cache):
    manager, requests = cache
    hold = requests.req_to_token_indexs[requests.HOLD_REQUEST_ID]
    torch.testing.assert_close(
        hold[:256], torch.arange(manager.size, manager.size + 256, dtype=torch.int32, device="cuda")
    )
    assert manager.full_to_swa_indexs[hold].ge(manager.swa_size).all()
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
    requests.prepare_prefill(torch.tensor([req_idx]), torch.tensor([0]), torch.tensor([2048]), src)
    for pool in (manager.c4_pool, manager.c4_indexer_pool, manager.c128_pool, manager.swa_pool):
        pool.buffer.random_(0, 256)
    manager.c4_state_buffer.uniform_()
    manager.c4_indexer_state_buffer.uniform_()
    staging = torch.empty((1, manager.cpu_cache_layout.page_nbytes), dtype=torch.uint8, device="cuda")
    manager.operator.pack_cpu_cache_pages(src.view(1, -1), staging)
    cpu_page = torch.empty(staging.shape, dtype=torch.uint8, pin_memory=True)
    cpu_page.copy_(staging)
    plan = manager.prepare_cpu_cache_load(2048, 2048)
    manager.operator.load_cpu_cache_pages(
        plan, torch.tensor([0], dtype=torch.int32, device="cuda"), SimpleNamespace(cpu_kv_cache_tensor=cpu_page)
    )
    manager.commit_cpu_cache_load_plan(plan)
    for pool, ratio in ((manager.c4_pool, 4), (manager.c4_indexer_pool, 4), (manager.c128_pool, 128)):
        torch.testing.assert_close(
            pool.read(0, src[ratio - 1 :: ratio].long() // ratio),
            pool.read(0, plan.mem_indexes[ratio - 1 :: ratio].long() // ratio),
            rtol=0,
            atol=0,
        )
    src_swa = manager.full_to_swa_indexs[src[-256:]].long()
    dst_swa = manager.full_to_swa_indexs[plan.mem_indexes[-256:]].long()
    for layer in range(manager.layer_num):
        torch.testing.assert_close(
            manager.swa_pool.read(layer, src_swa), manager.swa_pool.read(layer, dst_swa), rtol=0, atol=0
        )
    manager.free(torch.cat([src, plan.mem_indexes]))
    assert manager.allocator.can_use_mem_size == manager.size
    assert manager.swa_page_allocator.can_use_mem_size == manager.swa_num_pages
