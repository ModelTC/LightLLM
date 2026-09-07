from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel import sliding_window_cpu_cache_copy as copy_kernels
from lightllm.common.kv_cache_mem_manager.operator import hybrid_sliding as operator_module
from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow
from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig, SlidingWindowStateCacheManager
from lightllm.server.router.model_infer.infer_batch import g_infer_context


def _cpu_pages(size, keep_num=0):
    pages = object.__new__(SlidingWindowStateCacheManager)
    pages.size, pages.keep_num = size, keep_num
    pages.state_cache = torch.empty((size, 1, 4, 2, 4), dtype=torch.float32)
    pages.clear_to_init_state()
    return pages


@pytest.fixture
def sliding_operator(monkeypatch):
    # Exercise page ownership and transfer orchestration on CPU; kernel tests
    # separately cover real CUDA pointers, byte layout, and stream ordering.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(
        operator_module,
        "get_env_start_args",
        lambda: SimpleNamespace(cpu_cache_token_page_size=8, linear_att_hash_page_size=2, linear_att_page_block_num=4),
    )
    monkeypatch.setattr(operator_module, "get_current_rank_in_dp", lambda: 0)
    monkeypatch.setattr(operator_module, "get_dp_world_size", lambda: 1)
    manager = SimpleNamespace(
        sliding_config=object(),
        kv_buffer=torch.zeros((1, 33, 2, 4)),
        linear_att_big_page_buffers=_cpu_pages(6, keep_num=2),
        CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID=4,
        CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID=5,
    )
    req_manager = object.__new__(ReqManagerForSlidingWindow)
    req_manager.mem_manager = manager
    req_manager.sliding_window = 4
    req_manager.req_to_sliding_window = torch.full((1, 2, 4, 2, 4), -1.0)
    small_pages = _cpu_pages(2)
    small_pages.get_state_cache(0).fill_(17)
    radix = SimpleNamespace(
        linear_att_small_page_buffers=small_pages,
        get_big_page_ids_by_node=lambda node: [] if node is None else node.big_page_ids.copy(),
    )
    monkeypatch.setattr(g_infer_context, "req_manager", req_manager)
    monkeypatch.setattr(g_infer_context, "radix_cache", radix)
    return operator_module.HybridSlidingMemOperator(manager), req_manager, small_pages


@pytest.mark.parametrize(
    "cached_tokens,token_num,expected_endpoints,has_tail",
    [(0, 6, {}, True), (0, 16, {8: 1, 16: 0}, False), (0, 22, {8: 1, 16: 0}, True), (8, 22, {16: 1, 24: 0}, True)],
)
def test_load_restores_last_checkpoint_without_owning_tail_staging_slot(
    monkeypatch, sliding_operator, cached_tokens, token_num, expected_endpoints, has_tail
):
    operator, req_manager, _ = sliding_operator
    page_num = (token_num + 7) // 8
    captures = []

    def load(**kwargs):
        captures.append(kwargs)
        for buffer_id, page_id in zip(kwargs["big_page_buffer_ids"].tolist(), kwargs["page_indexes"].tolist()):
            kwargs["gpu_sliding_state"][buffer_id].fill_(10 + page_id)

    monkeypatch.setattr(copy_kernels, "copy_cpu_cache_to_kv_buffer", load)
    req = SimpleNamespace(req_idx=1, cur_kv_len=cached_tokens + token_num, linear_att_len_to_big_page_id={})
    operator.load_cpu_cache_to_gpu(
        torch.arange(cached_tokens, cached_tokens + token_num, dtype=torch.int32),
        torch.arange(cached_tokens // 8, cached_tokens // 8 + page_num, dtype=torch.int32),
        SimpleNamespace(cpu_kv_cache_tensor=object()),
        req,
    )

    assert req.linear_att_len_to_big_page_id == expected_endpoints
    assert len(captures) == 1
    padded_indexes = captures[0]["mem_indexes"]
    assert padded_indexes.tolist() == list(range(cached_tokens, cached_tokens + token_num)) + [-1] * (
        page_num * 8 - token_num
    )
    assert 4 not in req.linear_att_len_to_big_page_id.values()
    assert operator.mem_manager.linear_att_big_page_buffers.get_free_cache_num() == 4 - token_num // 8
    if has_tail:
        assert captures[0]["big_page_buffer_ids"][-1].item() == 4
    torch.testing.assert_close(
        req_manager.req_to_sliding_window[:, 1],
        torch.full((1, 4, 2, 4), 10.0 + cached_tokens // 8 + page_num - 1),
        atol=0,
        rtol=0,
    )
    assert torch.all(req_manager.req_to_sliding_window[:, 0] == -1)


@pytest.mark.parametrize("token_num", [16, 22])
def test_offload_combines_shared_owned_and_tail_checkpoints(monkeypatch, sliding_operator, token_num):
    operator, _, small_pages = sliding_operator
    captures = []
    monkeypatch.setattr(copy_kernels, "copy_kv_buffer_to_cpu_cache", lambda **kwargs: captures.append(kwargs))
    big_pages = operator.mem_manager.linear_att_big_page_buffers
    big_pages.get_state_cache(1).fill_(11)
    big_pages.get_state_cache(3).fill_(13)
    req = SimpleNamespace(
        shared_kv_node=SimpleNamespace(big_page_ids=[1]),
        linear_att_len_to_big_page_id={16: 3},
        tail_linear_att_small_page_buffer_id=0 if token_num % 8 else None,
    )
    page_num = (token_num + 7) // 8
    ready = torch.tensor([True] + [False] * (page_num - 1))
    operator.offload_gpu_kv_to_cpu_cache(
        torch.arange(token_num, dtype=torch.int32),
        torch.arange(page_num, dtype=torch.int32),
        ready,
        SimpleNamespace(cpu_kv_cache_tensor=object()),
        req,
    )

    assert len(captures) == 1
    assert captures[0]["big_page_buffer_ids"].tolist() == ([1, 3, 5] if token_num % 8 else [1, 3])
    assert captures[0]["mem_indexes"].tolist() == list(range(token_num)) + [-1] * (page_num * 8 - token_num)
    assert captures[0]["page_readies"] is ready
    assert req.linear_att_len_to_big_page_id == {16: 3}
    assert torch.count_nonzero(big_pages.get_state_cache(4)) == 0
    if token_num % 8:
        torch.testing.assert_close(big_pages.get_state_cache(5), small_pages.get_state_cache(0), atol=0, rtol=0)
        small_pages.get_state_cache(0).zero_()
        assert torch.all(big_pages.get_state_cache(5) == 17)


def test_reserved_state_slots_are_never_allocated_or_freed():
    pages = _cpu_pages(4, keep_num=2)
    assert pages.alloc_state_cache(2) == [0, 1]
    assert pages.alloc_one_state_cache() is None
    for reserved_id in [2, 3]:
        with pytest.raises(AssertionError):
            pages.free_state_cache([reserved_id])
    pages.free_state_cache([0, 1])
    assert pages.get_free_cache_num() == 2
    assert pages.get_used_cache_num() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cpu_transfers_reuse_separate_load_and_offload_slots_across_streams(monkeypatch):
    request_num, page_size, tail_len, window = 8, 8, 6, 4
    config = SlidingWindowCacheConfig({0: 0}, {1: 0}, window, 1, 8, 1, 8, torch.bfloat16)
    full_bytes = config.get_cpu_cache_full_att_bytes(page_size, 1)
    state_bytes = config.get_cpu_cache_state_bytes(1)
    cpu_cache = torch.zeros(
        (request_num * 2, config.get_cpu_cache_big_page_bytes(page_size, 1)),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )

    def full_page(page_id):
        return cpu_cache[page_id, :full_bytes].view(config.dtype).view(page_size, 1, 2, 8)

    def window_page(page_id):
        return (
            cpu_cache[page_id, full_bytes : full_bytes + state_bytes].view(config.dtype).view(config.get_state_shape())
        )

    # The load stream reads already-ready pages, while offload writes disjoint
    # CPU pages. Both directions share the same big-state pool, as in serving.
    for req_idx in range(request_num):
        full_page(req_idx).fill_(10 + req_idx)
        window_page(req_idx).fill_(200 + req_idx)

    big_pages = SlidingWindowStateCacheManager(2, config, keep_num=2)
    small_pages = SlidingWindowStateCacheManager(request_num, config)
    manager = SimpleNamespace(
        sliding_config=config,
        kv_buffer=torch.zeros((1, request_num * tail_len * 2, 2, 8), dtype=config.dtype, device="cuda"),
        linear_att_big_page_buffers=big_pages,
        CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID=0,
        CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID=1,
    )
    req_manager = object.__new__(ReqManagerForSlidingWindow)
    req_manager.mem_manager, req_manager.sliding_window = manager, window
    req_manager.req_to_sliding_window = torch.zeros((1, request_num, window, 2, 8), dtype=config.dtype, device="cuda")
    monkeypatch.setattr(g_infer_context, "req_manager", req_manager)
    monkeypatch.setattr(
        g_infer_context,
        "radix_cache",
        SimpleNamespace(linear_att_small_page_buffers=small_pages, get_big_page_ids_by_node=lambda node: []),
    )
    monkeypatch.setattr(
        operator_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            cpu_cache_token_page_size=page_size, linear_att_hash_page_size=2, linear_att_page_block_num=4
        ),
    )
    monkeypatch.setattr(operator_module, "get_current_rank_in_dp", lambda: 0)
    monkeypatch.setattr(operator_module, "get_dp_world_size", lambda: 1)
    operator = operator_module.HybridSlidingMemOperator(manager)
    client = SimpleNamespace(cpu_kv_cache_tensor=cpu_cache)
    transfers = []
    for req_idx in range(request_num):
        start = req_idx * tail_len
        manager.kv_buffer[:, start : start + tail_len].fill_(30 + req_idx)
        small_page_id = small_pages.alloc_one_state_cache()
        small_pages.get_state_cache(small_page_id).fill_(100 + req_idx)
        req = SimpleNamespace(
            req_idx=req_idx,
            cur_kv_len=tail_len,
            shared_kv_node=None,
            linear_att_len_to_big_page_id={},
            tail_linear_att_small_page_buffer_id=small_page_id,
        )
        source_indexes = torch.arange(start, start + tail_len, dtype=torch.int32, device="cuda")
        load_indexes = source_indexes + request_num * tail_len
        load_page = torch.tensor([req_idx], dtype=torch.int32, device="cuda")
        offload_page = torch.tensor([request_num + req_idx], dtype=torch.int32, device="cuda")
        ready = torch.tensor([False], dtype=torch.bool, device="cuda")
        transfers.append((req, source_indexes, load_indexes, load_page, offload_page, ready))

    offload_stream, load_stream = torch.cuda.Stream(), torch.cuda.Stream()
    offload_stream.wait_stream(torch.cuda.current_stream())
    load_stream.wait_stream(torch.cuda.current_stream())
    for req, source_indexes, load_indexes, load_page, offload_page, ready in transfers:
        with torch.cuda.stream(offload_stream):
            operator.offload_gpu_kv_to_cpu_cache(source_indexes, offload_page, ready, client, req)
        with torch.cuda.stream(load_stream):
            operator.load_cpu_cache_to_gpu(load_indexes, load_page, client, req)
    # No per-request wait: each reserved slot has been reused eight times.
    offload_stream.synchronize()
    load_stream.synchronize()

    for req_idx, (req, source_indexes, load_indexes, _, _, _) in enumerate(transfers):
        assert req.linear_att_len_to_big_page_id == {}
        torch.testing.assert_close(
            manager.kv_buffer[:, load_indexes],
            torch.full((1, tail_len, 2, 8), 10 + req_idx, dtype=config.dtype, device="cuda"),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            req_manager.req_to_sliding_window[:, req_idx],
            torch.full(config.get_state_shape(), 200 + req_idx, dtype=config.dtype, device="cuda"),
            atol=0,
            rtol=0,
        )
        assert torch.all(full_page(request_num + req_idx)[:tail_len] == 30 + req_idx)
        assert torch.count_nonzero(full_page(request_num + req_idx)[tail_len:]) == 0
        assert torch.all(window_page(request_num + req_idx) == 100 + req_idx)
        assert torch.all(full_page(req_idx) == 10 + req_idx)
        assert torch.all(window_page(req_idx) == 200 + req_idx)
        assert torch.all(manager.kv_buffer[:, source_indexes] == 30 + req_idx)
    assert big_pages.get_free_cache_num() == 0
    assert big_pages.alloc_one_state_cache() is None
    for slot in [manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID, manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID]:
        with pytest.raises(AssertionError):
            big_pages.free_state_cache([slot])
