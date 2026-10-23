import random
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import allocator as impl
from lightllm.common.kv_cache_mem_manager import mem_manager as mem_impl


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires pinned CPU allocations")


class _SharedInt:
    def __init__(self, name):
        self.name = name
        self.value = None

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value


@pytest.fixture
def make_allocator(monkeypatch):
    monkeypatch.setattr(impl, "SharedInt", _SharedInt)
    monkeypatch.setattr(impl, "get_current_rank_in_node", lambda: 0)
    return impl.KvCacheAllocator


def assert_capacity(allocator, available_tokens):
    assert allocator.can_use_mem_size == available_tokens
    assert allocator.shared_can_use_token_num.get_value() == available_tokens
    assert (allocator.mark_end - allocator.mark_start) * allocator.page_size == available_tokens


@pytest.mark.parametrize("page_size", [1, 3, 4, 64])
def test_compressed_page_state_returns_complete_pinned_token_indexes(make_allocator, page_size):
    allocator = make_allocator(5 * page_size, page_size)
    assert allocator.mem_state.tolist() == [i * page_size for i in range(5)]
    assert allocator.mem_state.dtype == torch.int32 and allocator.mem_state.is_pinned()
    assert allocator.mem_state.numel() == allocator.size // page_size
    assert allocator._return_buffer is None
    assert_capacity(allocator, 5 * page_size)

    first = allocator.alloc(2 * page_size)
    second = allocator.alloc(page_size)
    assert first.tolist() == list(range(2 * page_size))
    assert second.tolist() == list(range(2 * page_size, 3 * page_size))
    assert first.dtype == torch.int32 and first.is_pinned()
    assert first.untyped_storage().data_ptr() == allocator._return_buffer.untyped_storage().data_ptr()
    assert first.untyped_storage().nbytes() == 1024 * 1024 * first.element_size()
    assert allocator._return_buffer.numel() == 1024 * 1024 - 3 * page_size
    assert second.data_ptr() == first.data_ptr() + first.numel() * first.element_size()
    assert first.untyped_storage().data_ptr() != allocator.mem_state.untyped_storage().data_ptr()
    assert_capacity(allocator, 2 * page_size)


@pytest.mark.parametrize("page_size", [1, 3, 4, 64])
@pytest.mark.parametrize("index_kind", ["list", "cpu", "cuda"])
def test_nonadjacent_pages_are_recycled_from_token_indexes(make_allocator, page_size, index_kind):
    allocator = make_allocator(5 * page_size, page_size)
    indexes = allocator.alloc(5 * page_size).clone()
    # Recycle physical pages 3 and 1, in that order, using full token-level indexes.
    released = torch.cat([indexes[3 * page_size : 4 * page_size], indexes[page_size : 2 * page_size]])
    expected = released.clone()
    if index_kind == "list":
        released = released.tolist()
    elif index_kind == "cuda":
        released = released.cuda()
    allocator.free(released)
    assert_capacity(allocator, 2 * page_size)
    assert allocator.mem_state[allocator.mark_start :].tolist() == [3 * page_size, page_size]

    # free() owns the compressed indexes even if the caller reuses its input buffer.
    if isinstance(released, torch.Tensor):
        released.fill_(-1)
    else:
        released[:] = [-1] * len(released)
    recycled = allocator.alloc(2 * page_size)
    assert torch.equal(recycled, expected)
    assert_capacity(allocator, 0)
    with pytest.raises(AssertionError, match="error alloc state"):
        allocator.alloc(page_size)
    assert_capacity(allocator, 0)


@pytest.mark.parametrize("page_size", [1, 3, 64])
def test_return_buffer_replacement_and_free_preserve_retained_results(make_allocator, page_size):
    # A full slab for aligned page sizes; page_size=3 leaves one unusable element.
    size = 1024 * 1024 // page_size * page_size
    allocator = make_allocator(size, page_size)
    pointers, retained = [], []
    for iteration in range(4):
        indexes = allocator.alloc(size)
        pointers.append(indexes.data_ptr())
        for old, snapshot in retained:
            assert torch.equal(old, snapshot)
        snapshot = indexes.clone()
        retained.append((indexes, snapshot))
        allocator.free(indexes.view(-1, page_size).flip(0).reshape(-1))
        assert torch.equal(indexes, snapshot)
        assert_capacity(allocator, size)
    assert len(set(pointers)) == 4
    assert allocator._return_buffer.numel() == 1024 * 1024 - size


@pytest.mark.parametrize("page_size", [1, 3, 64])
def test_large_requests_allocate_separately_and_preserve_small_buffer(make_allocator, page_size):
    large_size = (1024 * 1024 // page_size + 1) * page_size
    allocator = make_allocator(large_size + page_size, page_size)
    large = allocator.alloc(large_size)
    snapshot = large.clone()
    assert allocator._return_buffer is None
    assert large.is_pinned() and large.dtype == torch.int32
    assert large.untyped_storage().nbytes() == large_size * large.element_size()
    assert large.tolist() == list(range(large_size))

    small = allocator.alloc(page_size)
    remaining = allocator._return_buffer
    assert remaining.numel() == 1024 * 1024 - page_size
    assert small.tolist() == list(range(large_size, large_size + page_size))
    allocator.free(large.view(-1, page_size).flip(0).reshape(-1))
    recycled = allocator.alloc(large_size)
    assert recycled.data_ptr() != large.data_ptr()
    assert allocator._return_buffer is remaining
    assert torch.equal(recycled, snapshot.view(-1, page_size).flip(0).reshape(-1))
    assert torch.equal(large, snapshot)
    assert_capacity(allocator, 0)


@pytest.mark.parametrize("page_size", [1, 3, 4, 64])
def test_free_all_restores_page_stride_and_token_capacity(make_allocator, page_size):
    allocator = make_allocator(6 * page_size, page_size)
    indexes = allocator.alloc(6 * page_size)
    allocator.free(indexes.view(-1, page_size).flip(0).reshape(-1))
    allocator.alloc(page_size)
    remaining = allocator._return_buffer
    allocator.free_all()
    assert allocator.mem_state.tolist() == [i * page_size for i in range(6)]
    assert allocator._return_buffer is remaining
    assert_capacity(allocator, 6 * page_size)
    assert allocator.alloc(6 * page_size).tolist() == list(range(6 * page_size))


@pytest.mark.parametrize("page_size", [1, 3, 4, 64])
def test_resize_and_empty_operations_preserve_page_size(make_allocator, page_size):
    allocator = make_allocator(3 * page_size, page_size)
    previous = allocator.alloc(page_size)
    snapshot = previous.clone()
    for page_count in [7, 2, 0, 4]:
        allocator.resize(page_count * page_size)
        assert allocator.mem_state.tolist() == [i * page_size for i in range(page_count)]
        assert allocator._return_buffer is None
        assert torch.equal(previous, snapshot)
        assert_capacity(allocator, page_count * page_size)
        assert allocator.alloc(0).numel() == 0
        assert allocator._return_buffer is None
        allocator.free([])
        allocator.free(torch.empty(0, dtype=torch.int32))
        assert_capacity(allocator, page_count * page_size)
        assert allocator.alloc(page_count * page_size).tolist() == list(range(page_count * page_size))
        assert_capacity(allocator, 0)
        allocator.free_all()
        assert_capacity(allocator, page_count * page_size)


@pytest.mark.parametrize("size,page_size", [(4, 0), (4, -1), (-4, 4), (5, 4)])
def test_invalid_constructor_sizes_are_rejected(make_allocator, size, page_size):
    with pytest.raises(AssertionError):
        make_allocator(size, page_size)


def test_invalid_operations_leave_allocation_state_unchanged(make_allocator):
    allocator = make_allocator(16, 4)
    allocator.alloc(4)
    for operation in [
        lambda: allocator.alloc(-4),
        lambda: allocator.alloc(1),
        lambda: allocator.alloc(16),
        lambda: allocator.free([0, 1, 2]),
        lambda: allocator.free(list(range(8))),
        lambda: allocator.resize(17),
    ]:
        before = allocator.mem_state.clone()
        remaining = allocator._return_buffer
        with pytest.raises(AssertionError):
            operation()
        assert torch.equal(allocator.mem_state, before)
        assert allocator._return_buffer is remaining
        assert allocator.size == 16
        assert_capacity(allocator, 12)


@pytest.mark.parametrize("page_size", [1, 3, 4, 64])
def test_randomized_alloc_free_never_overlaps_live_pages(make_allocator, page_size):
    page_count = 23
    allocator = make_allocator(page_count * page_size, page_size)
    rng = random.Random(7341 + page_size)
    live_pages = set()
    for _ in range(300):
        available = page_count - len(live_pages)
        if available and (not live_pages or rng.random() < 0.6):
            count = rng.randint(1, min(available, 5))
            indexes = allocator.alloc(count * page_size).clone()
            blocks = indexes.view(-1, page_size)
            bases = blocks[:, 0].tolist()
            assert len(set(bases)) == count
            assert all(0 <= base < allocator.size and base % page_size == 0 for base in bases)
            assert live_pages.isdisjoint(bases)
            assert torch.equal(blocks, blocks[:, :1] + torch.arange(page_size, dtype=torch.int32))
            live_pages.update(bases)
        else:
            bases = rng.sample(sorted(live_pages), rng.randint(1, min(len(live_pages), 5)))
            indexes = [base + offset for base in bases for offset in range(page_size)]
            allocator.free(indexes if rng.random() < 0.5 else torch.tensor(indexes, dtype=torch.int32))
            live_pages.difference_update(bases)
        assert_capacity(allocator, (page_count - len(live_pages)) * page_size)
    allocator.free([base + offset for base in sorted(live_pages) for offset in range(page_size)])
    assert_capacity(allocator, allocator.size)
    assert sorted(allocator.alloc(allocator.size).tolist()) == list(range(allocator.size))


@pytest.mark.parametrize("page_size", [1, 3, 4, 64])
def test_memory_manager_passes_page_size_and_resizes_token_capacity(make_allocator, monkeypatch, page_size):
    monkeypatch.setattr(mem_impl, "get_env_start_args", lambda: SimpleNamespace(page_size=page_size))
    manager = mem_impl.MemoryManager(5 * page_size + page_size - 1, torch.float16, 1, 8, 1)
    assert manager.size == 5 * page_size
    assert manager.allocator.page_size == page_size
    assert manager.allocator.mem_state.tolist() == [i * page_size for i in range(5)]
    assert manager.HOLD_TOKEN_MEMINDEXES == tuple(range(5 * page_size, 6 * page_size))
    indexes = manager.alloc(2 * page_size)
    manager.free(indexes)
    assert_capacity(manager.allocator, manager.size)

    manager.resize_mem(3 * page_size + page_size - 1)
    assert manager.size == 3 * page_size
    assert manager.allocator.mem_state.tolist() == [i * page_size for i in range(3)]
    assert manager.HOLD_TOKEN_MEMINDEXES == tuple(range(3 * page_size, 4 * page_size))
    indexes = manager.alloc(manager.size)
    assert indexes.tolist() == list(range(manager.size))
    manager.free(indexes.cuda())
    assert_capacity(manager.allocator, manager.size)


@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.parametrize("page_size", [1, 64])
def test_request_free_indexes_recycle_pages_after_reset_batch_and_slot_reuse(
    make_allocator, monkeypatch, mode, page_size
):
    # 延迟导入，沿用在线推理初始化顺序，不增加另一套请求/allocator 夹具。
    from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
    from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext
    from lightllm.server.router.model_infer.infer_req_ext1 import PinMemPageCacheReqTokenIndex, create_req_token_index

    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_page_offsets", None)
    allocator = make_allocator(6 * page_size, page_size)
    allocated = allocator.alloc(allocator.size)
    first = torch.cat([allocated[4 * page_size : 5 * page_size], allocated[page_size : 2 * page_size]]).pin_memory()
    second = allocated[3 * page_size : 4 * page_size]
    expected = torch.cat([first, second])
    table = torch.full((2, 2 * page_size), -1, dtype=torch.int32, device="cuda")
    manager = SimpleNamespace(req_to_token_indexs=table, page_size=page_size)
    context = InferenceContext.__new__(InferenceContext)
    context.req_manager = manager
    context.radix_cache = None
    reqs = []

    def unexpected_snapshot(*args):
        pytest.fail("The pure_gpu release path must not copy a full token index snapshot to CPU first")

    with torch.cuda.stream(torch.cuda.default_stream()):
        for row, indexes in enumerate((first, second)):
            req = InferReq.__new__(InferReq)
            req.req_idx = row
            req.req_manager = manager
            req.args = SimpleNamespace(req_token_index_mode=mode)
            req.cur_kv_len = req.hold_kv_len = indexes.numel()
            req.shm_req = SimpleNamespace(input_len=req.cur_kv_len, shm_cur_kv_len=req.cur_kv_len)
            req.token_index = create_req_token_index(req, index_stream=torch.cuda.default_stream())
            req.token_index.append_token_indexes(indexes)
            if mode == "pure_gpu":
                monkeypatch.setattr(req.token_index, "get_cpu_token_indexes", unexpected_snapshot)
            reqs.append(req)

        # 混合空区间和非空区间；GPU 模式的空区间也必须留在 GPU，才能直接 cat。
        torch.cuda._sleep(10_000_000)
        released = [reqs[0].token_index.get_free_token_indexes(0, 0)]
        for req in reqs:
            context.free_a_req_mem(released, req)
            assert req.token_index.token_index_len == req.token_index.gpu_index_submitted_len == 0
        assert released[0].numel() == 0
        assert all(indexes.device.type == ("cuda" if mode == "pure_gpu" else "cpu") for indexes in released)
        merged = torch.cat(released)

        # 默认流先合并旧视图，再重写请求槽位；allocator 只能回收已合并的旧索引。
        # 不在 free 前同步或 D2H，真实 free 必须等待默认流并按 page_size 抽取页首。
        table.fill_(-99)
        allocator.free(merged)
        assert_capacity(allocator, 3 * page_size)
        assert allocator.mem_state[allocator.mark_start :].tolist() == [4 * page_size, page_size, 3 * page_size]
        assert allocator.alloc(3 * page_size).tolist() == expected.tolist()
        assert_capacity(allocator, 0)


@pytest.mark.parametrize("page_size", [1, 64])
def test_pure_gpu_free_waits_for_index_stream_before_allocator_d2h(make_allocator, page_size):
    from lightllm.server.router.model_infer.infer_req_ext1 import PureGpuReqTokenIndex

    allocator = make_allocator(4 * page_size, page_size)
    indexes = allocator.alloc(2 * page_size)
    table = torch.full((1, indexes.numel()), -1, dtype=torch.int32, device="cuda")
    index_stream = torch.cuda.Stream()
    index_stream.wait_stream(torch.cuda.current_stream())
    req = SimpleNamespace(req_idx=0, req_manager=SimpleNamespace(req_to_token_indexs=table))
    helper = PureGpuReqTokenIndex(req, index_stream=index_stream)

    with torch.cuda.stream(index_stream):
        # 延迟索引写入；主线程不等待，默认流必须通过释放 getter 承接写入依赖。
        torch.cuda._sleep(100_000_000)
        helper.append_token_indexes(indexes)

    free_indexes = helper.get_free_token_indexes(0, indexes.numel())
    assert free_indexes.data_ptr() == table[0].data_ptr()
    # 不先同步索引流：真实 allocator.free 在当前流按页取首索引并阻塞 D2H。
    allocator.free(free_indexes)
    helper.reset()
    with torch.cuda.stream(index_stream):
        table.fill_(-99)
    assert_capacity(allocator, allocator.size)
    assert allocator.alloc(indexes.numel()).tolist() == indexes.tolist()
    index_stream.synchronize()
