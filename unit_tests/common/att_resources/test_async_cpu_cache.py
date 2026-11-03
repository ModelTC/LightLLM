"""Asynchronous cache copies, writer ownership and request-independent lifetimes."""

from collections import deque
from types import SimpleNamespace as NS

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.allocators import StatePageAllocator
from lightllm.common.attention_state.full_att import GQAFullAttentionState, GQAAttentionConfig
from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as impl
from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import (
    MultiLevelKvCacheModule,
    CacheSession,
    StagingSlot,
    LoadTask,
    StoreTask,
    CheckpointTask,
)
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from unit_tests.common.att_resources.test_state_identity import allocated_state, KINDS
from unit_tests.common.att_resources.test_dsv4_state import make_full
from unit_tests.common.att_resources.test_model_resources import make_resources
from unit_tests.common.att_resources.model_helpers import full_state_test_buffers


def test_allocator_lifetime():
    a = StatePageAllocator(2)
    slots = a.alloc(2)
    a.retain(slots)
    a.retain([slots[0]])
    a.free(slots)  # request exits
    assert a.free_count == 0
    a.free(slots)  # first copy finishes
    assert a.free_count == 1 and a.alloc(1) == [slots[1]]
    a.free([slots[0]])
    assert a.free_count == 1 and a.alloc(1) == [slots[0]]
    with pytest.raises(AssertionError):
        a.retain([2])


@pytest.mark.parametrize("kind", [*KINDS, "c4", "c128"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staged_full_pages_match_direct(kind, cpu_args):
    cpu_args.cpu_cache_token_page_size = 256 if kind.startswith("c") else 8
    n = cpu_args.cpu_cache_token_page_size
    if kind.startswith("c"):
        state = make_full(4 if kind == "c4" else 128, device="cuda")
        state.allocate_storage(3 * n)
    else:
        state = allocated_state(kind, device="cuda")
    for b in ({"storage": state.storage} if kind.startswith("c") else full_state_test_buffers(state)).values():
        b.view(torch.uint8).random_(0, 256)
    owner = AttentionStateManager(request_capacity=2)
    owner.register_state("main", state)
    owner.finish_state_registration()
    op = owner.get_cpu_cache_operator()
    size = op.get_page_bytes()
    direct = torch.full((3, size), 91, dtype=torch.uint8, pin_memory=True)
    actual = torch.full_like(direct, 91, pin_memory=True)
    stage = torch.full((2, size), 91, dtype=torch.uint8, device="cuda")
    # Non-contiguous CPU page order and a partial final page.
    ids = torch.arange(n + n // 2, dtype=torch.int64, device="cuda")
    if not kind.startswith("c"):
        ids = ids % 8
    op.write_req(ids, direct, [2, 0], op.get_page_layouts())
    op.write_req(ids, stage, [0, 1], op.get_page_layouts())
    op.scatter_staging_pages(stage, actual, [2, 0])
    torch.cuda.synchronize()
    # Unused tail bytes are not part of the cached prefix.
    for region in op.get_page_layouts()[state.state_name].regions:
        for p, length in ((2, n), (0, n // 2)):
            a = actual[p, region.byte_start : region.byte_end].view(region.dtype).view(region.shape)
            b = direct[p, region.byte_start : region.byte_end].view(region.dtype).view(region.shape)
            slices = [slice(None)] * len(region.axes)
            slices[region.axes.index("token")] = slice(0, length)
            assert torch.equal(a[tuple(slices)].view(torch.uint8), b[tuple(slices)].view(torch.uint8))
    assert (actual[1] == 91).all()


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scatter_does_not_overwrite_other_tp_shards(replicated, cpu_args):
    cpu_args.cpu_cache_token_page_size = 8
    owners = []
    for rank in range(4):
        owner = AttentionStateManager(request_capacity=1)
        state = GQAFullAttentionState(
            GQAAttentionConfig(
                logical_layer_groups=((0,),),
                global_head_num=2,
                head_dim=16,
                dtype=torch.bfloat16,
                tp_rank=rank,
                tp_world_size=4,
                replicated=replicated,
            ),
            page_size=4,
            device="cuda",
        )
        state.allocate_storage(8)
        state.k_storage.fill_(7 + rank)
        state.v_storage.fill_(17 + rank)
        owner.register_state("main", state)
        owner.finish_state_registration()
        owners.append(owner)
    size = owners[0].get_cpu_cache_operator().get_page_bytes()
    actual = torch.full((1, size), 91, dtype=torch.uint8, pin_memory=True)
    direct = torch.full_like(actual, 91, pin_memory=True)
    for owner in owners:
        op = owner.get_cpu_cache_operator()
        stage = torch.full((1, size), 203, dtype=torch.uint8, device="cuda")
        op.write_req(torch.arange(8, device="cuda"), direct, [0], op.get_page_layouts())
        op.write_req(torch.arange(8, device="cuda"), stage, [0], op.get_page_layouts())
        op.scatter_staging_pages(stage, actual, [0])
        torch.cuda.synchronize()
    assert torch.equal(actual, direct)


@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staged_bounded_states_and_main_draft(glm, tp, cpu_args):
    cpu_args.cpu_cache_token_page_size = 8
    direct = actual = None
    for rank in range(tp):
        owner, _ = make_resources(glm=glm, tp=tp, rank=rank, mtp=2, draft_layers=2)
        for state in owner.get_full_attention_states():
            for b in full_state_test_buffers(state).values():
                b.view(torch.uint8).random_(0, 256)
        for state in owner.get_bounded_attention_states():
            if state.get_cpu_transfer_regions():
                pool = state.big_pages_pool
                for attr in ("buffer", "conv_state", "ssm_state"):
                    b = getattr(pool, attr, None)
                    if b is not None:
                        b.view(torch.uint8).random_(0, 256)
        op = owner.get_cpu_cache_operator()
        if direct is None:
            direct = torch.full((1, op.get_page_bytes()), 91, dtype=torch.uint8, pin_memory=True)
            actual = torch.full_like(direct, 91, pin_memory=True)
        stage = torch.full((1, op.get_page_bytes()), 203, dtype=torch.uint8, device="cuda")
        kw = dict(
            token_indices=torch.arange(8, device="cuda"),
            cpu_page_indices=[0],
            layouts=op.get_page_layouts(),
            big_page_slot_ids=[0],
        )
        op.write_req(cpu_page_buffers=direct, **kw)
        op.write_req(cpu_page_buffers=stage, **kw)
        op.scatter_staging_pages(stage, actual, [0])
        torch.cuda.synchronize()
    for layout in op.get_page_layouts().values():
        for region in layout.regions:
            assert torch.equal(
                actual[:, region.byte_start : region.byte_end], direct[:, region.byte_start : region.byte_end]
            )


class FakeEvent:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


class Lock:
    def acquire_sleep1ms(self):
        pass

    def release(self):
        pass


class Client:
    def __init__(self):
        self.lock = Lock()
        self.refs = {}
        self.hashes = {}
        self.ready = set()
        self.disks = []
        self.capacity = 64

    def allocate_one_page(self, page_items, hash_key, disk_offload_enable):
        if hash_key not in self.hashes:
            if len(self.hashes) >= self.capacity:
                return None, False
            self.hashes[hash_key] = len(self.hashes)
        p = self.hashes[hash_key]
        self.refs[p] = self.refs.get(p, 0) + 1
        return p, p in self.ready

    allocate_pages = CpuKvCacheClient.allocate_pages
    allocate_pages_for_write = CpuKvCacheClient.allocate_pages_for_write

    def allocate_pages_with_ownership(self, hashes, disk_offload_enable):
        result = self.allocate_pages_for_write(hashes, disk_offload_enable)
        return result.page_indices, result.writer_flags

    def deref_pages(self, pages):
        for p in pages:
            assert self.refs[p] > 0
            self.refs[p] -= 1

    def update_pages_status_to_ready(self, pages, deref=True, disk_offload_enable=False, token_num_in_page_list=None):
        self.ready.update(pages)
        if disk_offload_enable:
            self.disks.append((list(pages), token_num_in_page_list))
        if deref:
            self.deref_pages(pages)

    def check_allpages_ready(self, pages):
        return set(pages) <= self.ready

    @property
    def page_hash_dict(self):
        return self.hashes

    @property
    def page_items(self):
        return NS(linked_items=None)


def module(monkeypatch):
    m = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    m.backend = NS(is_master_in_dp=True, dp_world_size=1, is_hybrid_att_model=False)
    m.cpu_cache_client = Client()
    m.cache_sessions = {}
    m.load_tasks = deque()
    m.store_tasks = deque()
    m.checkpoint_tasks = deque()
    m.filter_group = object()
    monkeypatch.setattr(impl.dist, "broadcast_object_list", lambda *a, **kw: None)
    return m


def test_duplicate_loading_page_has_single_writer():
    c = Client()
    assert c.allocate_pages_with_ownership([10, 20], False) == ([0, 1], [True, True])
    assert c.allocate_pages_with_ownership([10, 20], False) == ([0, 1], [False, False])
    assert c.allocate_pages([10], False) == ([0], [False])
    c.update_pages_status_to_ready([0], deref=False)
    assert c.allocate_pages([10], False) == ([0], [True])
    c.capacity = 2
    assert c.allocate_pages_with_ownership([20, 30, 40], False) == ([1, -1, -1], [False, False, False])


@pytest.mark.parametrize("disk", [False, True])
def test_abort_detaches_request_but_preserves_pending_copy(monkeypatch, disk):
    m = module(monkeypatch)
    pages, _ = m.cpu_cache_client.allocate_pages_with_ownership([10, 20], False)
    session = CacheSession(1, disk_pages=pages, page_count=len(pages), token_num=16, pending_tasks=1, disk_offload=disk)
    m.cache_sessions[1] = session
    a = StatePageAllocator(1)
    slots = a.alloc(1)
    a.retain(slots)
    a.free(slots)
    event = FakeEvent()
    slot = StagingSlot(None, True)
    m.store_tasks.append(StoreTask(pages, session, slot, event))
    m.checkpoint_tasks.append(CheckpointTask(event, [(a, slots)]))
    m.close_cache_sessions([NS(req_id=1)])
    m._poll_async_tasks()
    assert m.cpu_cache_client.refs == {0: 1, 1: 1} and not m.cpu_cache_client.ready
    assert a.alloc(1) is None and slot.in_use
    event.ready = True
    m._poll_async_tasks()
    assert m.cpu_cache_client.ready == {0, 1} and m.cpu_cache_client.refs == {0: 0, 1: 0}
    assert a.alloc(1) == slots and not slot.in_use and not m.cache_sessions
    assert bool(m.cpu_cache_client.disks) == disk
    m._poll_async_tasks()  # no double free


def test_disk_waits_for_other_writer(monkeypatch):
    m = module(monkeypatch)
    pages, _ = m.cpu_cache_client.allocate_pages_with_ownership([10, 20], False)
    m.cache_sessions[1] = CacheSession(
        1, disk_pages=pages, page_count=len(pages), token_num=16, closing=True, disk_offload=True
    )
    m.cpu_cache_client.ready.add(0)
    m._poll_async_tasks()
    assert 1 in m.cache_sessions and not m.cpu_cache_client.disks
    m.cpu_cache_client.ready.add(1)
    m._poll_async_tasks()
    assert m.cpu_cache_client.disks == [([0, 1], 16)] and not m.cache_sessions


def test_restore_pages_survive_request_exit(monkeypatch):
    m = module(monkeypatch)
    pages, _ = m.cpu_cache_client.allocate_pages_with_ownership([10], False)
    a = StatePageAllocator(1)
    slots = a.alloc(1)
    a.retain(slots)
    a.free(slots)
    e = FakeEvent()
    m.load_tasks.append(LoadTask(pages, e))
    m.checkpoint_tasks.append(CheckpointTask(e, [(a, slots)]))
    m.cache_sessions[1] = CacheSession(1, closing=True)
    m._poll_async_tasks()
    assert not m.cache_sessions and m.cpu_cache_client.refs[0] == 1 and a.alloc(1) is None
    e.ready = True
    m._poll_async_tasks()
    assert m.cpu_cache_client.refs[0] == 0 and a.alloc(1) == slots


@pytest.mark.parametrize("pages_num", [1, 2, 4, 9])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_offload_allows_source_reuse_and_limits_staging(monkeypatch, cpu_args, pages_num):
    n = 8
    cpu_args.cpu_cache_token_page_size = n
    m = module(monkeypatch)
    m.args = NS(cpu_cache_token_page_size=n, enable_disk_cache=True)
    m.async_cache = True
    m.incremental_offload = False
    m.gloo_group = object()
    owner = AttentionStateManager(request_capacity=1)
    state = GQAFullAttentionState(
        GQAAttentionConfig(logical_layer_groups=((0,),), global_head_num=1, head_dim=16, dtype=torch.bfloat16),
        page_size=4,
        device="cuda",
    )
    state.allocate_storage(n * pages_num)
    state.k_storage.fill_(42)
    state.v_storage.fill_(67)
    owner.register_state("main", state)
    owner.finish_state_registration()
    m.backend.model = NS(
        att_state_manager=owner,
        req_manager=NS(req_to_token_indexs=torch.arange(n * pages_num, device="cuda").unsqueeze(0)),
    )
    m.cpu_cache_operator = owner.get_cpu_cache_operator()
    m.cpu_cache_client.cpu_kv_cache_tensor = torch.empty(
        (64, m.cpu_cache_operator.get_page_bytes()), dtype=torch.uint8, pin_memory=True
    )
    m.staging_slots = [
        StagingSlot(torch.empty((4, m.cpu_cache_operator.get_page_bytes()), dtype=torch.uint8, device="cuda"))
        for _ in range(2)
    ]
    producer = torch.cuda.Stream()
    cpu = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    monkeypatch.setattr(impl.g_infer_context, "get_overlap_stream", lambda: producer)
    monkeypatch.setattr(impl.g_infer_context, "get_cpu_kv_cache_stream", lambda: cpu)
    monkeypatch.setattr(impl.g_infer_context, "is_hybrid_att_model", False)
    req = NS(
        req_id=123,
        req_idx=0,
        cur_kv_len=n * pages_num,
        shared_kv_node=None,
        cache_tiers=(CacheTier.CPU, CacheTier.DISK),
        shm_req=NS(
            group_req_id=123,
            request_id=123,
            input_len=n * pages_num + 1,
            token_hash_page_len_list=NS(get_all=lambda: list(range(n, n * (pages_num + 1), n))),
            token_hash_list=NS(get_all=lambda: list(range(pages_num))),
        ),
    )
    m.cache_sessions[123] = CacheSession(123)
    with torch.cuda.stream(cpu):
        torch.cuda._sleep(20_000_000)
    assert m.offload_finished_reqs_to_cpu_cache([req]) == [req]
    assert len(m.store_tasks) <= 2 and len(m.staging_slots) == 2
    # Simulate request token pages immediately being reused on the control stream.
    state.k_storage.fill_(0)
    state.v_storage.fill_(0)
    torch.cuda.synchronize()
    m._poll_async_tasks()
    assert not m.store_tasks and not m.cache_sessions
    assert all(count == 0 for count in m.cpu_cache_client.refs.values())
    assert len(m.cpu_cache_client.disks) == 1
    op = m.cpu_cache_operator
    op.read_req(
        m.cpu_cache_client.cpu_kv_cache_tensor,
        list(range(pages_num)),
        torch.arange(n * pages_num, device="cuda"),
        op.get_page_layouts(),
    )
    torch.cuda.synchronize()
    assert (state.k_storage[:, : n * pages_num] == 42).all()
    assert (state.v_storage[:, : n * pages_num] == 67).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_restore_defers_cpu_release_without_host_sync(monkeypatch, cpu_args):
    n = 256
    cpu_args.cpu_cache_token_page_size = n
    m = module(monkeypatch)
    m.args = NS(cpu_cache_token_page_size=n, page_size=4)
    m.async_cache = True
    m.init_sync_group = object()
    owner = AttentionStateManager(request_capacity=1)
    state = GQAFullAttentionState(
        GQAAttentionConfig(logical_layer_groups=((0,),), global_head_num=1, head_dim=16, dtype=torch.bfloat16),
        page_size=4,
        device="cuda",
    )
    state.allocate_storage(n)
    state.k_storage.fill_(42)
    state.v_storage.fill_(67)
    owner.register_state("main", state)
    owner.finish_state_registration()
    op = owner.get_cpu_cache_operator()
    cpu = torch.empty((1, op.get_page_bytes()), dtype=torch.uint8, pin_memory=True)
    op.write_req(torch.arange(n, device="cuda"), cpu, [0], op.get_page_layouts())
    torch.cuda.synchronize()
    state.k_storage.zero_()
    state.v_storage.zero_()
    m.cpu_cache_client.cpu_kv_cache_tensor = cpu
    m.cpu_cache_client.allocate_pages([10], False)
    req = NS(
        req_id=1,
        req_idx=0,
        cur_kv_len=0,
        hold_kv_len=0,
        mm_input_spans=[],
        sampling_param=NS(shm_param=NS(prompt_logprobs=-1)),
        shm_req=NS(
            input_len=n + 1,
            disk_prompt_cache_len=0,
            cpu_cache_match_page_indexes=NS(get_all=lambda: [0]),
            token_hash_page_len_list=NS(get_all=lambda: [n]),
        ),
    )
    table = torch.zeros((1, n), device="cuda", dtype=torch.int32)

    def alloc(req, size):
        ids = torch.arange(size, dtype=torch.int32)
        table[req.req_idx, :size] = ids.cuda()
        req.hold_kv_len = size
        return ids

    m.backend.model = NS(att_state_manager=owner, req_manager=NS(req_to_token_indexs=table))
    m.backend._alloc_req_kv_mem = alloc
    m.need_sync_compute_stream = lambda: False
    monkeypatch.setattr(impl.g_infer_context, "get_can_alloc_token_num", lambda: n)
    monkeypatch.setattr(impl.dist, "barrier", lambda **kw: None)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda.Stream, "synchronize", lambda self: pytest.fail("per-request host sync"))
        m.load_cpu_cache_to_reqs([req])
    assert req.cur_kv_len == n and m.cpu_cache_client.refs[0] == 1 and len(m.load_tasks) == 1
    torch.cuda.synchronize()
    m._poll_async_tasks()
    assert (state.k_storage[:, :n] == 42).all() and (state.v_storage[:, :n] == 67).all()
    assert m.cpu_cache_client.refs[0] == 0 and not m.load_tasks


@pytest.mark.parametrize("page_tokens,end", [(256, 512), (512, 768)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dsv4_incremental_full_and_tail_checkpoint_lifetimes(monkeypatch, cpu_args, page_tokens, end):
    from unit_tests.common.att_resources.test_dsv4_state import make_atomic_state, runtime_buffer, gather_swa_reference

    cpu_args.cpu_cache_token_page_size = page_tokens
    cpu_args.linear_att_hash_page_size = 256
    cpu_args.linear_att_page_block_num = page_tokens // 256
    owner = AttentionStateManager(3)
    full = [make_full(ratio) for ratio in (4, 128)]
    bounded = [make_atomic_state(kind, mtp_step=3, device="cuda") for kind in ("swa", "kv_w8", "indexer_w8", "kv_w128")]
    for s in (*full, *bounded):
        owner.register_state("main", s)
    owner.finish_state_registration()
    owner.allocate_storage(2048, 256, page_tokens, small_page_capacity=2, device="cuda")
    for s in full:
        s.storage.random_(0, 256)
    for s in bounded:
        runtime_buffer(s).random_(0, 127)
    big = owner.bounded_att_big_page_allocator.alloc(1)[0]
    owner.save_bounded_att_runtime_to_pages([0], [page_tokens], "big", [big])
    m = module(monkeypatch)
    m.backend.is_hybrid_att_model = True
    m.args = NS(
        cpu_cache_token_page_size=page_tokens,
        linear_att_hash_page_size=256,
        disable_linear_att_small_page_cpu_cache=False,
        enable_disk_cache=False,
    )
    m.incremental_offload = True
    m.async_cache = True
    m.gloo_group = object()
    m.backend.model = NS(
        att_state_manager=owner, req_manager=NS(req_to_token_indexs=torch.arange(end, device="cuda").unsqueeze(0))
    )
    m.cpu_cache_operator = owner.get_cpu_cache_operator()
    size = m.cpu_cache_operator.get_page_bytes()
    m.cpu_cache_client.cpu_kv_cache_tensor = torch.empty((64, size), dtype=torch.uint8, pin_memory=True)
    m.staging_slots = [StagingSlot(torch.empty((4, size), dtype=torch.uint8, device="cuda")) for _ in range(2)]
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    cpu = torch.cuda.Stream()
    monkeypatch.setattr(impl.g_infer_context, "get_overlap_stream", lambda: producer)
    monkeypatch.setattr(impl.g_infer_context, "get_cpu_kv_cache_stream", lambda: cpu)
    monkeypatch.setattr(impl.g_infer_context, "is_hybrid_att_model", True)
    req = NS(
        req_id=1,
        req_idx=0,
        cur_kv_len=page_tokens,
        shared_kv_node=None,
        tail_small_page_buffer_id=None,
        hybrid_len_to_big_page_id={page_tokens: big},
        cache_tiers=(CacheTier.GPU,),
        shm_req=NS(
            group_req_id=1,
            request_id=1,
            input_len=end + 1,
            token_hash_page_len_list=NS(get_all=lambda: [page_tokens, end]),
            token_hash_list=NS(get_all=lambda: [10, 20]),
        ),
    )
    m.cache_sessions[1] = CacheSession(1)
    m.store_completed_prefill_pages([req], producer)
    assert m.cache_sessions[1].page_count == 1
    if end % page_tokens:
        small = owner.bounded_att_small_page_allocator.alloc(1)[0]
        req.tail_small_page_buffer_id = small
        with torch.cuda.stream(producer):
            owner.save_bounded_att_runtime_to_pages([0], [end], "small", [small])
    else:
        second = owner.bounded_att_big_page_allocator.alloc(1)[0]
        req.hybrid_len_to_big_page_id[end] = second
        with torch.cuda.stream(producer):
            owner.save_bounded_att_runtime_to_pages([0], [end], "big", [second])
    req.cur_kv_len = end
    m.store_completed_prefill_pages([req], producer)
    m.close_cache_sessions([req])
    # Request can release the original pinned checkpoints while offload references remain.
    owner.bounded_att_big_page_allocator.free(list(req.hybrid_len_to_big_page_id.values()))
    if req.tail_small_page_buffer_id is not None:
        owner.bounded_att_small_page_allocator.free([req.tail_small_page_buffer_id])
    torch.cuda.synchronize()
    expected_swa = gather_swa_reference(bounded[0], 0, end)
    expected = [s.storage[:, :end].clone() for s in full]
    m._poll_async_tasks()
    assert not m.cache_sessions and m.cpu_cache_client.ready == {0, 1}
    slots = owner.bounded_att_big_page_allocator.alloc(end // page_tokens)
    op = m.cpu_cache_operator
    op.read_req(
        m.cpu_cache_client.cpu_kv_cache_tensor,
        [0, 1],
        torch.arange(1024, 1024 + end, device="cuda"),
        op.get_page_layouts(),
        big_page_slot_ids=slots,
        req_idx=1,
        state_token_len=end,
    )
    torch.cuda.synchronize()
    for s, b in zip(full, expected):
        assert torch.equal(s.storage[:, 1024 : 1024 + end], b)
    assert torch.equal(expected_swa, gather_swa_reference(bounded[0], 1, end))
