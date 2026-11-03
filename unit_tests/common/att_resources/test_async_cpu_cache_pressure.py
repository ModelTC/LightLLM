"""Tail-page, checkpoint pressure and CPU-only cache regressions."""

from types import SimpleNamespace as NS
import pytest
import torch
from unit_tests.common.att_resources.test_async_cpu_cache import module, FakeEvent, Client
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.allocators import StatePageAllocator
from lightllm.common.attention_state.full_att import GQAFullAttentionState, GQAAttentionConfig
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as impl
from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import (
    CacheSession,
    CheckpointTask,
    StoreTask,
    StagingSlot,
)
from lightllm.server.router.model_infer.infer_batch import InferenceContext
from unit_tests.common.att_resources.test_dsv4_state import (
    make_full,
    make_atomic_state,
    runtime_buffer,
    gather_swa_reference,
)
from unit_tests.common.att_resources.model_helpers import full_state_test_buffers


def test_named_allocation_keeps_ready_distinct_from_writer():
    c = Client()
    a = c.allocate_pages_for_write([10, 20], False)
    assert a.page_indices == [0, 1] and a.ready_flags == [False, False] and a.writer_flags == [True, True]
    c.update_pages_status_to_ready([0], deref=False)
    a = c.allocate_pages_for_write([10, 20], False)
    assert a.ready_flags == [True, False] and a.writer_flags == [False, False]
    assert c.allocate_pages([10, 20], False) == ([0, 1], [True, False])


@pytest.mark.parametrize("load", [False, True])
def test_pressure_reclaims_checkpoint_before_actual_snapshot(monkeypatch, load):
    m = module(monkeypatch)
    a = StatePageAllocator(4)
    ids = a.alloc(4)
    a.retain([ids[0]])
    a.free([ids[0]])  # KV/radix owner goes away, asynchronous checkpoint reference remains.
    packed = FakeEvent()
    m.checkpoint_tasks.append(CheckpointTask(packed, [(a, [ids[0]])]))
    a.on_allocation_pressure = m._reclaim_checkpoint_slots
    cpu_done = FakeEvent()
    if not load:
        m.store_tasks.append(StoreTask([], CacheSession(9), StagingSlot(None, True), cpu_done))
    saved = []
    context = InferenceContext()
    context.is_hybrid_att_model = True
    context.args = NS(linear_att_hash_page_size=256, linear_att_page_block_num=1, disable_chunked_prefill=False)
    context.att_state_manager = NS(
        bounded_att_big_page_allocator=a, save_bounded_att_runtime_to_pages=lambda **kw: saved.append(kw)
    )
    req = NS(
        req_idx=0,
        cur_kv_len=0,
        hybrid_cache_len=512,
        hybrid_len_to_big_page_id={},
        get_chuncked_input_token_len=lambda: 256,
    )
    context.save_hybrid_state_to_cache(torch.tensor([0]), [req])
    assert req.hybrid_len_to_big_page_id == {256: ids[0]} and packed.ready
    assert saved[0]["page_slot_ids"] == [ids[0]]
    assert not cpu_done.ready, "allocation must not wait for staging -> CPU completion"
    assert not m.checkpoint_tasks
    assert a.alloc(1) is None  # no invented spare capacity


def test_reclaim_waits_only_required_fifo_prefix(monkeypatch):
    m = module(monkeypatch)
    a = StatePageAllocator(2)
    ids = a.alloc(2)
    events = [FakeEvent(), FakeEvent()]
    for slot, event in zip(ids, events):
        a.retain([slot])
        a.free([slot])
        m.checkpoint_tasks.append(CheckpointTask(event, [(a, [slot])]))
    a.on_allocation_pressure = m._reclaim_checkpoint_slots
    assert a.alloc(1) == [ids[0]]
    assert events[0].ready and not events[1].ready and len(m.checkpoint_tasks) == 1


@pytest.mark.parametrize("layout", ["gqa", "packed"])
@pytest.mark.parametrize("ends,skip", [([256], 0), ([2048, 2304], 0), ([2048, 4096, 4352], 2)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_store_tail_never_passes_negative_slots(monkeypatch, cpu_args, layout, ends, skip):
    page_tokens = 2048
    cpu_args.cpu_cache_token_page_size = page_tokens
    cpu_args.linear_att_hash_page_size = 256
    cpu_args.linear_att_page_block_num = 8
    owner = AttentionStateManager(3)
    if layout == "packed":
        full = [make_full(ratio) for ratio in (4, 128)]
    else:
        full = [
            GQAFullAttentionState(
                GQAAttentionConfig(logical_layer_groups=((0,),), global_head_num=1, head_dim=16, dtype=torch.bfloat16),
                page_size=256,
                device="cuda",
            )
        ]
    swa = make_atomic_state("swa", mtp_step=3, device="cuda")
    for state in (*full, swa):
        owner.register_state("main", state)
    owner.finish_state_registration()
    owner.allocate_storage(16384, 256, page_tokens, small_page_capacity=4, device="cuda")
    for state in full:
        for buffer in ({"packed": state.storage} if layout == "packed" else full_state_test_buffers(state)).values():
            buffer.fill_(42)
    runtime_buffer(swa).random_(0, 127)
    big_slots = owner.bounded_att_big_page_allocator.alloc(len(ends) - 1)
    for end, slot in zip(ends[:-1], big_slots):
        owner.save_bounded_att_runtime_to_pages([0], [end], "big", [slot])
    small = owner.bounded_att_small_page_allocator.alloc(1)[0]
    owner.save_bounded_att_runtime_to_pages([0], [ends[-1]], "small", [small])
    m = module(monkeypatch)
    m.backend.is_hybrid_att_model = True
    m.args = NS(
        cpu_cache_token_page_size=page_tokens,
        linear_att_hash_page_size=256,
        disable_linear_att_small_page_cpu_cache=False,
        enable_disk_cache=False,
    )
    m.async_cache = True
    m.incremental_offload = True
    m.gloo_group = object()
    table = torch.arange(ends[-1], device="cuda", dtype=torch.int32).unsqueeze(0)
    m.backend.model = NS(att_state_manager=owner, req_manager=NS(req_to_token_indexs=table))
    m.cpu_cache_operator = owner.get_cpu_cache_operator()
    size = m.cpu_cache_operator.get_page_bytes()
    m.cpu_cache_client.cpu_kv_cache_tensor = torch.empty((64, size), dtype=torch.uint8, pin_memory=True)
    m.staging_slots = [StagingSlot(torch.empty((4, size), device="cuda", dtype=torch.uint8)) for _ in range(2)]
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    cpu = torch.cuda.Stream()
    monkeypatch.setattr(impl.g_infer_context, "get_cpu_kv_cache_stream", lambda: cpu)
    monkeypatch.setattr(impl.g_infer_context, "is_hybrid_att_model", True)
    hashes = list(range(10, 10 + len(ends)))
    if skip:
        pages, _ = m.cpu_cache_client.allocate_pages(hashes[:skip], False)
        m.cpu_cache_client.update_pages_status_to_ready(pages, deref=True)
    captured = []
    write = m.cpu_cache_operator.write_req

    def record(**kw):
        captured.append(kw["token_indices"].clone())
        return write(**kw)

    monkeypatch.setattr(m.cpu_cache_operator, "write_req", record)
    req = NS(
        req_id=1,
        req_idx=0,
        cur_kv_len=ends[-1],
        shared_kv_node=None,
        tail_small_page_buffer_id=small,
        hybrid_len_to_big_page_id=dict(zip(ends[:-1], big_slots)),
        cache_tiers=(CacheTier.GPU,),
        shm_req=NS(
            group_req_id=1,
            request_id=1,
            input_len=ends[-1] + 1,
            token_hash_page_len_list=NS(get_all=lambda: ends),
            token_hash_list=NS(get_all=lambda: hashes),
        ),
    )
    m.cache_sessions[1] = CacheSession(1)
    m.store_completed_prefill_pages([req], producer)
    torch.cuda.synchronize()
    m._poll_async_tasks()
    expected_start = skip * page_tokens
    assert torch.equal(torch.cat(captured), table[0, expected_start : ends[-1]])
    assert (torch.cat(captured) >= 0).all()
    assert all(ref == 0 for ref in m.cpu_cache_client.refs.values()), "CPU-only pages must be evictable during decode"
    assert m.cache_sessions[1].page_count == len(ends) and not m.cache_sessions[1].disk_pages
    selected_pages = [m.cpu_cache_client.hashes[h] for h in hashes[skip:]]
    new_ids = torch.arange(8192, 8192 + ends[-1] - expected_start, device="cuda", dtype=torch.int32)
    state_expected = [
        {
            name: value[:, expected_start : ends[-1]].clone()
            for name, value in ({"packed": s.storage} if layout == "packed" else full_state_test_buffers(s)).items()
        }
        for s in full
    ]
    expected_swa = gather_swa_reference(swa, 0, ends[-1])
    restored_big = owner.bounded_att_big_page_allocator.alloc(len(selected_pages) - 1)
    m.cpu_cache_operator.read_req(
        m.cpu_cache_client.cpu_kv_cache_tensor,
        selected_pages,
        new_ids,
        m.cpu_cache_operator.get_page_layouts(),
        big_page_slot_ids=restored_big,
        req_idx=1,
        state_token_len=ends[-1],
    )
    torch.cuda.synchronize()
    for state, expected in zip(full, state_expected):
        for name, value in (
            {"packed": state.storage} if layout == "packed" else full_state_test_buffers(state)
        ).items():
            assert torch.equal(value[:, 8192 : 8192 + new_ids.numel()], expected[name])
    assert torch.equal(gather_swa_reference(swa, 1, ends[-1]), expected_swa)


def test_poll_empty_and_batch_allocation_collectives(monkeypatch):
    m = module(monkeypatch)
    calls = []
    monkeypatch.setattr(impl.dist, "broadcast_object_list", lambda *a, **kw: calls.append(a))
    m._poll_async_tasks()
    assert not calls
    m.cache_sessions[1] = CacheSession(1, closing=True)
    m._poll_async_tasks()
    assert len(calls) == 1 and not m.cache_sessions


def test_staging_budget_uses_actual_main_and_draft_capabilities():
    owner = AttentionStateManager(1)
    op = owner.get_cpu_cache_operator()
    # Layout size is independent from the capability selection.
    op.get_page_bytes = lambda: 128
    op.configure_backends([NS(requires_cpu_cache_sync=False), None])
    assert op.async_cache and op.get_staging_bytes() == 8 * 128
    op.configure_backends([NS(requires_cpu_cache_sync=False), NS(requires_cpu_cache_sync=True)])
    assert not op.async_cache and op.get_staging_bytes() == 0


def test_fa3_capability_is_inherited_by_all_variants():
    from lightllm.common.basemodel.attention.base_att import BaseAttBackend
    from lightllm.common.basemodel.attention.fa3.fp import Fa3AttBackend
    from lightllm.common.basemodel.attention.fa3.fp8 import Fp8Fa3AttBackend
    from lightllm.common.basemodel.attention.fa3.mla import MlaFa3AttBackend
    from lightllm.common.basemodel.attention.fa3.neo import NeoFa3AttBackend

    assert not BaseAttBackend.requires_cpu_cache_sync
    assert all(c.requires_cpu_cache_sync for c in [Fa3AttBackend, Fp8Fa3AttBackend, MlaFa3AttBackend, NeoFa3AttBackend])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multiple_requests_allocate_once_and_do_not_pin_cpu_prefixes(monkeypatch, cpu_args):
    cpu_args.cpu_cache_token_page_size = 8
    owner = AttentionStateManager(2)
    state = GQAFullAttentionState(
        GQAAttentionConfig(logical_layer_groups=((0,),), global_head_num=1, head_dim=16, dtype=torch.bfloat16),
        page_size=4,
        device="cuda",
    )
    state.allocate_storage(16)
    state.k_storage.fill_(42)
    state.v_storage.fill_(67)
    owner.register_state("main", state)
    owner.finish_state_registration()
    m = module(monkeypatch)
    m.args = NS(cpu_cache_token_page_size=8, enable_disk_cache=False)
    m.async_cache = True
    m.incremental_offload = True
    m.gloo_group = object()
    m.backend.model = NS(
        att_state_manager=owner, req_manager=NS(req_to_token_indexs=torch.arange(16, device="cuda").repeat(2, 1))
    )
    m.cpu_cache_operator = owner.get_cpu_cache_operator()
    size = m.cpu_cache_operator.get_page_bytes()
    m.cpu_cache_client.cpu_kv_cache_tensor = torch.empty((64, size), dtype=torch.uint8, pin_memory=True)
    m.staging_slots = [StagingSlot(torch.empty((4, size), device="cuda", dtype=torch.uint8)) for _ in range(2)]
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    cpu = torch.cuda.Stream()
    monkeypatch.setattr(impl.g_infer_context, "get_cpu_kv_cache_stream", lambda: cpu)
    monkeypatch.setattr(impl.g_infer_context, "is_hybrid_att_model", False)
    reqs = []
    for rid, hashes in [(1, [10, 20]), (2, [10, 30])]:
        req = NS(
            req_id=rid,
            req_idx=rid - 1,
            cur_kv_len=16,
            shared_kv_node=None,
            cache_tiers=(CacheTier.GPU,),
            shm_req=NS(
                group_req_id=rid,
                request_id=rid,
                input_len=17,
                token_hash_page_len_list=NS(get_all=lambda: [8, 16]),
                token_hash_list=NS(get_all=lambda hashes=hashes: hashes),
            ),
        )
        reqs.append(req)
        m.cache_sessions[rid] = CacheSession(rid)
    calls = []
    monkeypatch.setattr(impl.dist, "broadcast_object_list", lambda *a, **kw: calls.append(a))
    m.store_completed_prefill_pages(reqs, producer)
    assert len(calls) == 1 and len(m.store_tasks) == 2
    assert m.cpu_cache_client.refs == {0: 1, 1: 1, 2: 1}, "shared LOADING ref should be released immediately"
    torch.cuda.synchronize()
    m._poll_async_tasks()
    assert len(calls) == 2 and all(v == 0 for v in m.cpu_cache_client.refs.values())
    assert len(m.cache_sessions) == 2, "requests may keep decoding without pinning CPU pages"
    m.store_completed_prefill_pages(reqs, producer)
    m._poll_async_tasks()
    assert len(calls) == 2, "idle replicated queues/cursors must skip collectives"
    m.close_cache_sessions(reqs)
    m._poll_async_tasks()
    assert not m.cache_sessions and len(calls) == 3
