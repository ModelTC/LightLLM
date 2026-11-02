from types import SimpleNamespace

import pytest

from lightllm.server.router.req_queue.chunked_prefill.impl import ChunkedPrefillQueue


def queue(reserved, current=(), enabled=True):
    result = ChunkedPrefillQueue.__new__(ChunkedPrefillQueue)
    result.dp_index = 0
    result.max_total_tokens = 100
    result.running_max_req_size = 10
    result.cache_len_list = list(current)
    result.tool_call_admission_enabled = enabled
    result.router = SimpleNamespace(
        router_statics=SimpleNamespace(ema_req_out_len=0),
        get_tool_call_reserved_tokens=lambda _: reserved,
        shared_token_load=SimpleNamespace(
            set_estimated_peak_token_count=lambda *args: None,
            set_dynamic_max_load=lambda *args: None,
        ),
    )
    return result


def req(tokens):
    return SimpleNamespace(get_tuple_tokens=lambda *args: (tokens, 0))


def test_waiting_cache_capacity_is_not_spent_on_unrelated_active_requests():
    assert queue(0, [(30, 0)])._can_add_new_req(req(40), False)
    assert not queue(40, [(30, 0)])._can_add_new_req(req(40), False)


@pytest.mark.parametrize("reserved", [0, 40, 100])
def test_first_feasible_request_can_borrow_waiting_space(reserved):
    assert queue(reserved)._can_add_new_req(req(80), False)
    assert not queue(reserved)._can_add_new_req(req(100), False)


def test_consumption_or_expiry_immediately_reopens_admission():
    result = queue(40, [(30, 0)])
    assert not result._can_add_new_req(req(40), False)
    result.cache_len_list = [(30, 0)]
    result.router.get_tool_call_reserved_tokens = lambda _: 0
    assert result._can_add_new_req(req(40), False)


def test_disabled_policy_does_not_read_reservation_or_change_capacity():
    result = queue(100, [(30, 0)], enabled=False)
    result.router.get_tool_call_reserved_tokens = lambda _: pytest.fail("disabled reservation read")
    assert result._can_add_new_req(req(40), False)


def test_explicit_high_priority_can_borrow_without_waiting_for_cache_expiry():
    result = queue(40, [(30, 0)])
    urgent = req(40)
    urgent.sample_params = SimpleNamespace(high_priority_request=True)
    assert result._can_add_new_req(urgent, False)


@pytest.mark.parametrize("mode,enabled", [("normal", True), ("prefill", False), ("decode", False)])
def test_reservation_admission_is_disabled_for_pd_roles(monkeypatch, mode, enabled):
    from lightllm.server.router.req_queue import base_queue

    monkeypatch.setattr(base_queue, "get_fixed_kv_len", lambda: 0)
    monkeypatch.setattr(base_queue, "get_running_max_req_size_per_dp", lambda _: 10)
    args = SimpleNamespace(
        max_total_token_num=100, router_token_ratio=0.8, tool_call_cache_retention_seconds=3, run_mode=mode
    )
    result = base_queue.BaseQueue(args, SimpleNamespace(), 0, 1)
    assert result.tool_call_admission_enabled is enabled


def test_router_accounts_for_idle_protected_tokens_once_and_keeps_disabled_accounting():
    from lightllm.server.router.manager import RouterManager

    router = RouterManager.__new__(RouterManager)
    router.args = SimpleNamespace(disable_dynamic_prompt_cache=False)
    router.max_total_token_num = 100
    router.read_only_statics_mem_manager = SimpleNamespace(get_unrefed_token_num=lambda _: 50)
    router.radix_cache_client = SimpleNamespace(
        get_unrefed_tokens_num=lambda _: 30, get_tool_call_reserved_tokens=lambda _: 20
    )
    router.tool_call_admission_enabled = True
    assert router.get_used_tokens(0) == 40
    router.tool_call_admission_enabled = False
    assert router.get_used_tokens(0) == 20
    assert router.get_tool_call_reserved_tokens(0) == 0


def test_actual_continuation_can_borrow_but_never_exceeds_full_capacity():
    continuation = req(40)
    continuation._tool_call_cache_hit = True
    assert queue(40, [(30, 0)])._can_add_new_req(continuation, False)
    assert not queue(40, [(70, 0)])._can_add_new_req(continuation, False)


def waiting_req(name, tokens, urgent=False):
    result = req(tokens)
    result.request_id = name
    result.sample_params = SimpleNamespace(high_priority_request=urgent, suggested_dp_index=0)
    result.shm_prompt_ids = True
    result.get_prompt_ids_numpy = lambda: [tokens, 1]
    return result


def priority_queue(waiting):
    result = queue(40)
    result.waiting_req_list = waiting
    result.router.get_tool_call_prefix_hints = lambda _: SimpleNamespace(
        by_length={1}, matches=lambda ids, memo: ids[0] == 40
    )
    return result


def scheduling_queue(waiting):
    result = priority_queue(waiting)
    result.dp_size_in_node = 1
    result.filter_aborted_reqs = lambda: None
    result.is_busy = lambda: False
    return result


def active(tokens):
    return SimpleNamespace(reqs=[waiting_req("active", tokens)])


def test_hint_refresh_does_not_reorder_any_waiter_or_override_explicit_priority():
    ordinary, a, b = waiting_req("ordinary", 20), waiting_req("a", 40), waiting_req("b", 40)
    urgent = waiting_req("urgent", 10, urgent=True)
    result = priority_queue([urgent, ordinary, a, b])
    result._refresh_tool_continuation_hints()
    assert result.waiting_req_list == [urgent, ordinary, a, b]
    assert a._tool_call_cache_hit and b._tool_call_cache_hit
    assert not ordinary._tool_call_cache_hit and not urgent._tool_call_cache_hit


def test_feasible_fifo_request_runs_before_a_later_matching_continuation():
    ordinary, continuation = waiting_req("ordinary", 20), waiting_req("agent", 40)
    result = scheduling_queue([ordinary, continuation])
    batch = result.generate_new_batch(active(30))
    assert batch.reqs == [ordinary, continuation]
    assert result.waiting_req_list == []
    assert getattr(ordinary, "_tool_call_priority_skips", 0) == 0


def test_first_fifo_head_is_not_preempted_even_when_reservation_is_full():
    ordinary, continuation = waiting_req("ordinary", 80), waiting_req("agent", 40)
    result = scheduling_queue([ordinary, continuation])
    result.router.get_tool_call_reserved_tokens = lambda _: 100
    batch = result.generate_new_batch(None)
    assert batch.reqs == [ordinary]
    assert result.waiting_req_list == [continuation]


def test_blocked_fifo_head_allows_cached_backfill_without_counting_failed_probes():
    ordinary, continuation = waiting_req("ordinary", 50), waiting_req("agent", 40)
    result = scheduling_queue([ordinary, continuation])
    assert result.generate_new_batch(active(90)) is None
    assert result.waiting_req_list == [ordinary, continuation]
    assert result.cache_len_list == [(90, 0)]
    assert getattr(ordinary, "_tool_call_priority_skips", 0) == 0
    batch = result.generate_new_batch(active(30))
    assert batch.reqs == [continuation]
    assert result.waiting_req_list == [ordinary]
    assert ordinary._tool_call_priority_skips == 1
    assert result.cache_len_list == [(30, 0), (40, 0)]


def test_exhausted_backfill_budget_waits_then_runs_old_fifo_head_first():
    ordinary, continuation = waiting_req("ordinary", 50), waiting_req("agent", 40)
    ordinary._tool_call_priority_skips = 8
    result = scheduling_queue([ordinary, continuation])
    assert result.generate_new_batch(active(30)) is None
    assert result.waiting_req_list == [ordinary, continuation]
    batch = result.generate_new_batch(None)
    assert batch.reqs == [ordinary, continuation]
    assert ordinary._tool_call_priority_skips == 8


def test_explicit_urgent_head_is_never_bypassed_by_cached_backfill():
    urgent, continuation = waiting_req("urgent", 80, urgent=True), waiting_req("agent", 40)
    result = scheduling_queue([urgent, continuation])
    assert result.generate_new_batch(active(30)) is None
    assert result.waiting_req_list == [urgent, continuation]


def test_only_matching_continuations_can_fill_space_behind_blocked_head():
    ordinary, shorter = waiting_req("ordinary", 50), waiting_req("shorter", 20)
    result = scheduling_queue([ordinary, shorter])
    assert result.generate_new_batch(active(30)) is None
    assert result.waiting_req_list == [ordinary, shorter]


def test_successful_backfill_limit_is_eight_total_for_the_same_blocked_head():
    ordinary = waiting_req("ordinary", 50)
    ordinary._tool_call_priority_skips = 7
    a, b = waiting_req("a", 40), waiting_req("b", 40)
    result = scheduling_queue([ordinary, a, b])
    result.max_total_tokens = 200
    result.router.get_tool_call_reserved_tokens = lambda _: 150
    batch = result.generate_new_batch(active(30))
    assert batch.reqs == [a]
    assert result.waiting_req_list == [ordinary, b]
    assert ordinary._tool_call_priority_skips == 8


def test_disabled_queue_keeps_original_fifo_and_never_reads_hint_or_reservation():
    ordinary, continuation = waiting_req("ordinary", 50), waiting_req("agent", 40)
    result = scheduling_queue([ordinary, continuation])
    result.tool_call_admission_enabled = False
    result.router.get_tool_call_prefix_hints = lambda _: pytest.fail("disabled table read")
    result.router.get_tool_call_reserved_tokens = lambda _: pytest.fail("disabled reservation read")
    batch = result.generate_new_batch(active(30))
    assert batch.reqs == [ordinary]
    assert result.waiting_req_list == [continuation]


def test_explicit_cache_disable_does_not_probe_a_matching_prompt():
    ordinary, disabled = waiting_req("ordinary", 50), waiting_req("disabled", 40)
    disabled.sample_params.disable_prompt_cache = True
    disabled.get_prompt_ids_numpy = lambda: pytest.fail("disabled cache prompt probe")
    result = scheduling_queue([ordinary, disabled])
    assert result.generate_new_batch(active(30)) is None
    assert result.waiting_req_list == [ordinary, disabled]
    assert not disabled._tool_call_cache_hit


def test_multinode_router_never_uses_local_prefix_hint_ordering():
    from lightllm.server.router.manager import RouterManager

    router = RouterManager.__new__(RouterManager)
    router.tool_call_admission_enabled = True
    router.is_multinode_tp = True
    router.radix_cache_client = SimpleNamespace(get_tool_call_prefix_hints=lambda _: pytest.fail("local hints"))
    assert router.get_tool_call_prefix_hints(0) is None
