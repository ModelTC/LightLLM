from types import SimpleNamespace
from unittest.mock import MagicMock

from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.mode_backend import base_backend
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import (
    decode_impl as pd_decode_impl,
)
from lightllm.server.router.req_queue import _get_req_queue_class
from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd import PDQueue


def _make_infer_req(cur_output_len: int, shm_output_len: int):
    return SimpleNamespace(
        req_id=1,
        filter_mark=False,
        wait_pause=False,
        paused=False,
        infer_aborted=False,
        finish_status=FinishStatus(),
        cpu_cache_task_status=SimpleNamespace(is_not_started=MagicMock(return_value=True)),
        cur_kv_len=10,
        cur_output_len=cur_output_len,
        shm_req=SimpleNamespace(shm_cur_output_len=shm_output_len),
        sampling_param=SimpleNamespace(
            shm_param=SimpleNamespace(max_new_tokens=65535),
        ),
        get_cur_total_len=MagicMock(return_value=11),
        decode_need_token_num=MagicMock(return_value=1),
    )


def _classify_without_token_capacity(monkeypatch, req, support_overlap=True):
    reqs = req if isinstance(req, list) else [req]
    backend = pd_decode_impl.PDDecodeNode.__new__(pd_decode_impl.PDDecodeNode)
    backend.args = SimpleNamespace(
        enable_cpu_cache=False,
        enable_prefill_decode_mixed=False,
        run_mode="decode",
    )
    backend.support_overlap = support_overlap
    backend.is_master_in_dp = True
    logger = MagicMock()
    backend.logger = logger
    backend._timer_merge_radix_tree = MagicMock()
    backend._filter_not_ready_reqs = MagicMock(return_value=reqs)
    backend._reorder_pd_high_priority_reqs = MagicMock(side_effect=lambda reqs: reqs)
    backend._reorder_long_prefill_reqs = MagicMock(side_effect=lambda reqs: reqs)

    infer_context = base_backend.g_infer_context
    monkeypatch.setattr(infer_context, "get_can_alloc_token_num", MagicMock(return_value=0))
    monkeypatch.setattr(
        infer_context,
        "cache_placement_controller",
        SimpleNamespace(set_req_cache_way=MagicMock()),
    )
    filter_reqs = MagicMock()
    monkeypatch.setattr(infer_context, "filter_reqs", filter_reqs)
    monkeypatch.setattr(infer_context, "pause_reqs", MagicMock())

    backend._get_classed_reqs(req_ids=[req.req_id for req in reqs])
    return filter_reqs, logger


def test_pd_decode_capacity_shortage_is_delayed_for_overlap(monkeypatch):
    req = _make_infer_req(cur_output_len=5, shm_output_len=4)

    filter_reqs, logger = _classify_without_token_capacity(monkeypatch, req, support_overlap=True)

    assert req.filter_mark
    assert req.finished_by_pd_decode_capacity
    assert not req.finish_status.is_finished()
    assert req.sampling_param.shm_param.max_new_tokens == 65535
    assert not req.wait_pause
    filter_reqs.assert_called_once_with(finished_reqs=[])
    assert logger.info.call_args.args[0] == (
        "force early finish for PD decode req_id=1 because token capacity is insufficient"
    )

    filter_reqs, _ = _classify_without_token_capacity(monkeypatch, req, support_overlap=True)
    filter_reqs.assert_called_once_with(finished_reqs=[req])


def test_pd_decode_capacity_shortage_is_filtered_without_overlap(monkeypatch):
    req = _make_infer_req(cur_output_len=5, shm_output_len=4)

    filter_reqs, logger = _classify_without_token_capacity(monkeypatch, req, support_overlap=False)

    assert not req.filter_mark
    assert req.finished_by_pd_decode_capacity
    assert not req.finish_status.is_finished()
    assert req.sampling_param.shm_param.max_new_tokens == 65535
    assert not req.wait_pause
    filter_reqs.assert_called_once_with(finished_reqs=[req])
    assert logger.info.call_args.args[0] == (
        "force early finish for PD decode req_id=1 because token capacity is insufficient"
    )


def test_pd_decode_capacity_shortage_only_handles_two_requests_per_iteration(monkeypatch):
    reqs = [_make_infer_req(cur_output_len=5, shm_output_len=4) for _ in range(3)]
    for req_id, req in enumerate(reqs, start=1):
        req.req_id = req_id

    filter_reqs, _ = _classify_without_token_capacity(monkeypatch, reqs, support_overlap=True)

    assert [req.filter_mark for req in reqs] == [True, True, False]
    filter_reqs.assert_called_once_with(finished_reqs=[])


def test_pd_decode_capacity_finish_status_is_written_to_shm():
    shm_req = SimpleNamespace(mark_simulated_finished=MagicMock())
    req = SimpleNamespace(
        finish_status=FinishStatus(),
        finished_by_pd_decode_capacity=True,
        shm_req=shm_req,
        cur_output_len=3,
    )

    pd_decode_impl.InferReq.mark_shm_aborted_finished(req)

    shm_req.mark_simulated_finished.assert_called_once_with(
        FinishStatus.FINISHED_PD_DECODE_CAPACITY,
        output_len=3,
    )


def test_request_finish_status_takes_priority_over_pd_decode_capacity_marker():
    shm_req = SimpleNamespace(mark_simulated_finished=MagicMock())
    req = SimpleNamespace(
        finish_status=FinishStatus(FinishStatus.FINISHED_STOP),
        finished_by_pd_decode_capacity=True,
        shm_req=shm_req,
        cur_output_len=3,
    )

    pd_decode_impl.InferReq.mark_shm_aborted_finished(req)

    shm_req.mark_simulated_finished.assert_not_called()


def test_pd_decode_capacity_limit_never_extends_original_length(monkeypatch):
    req = _make_infer_req(cur_output_len=5, shm_output_len=1)
    req.sampling_param.shm_param.max_new_tokens = 3

    _classify_without_token_capacity(monkeypatch, req)
    assert req.sampling_param.shm_param.max_new_tokens == 3


def test_pd_nodes_use_pd_queue():
    base_args = {
        "diverse_mode": False,
        "token_healing_mode": False,
        "output_constraint_mode": "none",
        "first_token_constraint_mode": False,
        "disable_chunked_prefill": False,
    }

    prefill_args = SimpleNamespace(**base_args, run_mode="prefill")
    decode_args = SimpleNamespace(**base_args, run_mode="decode")

    assert _get_req_queue_class(prefill_args, router=None, dp_size_in_node=1) is PDQueue
    assert _get_req_queue_class(decode_args, router=None, dp_size_in_node=1) is PDQueue


def test_pd_decode_queue_uses_ema_for_prefill_stage_output_length():
    queue = PDQueue.__new__(PDQueue)
    queue.args = SimpleNamespace(run_mode="decode")
    queue.dp_index = 0
    queue.max_total_tokens = 4096
    queue.running_max_req_size = 8
    queue.router = SimpleNamespace(
        router_statics=SimpleNamespace(ema_req_out_len=128),
        shared_token_load=SimpleNamespace(
            set_estimated_peak_token_count=MagicMock(),
            set_dynamic_max_load=MagicMock(),
        ),
    )
    queue.is_busy = MagicMock(return_value=False)

    req = SimpleNamespace(
        input_len=10,
        sample_params=SimpleNamespace(suggested_dp_index=0, max_new_tokens=1024),
        is_infer_decode=MagicMock(return_value=False),
    )
    batch = SimpleNamespace(reqs=[req])

    assert queue._caclu_batch_estimated_peak_token_num(batch) == 138
    assert queue._can_add_new_req(req, estimated_peak_token_num=0, batch_req_num=0) == (True, 138, 1)

    # 接近上下文上限的请求可能只剩很少输出额度，估算值不能超过请求自身的 max_new_tokens，
    # 否则本来能够运行的请求会因为 EMA 偏大而永久滞留在 Decode 等待队列。
    req.sample_params.max_new_tokens = 20
    assert queue._caclu_batch_estimated_peak_token_num(batch) == 30
    assert queue._can_add_new_req(req, estimated_peak_token_num=0, batch_req_num=0) == (True, 30, 1)

    req.sample_params.max_new_tokens = 1024
    queue.args.run_mode = "prefill"
    assert queue._caclu_batch_estimated_peak_token_num(batch) == 1034
    assert queue._can_add_new_req(req, estimated_peak_token_num=0, batch_req_num=0) == (True, 1034, 1)
