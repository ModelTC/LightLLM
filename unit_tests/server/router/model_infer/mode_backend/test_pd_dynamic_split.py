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
        cur_kv_len=10,
        cur_output_len=cur_output_len,
        shm_req=SimpleNamespace(shm_cur_output_len=shm_output_len),
        sampling_param=SimpleNamespace(
            shm_param=SimpleNamespace(max_new_tokens=65535),
        ),
        get_cur_total_len=MagicMock(return_value=11),
        decode_need_token_num=MagicMock(return_value=1),
    )


def _classify_without_token_capacity(monkeypatch, req):
    backend = pd_decode_impl.PDDecodeNode.__new__(pd_decode_impl.PDDecodeNode)
    backend.args = SimpleNamespace(
        enable_cpu_cache=False,
        enable_prefill_decode_mixed=False,
        run_mode="decode",
    )
    backend.support_overlap = True
    backend.is_master_in_dp = True
    backend.logger = MagicMock()
    backend._timer_merge_radix_tree = MagicMock()
    backend._filter_not_ready_reqs = MagicMock(return_value=[req])
    backend._reorder_pd_high_priority_reqs = MagicMock(side_effect=lambda reqs: reqs)
    backend._reorder_long_prefill_reqs = MagicMock(side_effect=lambda reqs: reqs)

    infer_context = base_backend.g_infer_context
    monkeypatch.setattr(infer_context, "get_can_alloc_token_num", MagicMock(return_value=0))
    monkeypatch.setattr(
        infer_context,
        "cache_placement_controller",
        SimpleNamespace(set_req_cache_way=MagicMock()),
    )
    monkeypatch.setattr(infer_context, "filter_reqs", MagicMock())
    monkeypatch.setattr(infer_context, "pause_reqs", MagicMock())

    backend._get_classed_reqs(req_ids=[req.req_id])


def test_pd_decode_capacity_yield_only_limits_running_overlap_output(monkeypatch):

    running_req = _make_infer_req(cur_output_len=5, shm_output_len=4)
    not_running_req = _make_infer_req(cur_output_len=4, shm_output_len=4)

    _classify_without_token_capacity(monkeypatch, running_req)
    assert running_req.sampling_param.shm_param.max_new_tokens == 5
    assert not running_req.wait_pause

    _classify_without_token_capacity(monkeypatch, not_running_req)
    assert not_running_req.sampling_param.shm_param.max_new_tokens == 65535
    assert not not_running_req.wait_pause


def test_pd_decode_capacity_limit_finishes_at_committed_mtp_tail(monkeypatch):
    req = _make_infer_req(cur_output_len=3, shm_output_len=1)
    req.stop_sequences = []
    req.shm_req.input_len = 10
    req.shm_req.shm_prompt_ids = SimpleNamespace(arr=[0] * 13)
    req.sampling_param.shm_param.ignore_eos = False
    req.finish_status = FinishStatus()
    req._stop_sequences_matched = MagicMock(return_value=False)

    _classify_without_token_capacity(monkeypatch, req)
    assert req.sampling_param.shm_param.max_new_tokens == 3

    pd_decode_impl.InferReq.update_finish_status(req, eos_ids=[], output_len=2)
    assert not req.finish_status.is_finished()

    pd_decode_impl.InferReq.update_finish_status(req, eos_ids=[], output_len=3)
    assert req.finish_status.is_finished_length()


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
    queue.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=128))
    queue.is_busy = MagicMock(return_value=False)

    req = SimpleNamespace(
        input_len=10,
        sample_params=SimpleNamespace(suggested_dp_index=0, max_new_tokens=1024),
        is_infer_decode=MagicMock(return_value=False),
    )
    batch = SimpleNamespace(reqs=[req])

    assert queue._caclu_batch_estimated_peak_token_num(batch) == 138

    queue.args.run_mode = "prefill"
    assert queue._caclu_batch_estimated_peak_token_num(batch) == 1034
