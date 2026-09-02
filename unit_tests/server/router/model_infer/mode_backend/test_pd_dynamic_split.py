from types import SimpleNamespace
from unittest.mock import MagicMock

from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import (
    decode_impl as pd_decode_impl,
)
from lightllm.server.router.req_queue import _get_req_queue_class
from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd import (
    PDDQueue,
    PDPQueue,
)


def _make_infer_req(cur_output_len: int, shm_output_len: int):
    return SimpleNamespace(
        req_id=1,
        cur_output_len=cur_output_len,
        shm_req=SimpleNamespace(shm_cur_output_len=shm_output_len),
        sampling_param=SimpleNamespace(
            shm_param=SimpleNamespace(max_new_tokens=65535),
        ),
        wait_pause=False,
    )


def test_pd_decode_capacity_yield_only_limits_running_overlap_output():
    backend = pd_decode_impl.PDDecodeNode.__new__(pd_decode_impl.PDDecodeNode)
    backend.support_overlap = True

    running_req = _make_infer_req(cur_output_len=5, shm_output_len=4)
    not_running_req = _make_infer_req(cur_output_len=4, shm_output_len=4)

    assert backend._handle_decode_alloc_failure(running_req) == 1
    assert running_req.sampling_param.shm_param.max_new_tokens == 5
    assert not running_req.wait_pause

    assert backend._handle_decode_alloc_failure(not_running_req) == 0
    assert not_running_req.sampling_param.shm_param.max_new_tokens == 65535
    assert not not_running_req.wait_pause


def test_pd_decode_capacity_limit_finishes_at_committed_mtp_tail():
    backend = pd_decode_impl.PDDecodeNode.__new__(pd_decode_impl.PDDecodeNode)
    backend.support_overlap = True
    req = _make_infer_req(cur_output_len=3, shm_output_len=1)
    req.stop_sequences = []
    req.shm_req.input_len = 10
    req.shm_req.shm_prompt_ids = SimpleNamespace(arr=[0] * 13)
    req.sampling_param.shm_param.ignore_eos = False
    req.finish_status = FinishStatus()
    req._stop_sequences_matched = MagicMock(return_value=False)

    assert backend._handle_decode_alloc_failure(req) == 1
    assert req.sampling_param.shm_param.max_new_tokens == 3

    pd_decode_impl.InferReq.update_finish_status(req, eos_ids=[], output_len=2)
    assert not req.finish_status.is_finished()

    pd_decode_impl.InferReq.update_finish_status(req, eos_ids=[], output_len=3)
    assert req.finish_status.is_finished_length()


def test_pd_decode_capacity_limit_never_extends_original_length():
    backend = pd_decode_impl.PDDecodeNode.__new__(pd_decode_impl.PDDecodeNode)
    backend.support_overlap = True
    req = SimpleNamespace(
        req_id=1,
        cur_output_len=5,
        sampling_param=SimpleNamespace(
            shm_param=SimpleNamespace(max_new_tokens=3),
        ),
        shm_req=SimpleNamespace(shm_cur_output_len=1),
        finish_status=FinishStatus(),
    )

    assert backend._handle_decode_alloc_failure(req) == 1
    assert req.sampling_param.shm_param.max_new_tokens == 3


def test_pd_queue_admission_always_uses_aggressive_tuple_estimate():
    queue = PDDQueue.__new__(PDDQueue)
    queue.dp_index = 0
    queue.max_total_tokens = 100
    queue.running_max_req_size = 8
    queue.batch_max_tokens = 1
    queue.cache_len_list = [(20, 10)]
    queue.router = SimpleNamespace(
        router_statics=SimpleNamespace(ema_req_out_len=128),
        shared_token_load=SimpleNamespace(
            set_estimated_peak_token_count=MagicMock(),
            set_dynamic_max_load=MagicMock(),
        ),
    )

    req = SimpleNamespace(
        get_tuple_tokens=MagicMock(return_value=(20, 10)),
        get_first_router_need_tokens=MagicMock(return_value=1024),
    )
    admitted, first_router_tokens = queue._can_add_new_req(req, queue.is_busy(), 0)

    assert admitted
    assert first_router_tokens == 1024
    req.get_tuple_tokens.assert_called_once_with(False, 128)
    queue.router.shared_token_load.set_estimated_peak_token_count.assert_called_once_with(
        60, 0
    )


def test_aggressive_pd_queue_is_only_selected_for_decode_nodes():
    base_args = {
        "diverse_mode": False,
        "token_healing_mode": False,
        "output_constraint_mode": "none",
        "first_token_constraint_mode": False,
        "disable_chunked_prefill": False,
    }

    prefill_args = SimpleNamespace(**base_args, run_mode="prefill")
    decode_args = SimpleNamespace(**base_args, run_mode="decode")

    assert (
        _get_req_queue_class(prefill_args, router=None, dp_size_in_node=1) is PDPQueue
    )
    assert _get_req_queue_class(decode_args, router=None, dp_size_in_node=1) is PDDQueue
