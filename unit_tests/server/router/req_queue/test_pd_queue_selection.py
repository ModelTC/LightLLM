from types import SimpleNamespace
from unittest.mock import MagicMock

from lightllm.server.router.req_queue import _get_req_queue_class
from lightllm.server.router.batch import Batch
from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd_decode import PDDecodeQueue
from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd_prefill import PDPrefillQueue


def _make_args(run_mode: str):
    return SimpleNamespace(
        run_mode=run_mode,
        diverse_mode=False,
        output_constraint_mode="none",
        first_token_constraint_mode=False,
        disable_chunked_prefill=False,
    )


def test_pd_prefill_uses_prefill_queue():
    args = _make_args("prefill")

    assert _get_req_queue_class(args, router=None, dp_size_in_node=1) is PDPrefillQueue


def test_pd_decode_uses_decode_queue():
    args = _make_args("decode")

    assert _get_req_queue_class(args, router=None, dp_size_in_node=1) is PDDecodeQueue


def test_pd_prefill_peak_tokens_sum_request_tuples():
    queue = PDPrefillQueue.__new__(PDPrefillQueue)
    queue.dp_index = 0
    queue.args = SimpleNamespace(page_size=16)
    queue.is_busy = lambda: False
    queue.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=10))
    reqs = [
        SimpleNamespace(
            request_id="req-0",
            input_len=10,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=0),
            get_tuple_tokens=MagicMock(return_value=(11, 34)),
        ),
        SimpleNamespace(
            request_id="req-1",
            input_len=20,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=0),
            get_tuple_tokens=MagicMock(return_value=(21, 34)),
        ),
        SimpleNamespace(
            request_id="req-2",
            input_len=100,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=1),
            get_tuple_tokens=MagicMock(return_value=(101, 34)),
        ),
    ]
    batch = Batch(batch_id=1, reqs=reqs, dp_size_in_node=2)

    assert queue._caclu_batch_estimated_peak_token_num(batch) == 100
    reqs[0].get_tuple_tokens.assert_called_once_with(False, 10)
    reqs[1].get_tuple_tokens.assert_called_once_with(False, 10)
    reqs[2].get_tuple_tokens.assert_not_called()


def test_pd_decode_sums_request_tuples_for_non_decode_requests():
    queue = PDDecodeQueue.__new__(PDDecodeQueue)
    queue.args = SimpleNamespace(page_size=16)
    queue.dp_index = 0
    queue.is_busy = lambda: False
    queue.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=10))
    reqs = [
        SimpleNamespace(
            request_id="req-0",
            input_len=20,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=0),
            is_infer_decode=lambda: False,
            get_tuple_tokens=MagicMock(return_value=(21, 34)),
        ),
        SimpleNamespace(
            request_id="req-1",
            input_len=100,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=1),
            is_infer_decode=lambda: False,
            get_tuple_tokens=MagicMock(return_value=(101, 34)),
        ),
    ]
    batch = Batch(batch_id=1, reqs=reqs, dp_size_in_node=2)

    assert queue._caclu_batch_estimated_peak_token_num(batch) == 55
    reqs[0].get_tuple_tokens.assert_called_once_with(False, 10)
    reqs[1].get_tuple_tokens.assert_not_called()


def test_pd_decode_uses_tuple_estimation():
    queue = PDDecodeQueue.__new__(PDDecodeQueue)
    queue.args = SimpleNamespace(page_size=16)
    queue.dp_index = 0
    queue.is_busy = lambda: False
    queue.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=10))
    req = SimpleNamespace(
        request_id="req-0",
        sample_params=SimpleNamespace(suggested_dp_index=0),
        is_infer_decode=lambda: True,
        get_tuple_tokens=lambda is_busy, ema_req_out_len: (16, 32),
    )
    batch = Batch(batch_id=1, reqs=[req], dp_size_in_node=1)

    assert queue._caclu_batch_estimated_peak_token_num(batch) == 48
