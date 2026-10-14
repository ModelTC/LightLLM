from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xgrammar as xgr

from unit_tests.server.grammar_helpers import init_request, make_req
from lightllm.server.router.model_infer.infer_batch import InferReq, InferReqUpdatePack


@pytest.mark.parametrize("failure", ["mask", "commit", "rollback"])
@pytest.mark.parametrize("exception_type", [RuntimeError, BaseException])
def test_grammar_failure_finishes_after_publishing_inflight_token(compiler, monkeypatch, failure, exception_type):
    req = make_req(regular_constraint="abc")
    init_request(compiler, req)
    shm_req = req.shm_req
    shm_req.shm_prompt_ids.arr = np.array([ord("P")] + [0] * 8, dtype=np.int64)
    shm_req.shm_logprobs = SimpleNamespace(arr=np.zeros(9, dtype=[("logprob", np.float32), ("rank", np.int32)]))
    req.set_next_gen_token_id = MethodType(InferReq.set_next_gen_token_id, req)
    req.update_finish_status = lambda **kwargs: None
    InferReqUpdatePack(req, 1).handle(ord("a"), -0.1, -1, [256], is_master_in_dp=True)

    if failure == "mask":
        monkeypatch.setattr(
            xgr.GrammarMatcher,
            "fill_next_token_bitmask",
            MagicMock(side_effect=exception_type("injected mask failure")),
        )
        assert not req.output_constraint.fill_masks(torch.empty((1, 9), dtype=torch.int32), [])
    elif failure == "commit":
        monkeypatch.setattr(
            xgr.GrammarMatcher, "accept_token", MagicMock(side_effect=exception_type("injected commit failure"))
        )
    else:
        monkeypatch.setattr(
            xgr.GrammarMatcher, "rollback", MagicMock(side_effect=exception_type("injected rollback failure"))
        )
        assert req.output_constraint.fill_masks(torch.empty((2, 9), dtype=torch.int32), [ord("b")])

    next_token = ord("b") if failure == "rollback" else ord("!")
    InferReqUpdatePack(req, 2).handle(next_token, -0.2, -1, [256], is_master_in_dp=True)
    InferReqUpdatePack(req, 3).handle(ord("z"), -0.3, -1, [256], is_master_in_dp=True)

    # In-flight tokens can update KV history, but output publication stops at the error.
    assert req.finish_status.is_finished_error() and shm_req.finish_status.is_finished_error()
    assert req.output_constraint.is_terminated()
    assert shm_req.shm_prompt_ids.arr[1:4].tolist() == [ord("a"), next_token, ord("z")]
    assert req.output_constraint.error == f"injected {failure} failure"
    assert shm_req.candetoken_out_len == shm_req.shm_cur_output_len == 2
    assert shm_req.finish_token_index == 2


def test_draft_length_failure_disables_mask(compiler):
    req = make_req(regular_constraint="ab")
    init_request(compiler, req)
    state = req.output_constraint
    bitmask = torch.empty((1, 9), dtype=torch.int32)
    draft_tokens = MagicMock()
    draft_tokens.__len__.side_effect = BaseException("injected draft_length failure")

    assert not state.fill_masks(bitmask, draft_tokens)
    assert state.error == "injected draft_length failure"
    assert state.is_terminated()
    assert state.matcher.accept_token(ord("a"))
