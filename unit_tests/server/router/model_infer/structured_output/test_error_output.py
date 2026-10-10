from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xgrammar as xgr

from unit_tests.server.grammar_helpers import init_request, make_req
from lightllm.server.router.model_infer.infer_batch import InferReq, InferReqUpdatePack


@pytest.mark.parametrize("failure", ["mask", "commit"])
def test_grammar_failure_finishes_after_publishing_inflight_token(compiler, monkeypatch, failure):
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
            xgr.GrammarMatcher, "fill_next_token_bitmask", MagicMock(side_effect=RuntimeError("injected mask failure"))
        )
        assert not req.output_constraint.fill_masks(torch.empty((1, 9), dtype=torch.int32), [])

    InferReqUpdatePack(req, 2).handle(ord("!"), -0.2, -1, [256], is_master_in_dp=True)
    InferReqUpdatePack(req, 3).handle(ord("z"), -0.3, -1, [256], is_master_in_dp=True)

    # In-flight tokens can update KV history, but output publication stops at the error.
    assert req.finish_status.is_finished_error() and shm_req.finish_status.is_finished_error()
    assert shm_req.shm_prompt_ids.arr[1:4].tolist() == list(b"a!z")
    assert shm_req.candetoken_out_len == shm_req.shm_cur_output_len == 2
    assert shm_req.finish_token_index == 2
