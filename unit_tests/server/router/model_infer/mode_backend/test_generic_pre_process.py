from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend import generic_pre_process


def _patch_empty_input_context(monkeypatch):
    mem_manager = SimpleNamespace(
        HOLD_TOKEN_MEMINDEXES=(-1,),
        alloc=lambda size: torch.empty((size,), dtype=torch.int32),
    )
    infer_context = SimpleNamespace(
        backend=SimpleNamespace(model=SimpleNamespace(token_lookback_depth=0)),
        args=SimpleNamespace(page_size=1),
        req_manager=SimpleNamespace(
            HOLD_REQUEST_ID=-1,
            mem_manager=mem_manager,
            req_to_token_indexs=torch.zeros((16, 32), dtype=torch.int32),
        ),
        radix_cache=None,
    )
    monkeypatch.setattr(generic_pre_process, "g_infer_context", infer_context)
    return infer_context


def _patch_overlap_input_context(monkeypatch):
    return _patch_empty_input_context(monkeypatch)


def _make_prefill_req(req_idx: int, token_num: int):
    input_token_ids = [req_idx] * token_num
    return SimpleNamespace(
        req_idx=req_idx,
        cur_kv_len=0,
        hold_kv_len=token_num,
        multimodal_params={"images": [], "audios": []},
        get_chuncked_input_token_ids=lambda: input_token_ids,
        get_input_token_ids=lambda: input_token_ids,
        get_cur_total_len=lambda: token_num,
    )


def _make_decode_req(req_idx: int):
    return SimpleNamespace(
        req_idx=req_idx,
        cur_kv_len=3,
        hold_kv_len=4,
        mtp_step=0,
        multimodal_params={"images": [], "audios": []},
        shared_kv_node=None,
        get_cur_total_len=lambda: 4,
        get_radix_cache_shared_len=lambda: 0,
    )


def test_prepare_prefill_inputs_allows_empty_batch(monkeypatch):
    _patch_empty_input_context(monkeypatch)

    model_input, run_reqs = generic_pre_process.prepare_prefill_inputs([], is_chuncked_mode=True)

    assert run_reqs == []
    assert model_input.batch_size == 0
    assert model_input.input_ids.shape == (0,)
    assert model_input.b_req_idx.shape == (0,)
    assert model_input.b_prefill_start_loc.shape == (0,)
    assert model_input.b_prefill_has_output_cpu == []
    assert model_input.max_q_seq_len == 0
    assert model_input.max_kv_seq_len == 0


def test_prepare_decode_inputs_allows_empty_batch(monkeypatch):
    _patch_empty_input_context(monkeypatch)

    model_input, run_reqs = generic_pre_process.prepare_decode_inputs([])

    assert run_reqs == []
    assert model_input.batch_size == 0
    assert model_input.input_ids is None
    assert model_input.b_req_idx.shape == (0,)
    assert model_input.b_position_delta.shape == (0,)
    assert model_input.b_shared_seq_len.shape == (0,)
    assert model_input.b_shared_radix_node_id.shape == (0,)
    assert model_input.max_q_seq_len == 1
    assert model_input.max_kv_seq_len == 0


def test_overlap_prefill_balances_request_token_load_without_padding(monkeypatch):
    _patch_overlap_input_context(monkeypatch)
    reqs = [
        _make_prefill_req(req_idx=0, token_num=8),
        _make_prefill_req(req_idx=1, token_num=7),
        _make_prefill_req(req_idx=2, token_num=6),
        _make_prefill_req(req_idx=3, token_num=5),
        _make_prefill_req(req_idx=4, token_num=1),
    ]

    (
        model_input0,
        run_reqs0,
        model_input1,
        run_reqs1,
    ) = generic_pre_process.overlap_prepare_prefill_inputs(reqs)

    assert [req.req_idx for req in run_reqs0] == [0, 3, 4]
    assert [req.req_idx for req in run_reqs1] == [1, 2]
    assert model_input0.b_req_idx.tolist() == [0, 3, 4]
    assert model_input1.b_req_idx.tolist() == [1, 2]
    assert model_input0.input_ids.shape == (14,)
    assert model_input1.input_ids.shape == (13,)
    assert model_input0.batch_size == 3
    assert model_input1.batch_size == 2


def test_overlap_prefill_balances_single_token_request_normally(monkeypatch):
    _patch_overlap_input_context(monkeypatch)
    req = _make_prefill_req(req_idx=7, token_num=1)

    (
        model_input0,
        run_reqs0,
        model_input1,
        run_reqs1,
    ) = generic_pre_process.overlap_prepare_prefill_inputs([req])

    assert run_reqs0 == [req]
    assert model_input0.batch_size == 1
    assert model_input0.input_ids.tolist() == [7]
    assert model_input0.b_req_idx.tolist() == [7]
    assert run_reqs1 == []
    assert model_input1.batch_size == 0
    assert model_input1.input_ids.shape == (0,)
    assert model_input1.b_req_idx.shape == (0,)


def test_overlap_decode_builds_two_unpadded_inputs(monkeypatch):
    _patch_overlap_input_context(monkeypatch)
    reqs = [_make_decode_req(req_idx=index) for index in range(3)]

    (
        model_input0,
        run_reqs0,
        decode_reqs0,
        model_input1,
        run_reqs1,
        decode_reqs1,
    ) = generic_pre_process.overlap_prepare_decode_inputs(reqs)

    assert decode_reqs0 == reqs[:2]
    assert decode_reqs1 == reqs[2:]
    assert run_reqs0 == reqs[:2]
    assert run_reqs1 == reqs[2:]
    assert model_input0.batch_size == 2
    assert model_input1.batch_size == 1
    assert model_input0.b_req_idx.tolist() == [0, 1]
    assert model_input1.b_req_idx.tolist() == [2]


def test_overlap_decode_preserves_empty_microbatch(monkeypatch):
    _patch_overlap_input_context(monkeypatch)
    req = _make_decode_req(req_idx=7)

    (
        model_input0,
        run_reqs0,
        decode_reqs0,
        model_input1,
        run_reqs1,
        decode_reqs1,
    ) = generic_pre_process.overlap_prepare_decode_inputs([req])

    assert decode_reqs0 == [req]
    assert decode_reqs1 == []
    assert run_reqs0 == [req]
    assert model_input0.batch_size == 1
    assert run_reqs1 == []
    assert model_input1.batch_size == 0
    assert model_input1.b_req_idx.shape == (0,)


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("start", [0, 1, 2, 3, 7])
def test_token_lookback_is_captured_once_before_current_token(monkeypatch, is_prefill, start):
    context = _patch_empty_input_context(monkeypatch)
    context.backend.model.token_lookback_depth = 3
    ids = np.arange(20, 20 + start + 2, dtype=np.int64)
    if start > 2:
        ids[start - 2] = 1 << 61  # Image cache IDs must retain their int64 range.
    req = InferReq.__new__(InferReq)
    req.req_idx = 5
    req.cur_kv_len = start
    req.cur_output_len = 4  # A resumed generation may already extend beyond the prompt.
    req.token_history_ready = False
    req.mtp_step = 2
    req.multimodal_params = {"images": [], "audios": []}
    req.shared_kv_node = None
    req.get_input_token_ids = lambda: ids
    req.get_chuncked_input_token_ids = lambda: ids[: start + 1]
    req.get_cur_total_len = lambda: start + 1

    def prepare():
        if is_prefill:
            return generic_pre_process.prepare_prefill_inputs([req], is_chuncked_mode=True)[0]
        return generic_pre_process.prepare_decode_inputs([req])[0]

    model_input = prepare()
    expected = [(5, start, ids[max(0, start - 3) : start].tolist())] if start else None
    assert model_input.token_lookback == expected
    assert req.token_history_ready
    if not is_prefill:
        assert model_input.b_req_idx.tolist() == [5, 5, 5]
        assert model_input.batch_size == 3  # Lookback remains request-aligned, not draft-row-aligned.
    if start:
        ids[:start] = -999
        assert model_input.token_lookback == expected  # Own the snapshot while CPU tokens advance.
    if not is_prefill:
        req.get_input_token_ids = lambda: pytest.fail("ordinary decode must not fetch CPU token history")
    assert prepare().token_lookback is None


def test_freeing_request_kv_requires_local_lookback_again(monkeypatch):
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    context = _patch_empty_input_context(monkeypatch)
    context.backend.model.token_lookback_depth = 3
    request = InferReq.__new__(InferReq)
    request.req_idx = 0
    request.cur_kv_len = request.hold_kv_len = 4
    request.token_history_ready = True
    request.shm_req = SimpleNamespace(shm_cur_kv_len=4)
    InferenceContext.free_a_req_mem(context, [], request)
    assert not request.token_history_ready
    request.cur_kv_len = 3
    request.get_input_token_ids = lambda: np.array([11, 12, 13, 14], dtype=np.int64)
    assert request.get_token_lookback(3) == (0, 3, [11, 12, 13])
