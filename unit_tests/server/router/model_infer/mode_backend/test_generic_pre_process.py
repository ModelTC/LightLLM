from types import SimpleNamespace

import torch
from lightllm.server.router.model_infer.mode_backend import generic_pre_process


def _patch_empty_input_context(monkeypatch):
    mem_manager = SimpleNamespace(
        HOLD_TOKEN_MEMINDEX=-1,
        alloc=lambda size: torch.empty((size,), dtype=torch.int32),
    )
    infer_context = SimpleNamespace(
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=-1, mem_manager=mem_manager),
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
        multimodal_params={"images": [], "audios": []},
        get_chuncked_input_token_ids=lambda: input_token_ids,
        get_input_token_ids=lambda: input_token_ids,
        get_cur_total_len=lambda: token_num,
    )


def _make_decode_req(req_idx: int):
    return SimpleNamespace(
        req_idx=req_idx,
        cur_kv_len=3,
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
    assert model_input.mem_indexes_cpu.shape == (0,)
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
    assert model_input.mem_indexes_cpu.shape == (0,)
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
    assert model_input1.mem_indexes_cpu.shape == (0,)
