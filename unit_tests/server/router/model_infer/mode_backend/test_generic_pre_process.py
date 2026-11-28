from types import SimpleNamespace

import numpy as np
import torch
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.server.router.model_infer.mode_backend import generic_pre_process
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


def _patch_empty_input_context(monkeypatch):
    mem_manager = SimpleNamespace(
        HOLD_TOKEN_MEMINDEXES=(-1,),
        alloc=lambda size: torch.empty((size,), dtype=torch.int32),
    )
    infer_context = SimpleNamespace(
        args=SimpleNamespace(page_size=1, run_mode="normal"),
        req_manager=SimpleNamespace(
            HOLD_REQUEST_ID=-1,
            mem_manager=mem_manager,
            req_to_token_indexs=torch.zeros((16, 32), dtype=torch.int32),
        ),
        radix_cache=None,
    )
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
        take_token_lookback=lambda: (),
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
        take_token_lookback=lambda: (),
    )


def test_prepare_prefill_inputs_allows_empty_batch():
    model_input, run_reqs = generic_pre_process.prepare_prefill_inputs([], is_chuncked_mode=True)

    assert run_reqs == []
    assert model_input.batch_size == 0
    assert model_input.input_ids.shape == (0,)
    assert model_input.b_req_idx.shape == (0,)
    assert model_input.b_prefill_start_loc.shape == (0,)
    assert model_input.b_prefill_has_output_cpu == []
    assert model_input.max_q_seq_len == 0
    assert model_input.max_kv_seq_len == 0


def test_prepare_decode_inputs_allows_empty_batch():
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


def test_batch_collects_cached_lookback_once(monkeypatch):
    req = _make_decode_req(req_idx=7)
    req.mtp_step = 2
    req.token_lookback_taken = False
    req.get_input_token_ids = lambda: np.arange(4)
    req.take_token_lookback = lambda: InferReq.take_token_lookback(req)
    monkeypatch.setattr(g_infer_context, "token_lookback_depth", 3)

    model_input, run_reqs = generic_pre_process.prepare_decode_inputs([req])
    assert run_reqs == [req, req, req]
    assert model_input.token_lookback == [(7, 0, 0), (7, 1, 1), (7, 2, 2)]

    model_input, _ = generic_pre_process.prepare_decode_inputs([req])
    assert model_input.token_lookback == []


def test_uncached_prefill_consumes_lookback_before_next_chunk(monkeypatch):
    req = _make_prefill_req(req_idx=7, token_num=4)
    req.token_lookback_taken = False
    req.get_input_token_ids = lambda: np.arange(4)
    req.get_chuncked_input_token_ids = lambda: np.arange(3)
    req.take_token_lookback = lambda: InferReq.take_token_lookback(req)
    monkeypatch.setattr(g_infer_context, "token_lookback_depth", 3)

    first_input, _ = generic_pre_process.prepare_prefill_inputs([req], is_chuncked_mode=True)
    assert first_input.token_lookback == []
    assert req.token_lookback_taken

    req.cur_kv_len = 3
    next_input, _ = generic_pre_process.prepare_prefill_inputs([req], is_chuncked_mode=False)
    assert next_input.input_ids.tolist() == [3]
    assert next_input.token_lookback == []


def test_overlap_prefill_balances_request_token_load_without_padding():
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


def test_overlap_prefill_balances_single_token_request_normally():
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


def test_overlap_decode_builds_two_unpadded_inputs():
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


def test_overlap_decode_preserves_empty_microbatch():
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


class _ReadyPush:
    def __init__(self) -> None:
        self.sent = []

    def send(self, message) -> None:
        self.sent.append(message)


def _report_backend(run_mode: str, is_master_in_dp: bool, disable_chunked_prefill: bool) -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(run_mode=run_mode),
        is_master_in_dp=is_master_in_dp,
        disable_chunked_prefill=disable_chunked_prefill,
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
    )


def _attach_prompt(req, group_request_id: int, prompt_ids: list, pd_master_node_id: int) -> None:
    req.shm_req.index_in_shm_mem = group_request_id
    req.shm_req.sample_params.group_request_id = group_request_id
    req.sampling_param = SimpleNamespace(pd_master_node_id=pd_master_node_id)


def test_only_selected_final_prefill_chunks_signal_decode_allocation(monkeypatch):
    monkeypatch.setattr(
        "lightllm.server.router.model_infer.mode_backend.base_backend.ReportIpc.Push",
        _ReadyPush,
    )
    backend = _report_backend(run_mode="prefill", is_master_in_dp=True, disable_chunked_prefill=False)
    reqs = [_make_prefill_req(i, 8) for i in range(3)]
    for i, req in enumerate(reqs):
        req.shm_req = SimpleNamespace(input_len=8, sample_params=SimpleNamespace(pd_prefill_first=i != 2))
        req.get_chuncked_input_token_len = lambda: 8
        _attach_prompt(req, group_request_id=i, prompt_ids=[i], pd_master_node_id=7)
    # The first request still has another chunk. The second resumes from a cached prefix.
    reqs[0].get_chuncked_input_token_len = lambda: 4
    reqs[1].cur_kv_len = 6
    for req in reqs:
        ModeBackend._report_prefill_first_last_chunk(backend, req_obj=req)
    assert [hasattr(req, "pd_prefill_prompt_ids_reported") for req in reqs] == [False, True, False]

    backend.disable_chunked_prefill = True
    ModeBackend._report_prefill_first_last_chunk(backend, req_obj=reqs[0])
    assert hasattr(reqs[0], "pd_prefill_prompt_ids_reported")


def test_last_chunk_is_reported_once_by_dp_rank0(monkeypatch):
    monkeypatch.setattr(
        "lightllm.server.router.model_infer.mode_backend.base_backend.ReportIpc.Push",
        _ReadyPush,
    )
    backend = _report_backend(run_mode="prefill", is_master_in_dp=True, disable_chunked_prefill=False)
    req = _make_prefill_req(req_idx=1, token_num=8)
    req.cur_kv_len = 6
    req.shm_req = SimpleNamespace(input_len=8, sample_params=SimpleNamespace(pd_prefill_first=True))
    req.get_chuncked_input_token_len = lambda: 8
    _attach_prompt(req, group_request_id=9, prompt_ids=[-(2 ** 63), 2 ** 63 - 1], pd_master_node_id=100)

    ModeBackend._report_prefill_first_last_chunk(backend, req_obj=req)
    ModeBackend._report_prefill_first_last_chunk(backend, req_obj=req)

    (report,) = backend.pd_prefill_prompt_ids_push.sent
    assert (report.group_request_id, report.pd_master_node_id, report.shm_index, report.input_len) == (
        9,
        100,
        req.shm_req.index_in_shm_mem,
        req.shm_req.input_len,
    )
    assert hasattr(req, "pd_prefill_prompt_ids_reported")

    backend.is_master_in_dp = False
    other = _make_prefill_req(req_idx=2, token_num=8)
    other.cur_kv_len = 6
    other.shm_req = SimpleNamespace(input_len=8, sample_params=SimpleNamespace(pd_prefill_first=True))
    other.get_chuncked_input_token_len = lambda: 8
    ModeBackend._report_prefill_first_last_chunk(backend, req_obj=other)
    assert len(backend.pd_prefill_prompt_ids_push.sent) == 1
    assert not hasattr(other, "pd_prefill_prompt_ids_reported")


def test_decode_node_does_not_signal_last_chunk():
    backend = _report_backend(run_mode="decode", is_master_in_dp=True, disable_chunked_prefill=True)
    req = _make_prefill_req(0, 8)
    req.shm_req = SimpleNamespace(input_len=8, sample_params=SimpleNamespace(pd_prefill_first=True))
    ModeBackend._report_prefill_first_last_chunk(backend, req_obj=req)
    assert not hasattr(req, "pd_prefill_prompt_ids_reported")
