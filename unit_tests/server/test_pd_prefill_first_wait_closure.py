import gc
import weakref

from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl import (
    PDChunkedPrefillForPrefillNode,
)


class _ShmReq:
    def __init__(self, input_len: int) -> None:
        self.input_len = input_len


class _TransParams:
    def __init__(self, data_len: int) -> None:
        self.data_len = data_len


class _Req:
    def __init__(self, input_len: int, data_len: int = 0) -> None:
        self.cur_kv_len = input_len
        self.shm_req = _ShmReq(input_len=input_len)
        self.infer_aborted = False
        self.sampling_param = type(
            "SamplingParam",
            (),
            {"shm_param": type("ShmParam", (), {"pd_kv_trans_params": _TransParams(data_len=data_len)})()},
        )()


class _Backend(PDChunkedPrefillForPrefillNode):
    def __init__(self) -> None:
        self.dp_world_size = 1


def test_wait_closure_does_not_capture_req_or_backend():
    gc.disable()
    try:
        req = _Req(input_len=4)
        backend = _Backend()
        req_ref = weakref.ref(req)
        backend_ref = weakref.ref(backend)
        PDChunkedPrefillForPrefillNode._pf_prefill_chuncked_handle_func(
            backend,
            req_obj=req,
            next_token_id=7,
            next_token_prob=-0.2,
            output_len=1,
        )
        wait_decode_kv_info_ready = req.pd_wait_decode_kv_info_ready
        assert wait_decode_kv_info_ready.__code__.co_freevars == ("next_token_id", "next_token_prob")
        assert [cell.cell_contents for cell in wait_decode_kv_info_ready.__closure__] == [7, -0.2]
        del wait_decode_kv_info_ready
        del req
        del backend
        assert req_ref() is None
        assert backend_ref() is None
    finally:
        gc.enable()


def test_ready_data_len_submits_transfer_without_waiting(monkeypatch):
    submitted = {}

    def fake_submit(backend, req_obj, next_token_id, next_token_logprob):
        submitted["call"] = (backend, req_obj, next_token_id, next_token_logprob)

    monkeypatch.setattr(
        "lightllm.server.router.model_infer.mode_backend.pd.helper.submit_prefill_first_tasks_if_ready",
        fake_submit,
    )
    req = _Req(input_len=4, data_len=8)
    backend = _Backend()
    PDChunkedPrefillForPrefillNode._pf_prefill_chuncked_handle_func(
        backend,
        req_obj=req,
        next_token_id=7,
        next_token_prob=-0.2,
        output_len=1,
    )
    assert not hasattr(req, "pd_wait_decode_kv_info_ready")
    assert submitted["call"][0] is backend
    assert submitted["call"][1] is req
    assert submitted["call"][2:] == (7, -0.2)
