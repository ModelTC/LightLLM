import gc
import weakref

from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl import (
    PDChunkedPrefillForPrefillNode,
)


class _ShmReq:
    def __init__(self, input_len: int) -> None:
        self.input_len = input_len


class _Req:
    def __init__(self, input_len: int) -> None:
        self.cur_kv_len = input_len
        self.shm_req = _ShmReq(input_len=input_len)


class _Backend:
    pass


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
