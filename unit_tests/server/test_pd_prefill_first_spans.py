import pytest

from lightllm.server.pd_io_struct import ObjType
from lightllm.server.router.model_infer.mode_backend.pd.helper import iter_kv_spans


def test_full_match_still_sends_the_reserved_token():
    spans = iter_kv_spans(start=7, end=8, page_size=4)
    assert spans == [(7, 8)]


def test_pages_follow_pd_kv_page_size():
    spans = iter_kv_spans(start=1, end=6, page_size=2)
    assert spans == [(1, 3), (3, 5), (5, 6)]


def test_empty_span_is_rejected():
    with pytest.raises(ValueError):
        iter_kv_spans(start=4, end=4, page_size=2)


def test_prefill_first_messages_do_not_reuse_the_old_handshake_types():
    assert ObjType.PD_PF_PREFILL_REQ.value == 10
    assert ObjType.PD_PF_PREFILL_KV_READY.value == 11
    assert ObjType.PD_PF_DECODE_KV_READY.value == 12
    assert ObjType.PD_PF_PREFILL_REQ != ObjType.REQ
    assert ObjType.PD_PF_DECODE_KV_READY != ObjType.PD_REQ_DECODE_NODE_INFO
