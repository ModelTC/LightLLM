from types import SimpleNamespace

import pytest
import torch

from lightllm.common.req_manager import ReqManager


def test_bind_hold_page_initializes_only_padding_request():
    req_manager = ReqManager.__new__(ReqManager)
    req_manager.HOLD_REQUEST_ID = 2
    req_manager.req_list = SimpleNamespace(is_all_free=lambda: True)
    req_manager.req_to_token_indexs = torch.full((3, 8), -1, dtype=torch.int32)
    req_manager.page_size = 4
    req_manager.bind_hold_page((100, 101, 102, 103))

    assert not hasattr(req_manager, "mem_manager")
    assert req_manager.req_to_token_indexs[2].tolist() == [100, 101, 102, 103] * 2
    assert req_manager.req_to_token_indexs[:2].eq(-1).all()


def test_hold_request_indexes_require_all_requests_released():
    req_manager = ReqManager.__new__(ReqManager)
    req_manager.req_list = SimpleNamespace(is_all_free=lambda: False)

    with pytest.raises(AssertionError, match="all requests are released"):
        req_manager.bind_hold_page((100, 101, 102, 103))
