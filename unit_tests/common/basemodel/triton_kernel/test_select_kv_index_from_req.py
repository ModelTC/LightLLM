import pytest
import torch
from types import SimpleNamespace

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import (
    select_kv_index_from_req,
    select_kv_index_from_req_prefill,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_select_kv_index_from_req_for_decode():
    req_to_token_indexs = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    mem_indexes = select_kv_index_from_req(
        req_to_token_indexs=req_to_token_indexs,
        b_req_idx=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 6], dtype=torch.int32, device="cuda"),
    )

    assert mem_indexes.cpu().tolist() == [22, 15]


def test_select_kv_index_from_req_for_prefill():
    req_to_token_indexs = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    mem_indexes = select_kv_index_from_req_prefill(
        req_to_token_indexs=req_to_token_indexs,
        b_req_idx=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([4, 7], dtype=torch.int32, device="cuda"),
        b_ready_cache_len=torch.tensor([2, 3], dtype=torch.int32, device="cuda"),
        b_start_loc=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        max_q_seq_len=4,
        token_num=6,
    )

    assert mem_indexes.cpu().tolist() == [22, 23, 13, 14, 15, 16]


def test_model_selects_reserved_indexes_when_execution_starts():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = SimpleNamespace(page_size=4)
    model.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int32,
            device="cuda",
        )
    )
    model_input = SimpleNamespace(
        is_prefill=False,
        mem_indexes_from_req_table=True,
        b_req_idx=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 4], dtype=torch.int32, device="cuda"),
        mem_indexes=None,
    )

    model._select_page_mem_indexes(model_input)

    assert model_input.mem_indexes.cpu().tolist() == [22, 13]
