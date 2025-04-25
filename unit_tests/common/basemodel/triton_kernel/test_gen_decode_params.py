import torch
import pytest
import numpy as np
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.triton_kernel.gen_decode_params import gen_decode_params


def test_gen_decode_params_basic():
    b_seq_len = torch.ones((9,), dtype=torch.int64, device="cuda") * 8192
    (
        b_q_seq_len,
        b1_cu_q_seq_len,
        b_kv_seq_len,
        b1_cu_kv_seq_len,
        position_ids,
        max_q_seq_len,
        max_kv_seq_len,
    ) = gen_decode_params(b_seq_len)

    true_b_q_seq_len = torch.ones_like(b_seq_len)
    b_q_seq_len, b1_cu_q_seq_len, b_kv_seq_len, b1_cu_kv_seq_len, position_ids, max_q_seq_len, max_kv_seq_len

    assert max_q_seq_len == 1
    assert max_kv_seq_len == b_seq_len.max().item()
    assert torch.equal(b_q_seq_len, true_b_q_seq_len)
    assert torch.equal(b1_cu_q_seq_len, torch.nn.functional.pad(torch.cumsum(true_b_q_seq_len, dim=0), (1, 0), value=0))
    assert torch.equal(b_kv_seq_len, b_seq_len)
    assert torch.equal(b1_cu_kv_seq_len, torch.nn.functional.pad(torch.cumsum(b_seq_len, dim=0), (1, 0), value=0))
    assert torch.equal(position_ids, b_seq_len - 1)


if __name__ == "__main__":
    pytest.main()
