import torch
import pytest
import numpy as np
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_prefill_params


def test_gen_prefill_params_basic():

    b_ready_cache_len = torch.ones((9,), dtype=torch.int64, device="cuda")
    b_seq_len = torch.ones((9,), dtype=torch.int64, device="cuda") * 8192
    b_seq_len[0:5] = 4567

    input_token_num = (b_seq_len - b_ready_cache_len).sum().item()
    true_b_q_seq_len = b_seq_len - b_ready_cache_len

    (
        b_q_seq_len,
        b1_cu_q_seq_len,
        b_kv_seq_len,
        b1_cu_kv_seq_len,
        position_ids,
    ) = gen_prefill_params(input_token_num, b_ready_cache_len, b_seq_len)

    assert torch.equal(b_q_seq_len, true_b_q_seq_len)
    assert torch.equal(b1_cu_q_seq_len, torch.nn.functional.pad(torch.cumsum(true_b_q_seq_len, dim=0), (1, 0), value=0))
    assert torch.equal(b_kv_seq_len, b_seq_len)
    assert torch.equal(b1_cu_kv_seq_len, torch.nn.functional.pad(torch.cumsum(b_seq_len, dim=0), (1, 0), value=0))

    b_ready_cache_len_numpy = b_ready_cache_len.cpu().numpy()
    b_seq_len_numpy = b_seq_len.cpu().numpy()
    true_position_ids = torch.from_numpy(
        np.concatenate(
            [np.arange(b_ready_cache_len_numpy[i], b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))],
            axis=0,
        )
    ).cuda()

    assert torch.equal(position_ids, true_position_ids)


if __name__ == "__main__":
    pytest.main()
