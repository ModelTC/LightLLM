import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_mtp_fa3_decode_params


def _reference_dynamic_mtp_fa3_decode_params(b_req_idx, b_seq_len, b_mark_shared_group, hold_req_id):
    batch_size = b_req_idx.shape[0]
    b_req_idx_cpu = b_req_idx.cpu()
    b_seq_len_cpu = b_seq_len.cpu()
    b_mark_cpu = b_mark_shared_group.cpu()
    valid_idx = torch.nonzero(b_mark_cpu > 0, as_tuple=False).flatten()
    valid_size = valid_idx.numel()

    b_q_seq_len = torch.zeros((batch_size,), dtype=torch.int32)
    b_kv_seq_len = torch.zeros((batch_size,), dtype=torch.int32)
    b_att_req_idx = torch.full((batch_size,), hold_req_id, dtype=torch.int32)
    b_att_seq_len = torch.zeros((batch_size,), dtype=torch.int32)

    b_q_seq_len[:valid_size] = b_mark_cpu[valid_idx]
    b_kv_seq_len[:valid_size] = b_seq_len_cpu[valid_idx]
    b_att_req_idx[:valid_size] = b_req_idx_cpu[valid_idx]
    b_att_seq_len[:valid_size] = b_seq_len_cpu[valid_idx]
    return b_q_seq_len, b_kv_seq_len, b_att_req_idx, b_att_seq_len


@pytest.mark.parametrize("batch_size", [1, 7, 256, 257, 777, 1025, 1537])
def test_build_dynamic_mtp_fa3_decode_params(batch_size):
    hold_req_id = -1
    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda") + 100
    b_seq_len = (torch.arange(batch_size, dtype=torch.int32, device="cuda") % 97) + 1
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")

    mark_positions = [0, 3, 17, 255, 256, 511, batch_size - 1]
    for pos in mark_positions:
        if 0 <= pos < batch_size:
            b_mark_shared_group[pos] = pos % 5 + 1

    actual = build_dynamic_mtp_fa3_decode_params(
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        att_batch_size=batch_size,
        hold_req_id=hold_req_id,
    )
    expected = _reference_dynamic_mtp_fa3_decode_params(
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        hold_req_id=hold_req_id,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor.cpu(), expected_tensor)


def test_build_dynamic_mtp_fa3_decode_params_all_padding():
    batch_size = 513
    hold_req_id = -1
    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.arange(batch_size, dtype=torch.int32, device="cuda") + 1
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")

    actual = build_dynamic_mtp_fa3_decode_params(
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        att_batch_size=batch_size,
        hold_req_id=hold_req_id,
    )
    expected = _reference_dynamic_mtp_fa3_decode_params(
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        hold_req_id=hold_req_id,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor.cpu(), expected_tensor)
