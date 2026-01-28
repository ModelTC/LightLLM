import torch
import pytest

from lightllm.models.qwen2_vl.triton_kernel.get_mrope_position_ids import get_mrope_position_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 支持")
def test_get_mrope_position_triton():
    """
    测试 get_mrope_position_triton 函数的正确性
    """
    b_image_start_idx = torch.tensor([0, 0, 4], dtype=torch.int32, device="cuda")
    b_image_thwd = torch.tensor([[1, 2, 2, -2], [1, 2, 2, -2], [1, 2, 2, -2]], dtype=torch.int32, device="cuda")
    b_image_nums = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    b_image_start_num = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    b_image_len = torch.tensor([4, 4, 4], dtype=torch.int32, device="cuda")
    position_ids = (
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(3, -1)
        .contiguous()
    )
    b_ready_cache_len = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    b_q_seq_len = torch.tensor([7, 13], dtype=torch.int32, device="cuda")
    b_start_loc = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
    get_mrope_position_triton(
        b_image_start_idx,
        b_image_thwd,
        b_image_nums,
        b_image_start_num,
        b_image_len,
        position_ids,
        b_ready_cache_len,
        b_q_seq_len,
        b_start_loc,
    )
    # 预期的输出结果
    expected_output = torch.tensor(
        [
            [0, 0, 0, 0, 2, 3, 4, 0, 0, 0, 0, 2, 2, 2, 2, 4, 5, 6, 7, 8],
            [0, 0, 1, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
            [0, 1, 0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    assert torch.equal(position_ids, expected_output), "position_ids 输出与预期结果不一致"
