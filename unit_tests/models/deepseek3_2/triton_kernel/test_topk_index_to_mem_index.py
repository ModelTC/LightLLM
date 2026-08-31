import torch
import pytest
from lightllm.models.deepseek3_2.triton_kernel.topk_index_to_mem_index import (
    trans_topk_index_to_mem_index,
)


@pytest.mark.parametrize("topk", [2048, 2176])
def test_trans_topk_index_to_mem_index(topk):
    """Test trans_topk_index_to_mem_index converts topk indices to memory indices correctly."""
    batch_size = 1

    # Create topk_index tensor with some valid indices and some -1 (padding)
    topk_index = torch.zeros((batch_size, topk), dtype=torch.int32, device="cuda")
    topk_index[:, 0 : topk - 1] = torch.arange(0, topk - 1, dtype=torch.int32, device="cuda")
    topk_index[:, -1] = -1
    ragged_start_index = torch.tensor([2], dtype=torch.int32, device="cuda")

    # Create ragged_mem_index lookup table
    ragged_mem_index = torch.arange(0, topk + 2, dtype=torch.int32, device="cuda") + 10

    topk_mem_index = trans_topk_index_to_mem_index(topk_index, ragged_start_index, ragged_mem_index)

    expected_index = torch.cat(
        (
            torch.arange(2, topk + 1, dtype=torch.int32, device="cuda"),
            torch.tensor([-1], dtype=torch.int32, device="cuda"),
        )
    ).view(1, -1)
    expected_mem_index = torch.where(expected_index != -1, expected_index + 10, -1)
    assert torch.equal(topk_index, expected_index)
    assert torch.equal(topk_mem_index, expected_mem_index)


if __name__ == "__main__":
    pytest.main()
