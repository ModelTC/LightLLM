import torch
import pytest
from lightllm.models.deepseek3_2.triton_kernel.topk_index_to_mem_index import trans_topk_index_to_mem_index


def test_trans_topk_index_to_mem_index():
    """Test trans_topk_index_to_mem_index converts topk indices to memory indices correctly."""
    batch_size = 1
    topk = 2048

    # Create topk_index tensor with some valid indices and some -1 (padding)
    topk_index = torch.zeros((batch_size, topk), dtype=torch.int32, device="cuda")
    topk_index[:, 0:2048] = torch.arange(0, 2048, dtype=torch.int32, device="cuda")

    # Create ragged_mem_index lookup table
    ragged_mem_index = torch.arange(0, 2048, dtype=torch.int32, device="cuda") + 10

    topk_mem_index = trans_topk_index_to_mem_index(topk_index, ragged_mem_index)

    assert torch.equal(topk_mem_index, (torch.arange(0, 2048, dtype=torch.int32, device="cuda") + 10).view(1, -1))


if __name__ == "__main__":
    pytest.main()
