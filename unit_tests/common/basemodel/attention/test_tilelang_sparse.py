import pytest
import torch

from lightllm.common.basemodel.attention.nsa.tilelang_sparse import pad_sparse_indices


def test_pad_sparse_indices_adds_masked_block_tail():
    indices = torch.arange(65, dtype=torch.int32).view(1, 65)

    padded = pad_sparse_indices(indices)

    assert padded.shape == (1, 1, 128)
    torch.testing.assert_close(padded[0, 0, :65], indices[0])
    assert torch.all(padded[0, 0, 65:] == -1)


def test_pad_sparse_indices_preserves_aligned_storage():
    indices = torch.zeros((4, 1, 128), dtype=torch.int32)

    assert pad_sparse_indices(indices) is indices


def test_pad_sparse_indices_rejects_invalid_shape_or_block():
    with pytest.raises(ValueError, match="2D or 3D"):
        pad_sparse_indices(torch.zeros((2, 3, 4, 5), dtype=torch.int32))
    with pytest.raises(ValueError, match="positive"):
        pad_sparse_indices(torch.zeros((2, 64), dtype=torch.int32), block_size=0)
