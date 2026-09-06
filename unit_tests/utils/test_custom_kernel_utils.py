import torch
import time
import pytest
from lightllm.utils.custom_kernel_utis import pad2dim_tensor_to_new_batch, torch_cat_3


def test_torch_cat():
    a = torch.tensor([[[1, 2], [3, 4]]], device="cuda")
    b = torch.tensor([[[5, 6], [7, 8]]], device="cuda")
    c = torch_cat_3([a, b], dim=0)
    torch.equal(torch.cat((a, b), dim=0), c)

    d = torch_cat_3([a, b], dim=1)
    torch.equal(torch.cat((a, b), dim=1), d)

    e = torch_cat_3([a, b], dim=-1)
    torch.equal(torch.cat((a, b), dim=-1), e)

    empty = torch.empty((0, 2), device="cuda")
    torch_cat_3([a, empty, b], dim=0)
    return


def test_pad2dim_tensor_to_new_batch():
    input_tensor = torch.tensor([[1.0, 2.0]])
    padded_tensor = pad2dim_tensor_to_new_batch(input=input_tensor, new_batch_size=3)
    assert torch.equal(padded_tensor, torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))

    empty_input = torch.empty((0, 2))
    padded_empty_input = pad2dim_tensor_to_new_batch(input=empty_input, new_batch_size=2)
    assert torch.equal(padded_empty_input, torch.zeros((2, 2)))


if __name__ == "__main__":
    pytest.main()
