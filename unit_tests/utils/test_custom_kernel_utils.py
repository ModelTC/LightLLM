import torch
import time
import pytest
from lightllm.utils.custom_kernel_utis import torch_cat_3


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


if __name__ == "__main__":
    pytest.main()
