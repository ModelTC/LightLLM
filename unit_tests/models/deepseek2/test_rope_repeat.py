import torch
import time
import pytest
from lightllm.models.deepseek2.triton_kernel.repeat_rope import repeat_rope


def test_torch_cat():
    source = torch.randn((100, 1, 1077), device="cuda")
    dest = torch.randn((100, 7, 1077), device="cuda")

    repeat_rope(dest, source)
    torch.equal(dest[:, 0, :], source)
    torch.equal(dest[:, -1, :], source)
    return


if __name__ == "__main__":
    pytest.main()
