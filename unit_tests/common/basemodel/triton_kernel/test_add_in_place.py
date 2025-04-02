import torch
import time
import pytest
from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy
from lightllm.common.basemodel.triton_kernel.add_in_place import add_in_place
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@pytest.mark.parametrize(
    "dim1, dim2, alpha",
    [
        (dim1, dim2, alpha)
        for dim1 in range(1, 1024, 100)
        for dim2 in range(1, 1024, 100)
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.1]
    ],
)
def test_add_in_place(dim1, dim2, alpha):
    input = torch.rand((dim1, dim2), device="cuda")
    other = torch.rand((dim1, dim2), device="cuda")

    output = input + other * alpha
    add_in_place(input, other, alpha=alpha)
    rlt = torch.allclose(input, output, atol=1e-5, rtol=0)
    assert rlt


if __name__ == "__main__":
    pytest.main()
