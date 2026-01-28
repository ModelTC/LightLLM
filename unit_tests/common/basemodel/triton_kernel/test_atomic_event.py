import torch
import pytest
from lightllm.common.basemodel.triton_kernel.atomic_event import wait_value, add_value
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def test_add_in_place():
    input = torch.zeros((1,), device="cuda", dtype=torch.int32)
    wait_value(input, 0)
    add_value(input)
    wait_value(input, 1)
    add_value(input)
    wait_value(input, 2)
    add_value(input)
    wait_value(input, 3)
    assert input.item() == 3, "最终值应为 3"


# @pytest.mark.timeout(2)
# def test_wait_timeout():
#     input = torch.zeros((1,), device="cuda", dtype=torch.int32)
#     wait_value(input, 4)


if __name__ == "__main__":
    pytest.main()
