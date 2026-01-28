import torch

import triton
import triton.language as tl


@triton.jit
def _wait_value(
    input_ptr,
    expected_value,
):
    value = tl.atomic_add(input_ptr, 0, scope="gpu", sem="acq_rel")
    while value != expected_value:
        value = tl.atomic_add(input_ptr, 0, scope="gpu", sem="acq_rel")


@triton.jit
def _add_value(
    input_ptr,
):
    tl.atomic_add(input_ptr, 1, scope="gpu", sem="acq_rel")


@torch.inference_mode()
def wait_value(input: torch.Tensor, expected_value: int):
    assert input.is_contiguous(), "input tensor must be contiguous"
    _wait_value[(1,)](
        input,
        expected_value,
        num_warps=1,
    )


@torch.inference_mode()
def add_value(input: torch.Tensor):
    assert input.is_contiguous(), "input tensor must be contiguous"
    _add_value[(1,)](
        input,
        num_warps=1,
    )
