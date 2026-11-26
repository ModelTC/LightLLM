"""Exercise real NVFP4 addresses beyond the signed 32-bit element range."""

import pytest
import torch

from lightllm.common.attention.kernels.gqa.nvfp4.decode import decode_attention_nvfp4
from lightllm.common.attention.kernels.gqa.nvfp4.prefill import context_attention_nvfp4
from lightllm.common.basemodel.triton_kernel.kv_copy.dequantize_nvfp4 import dequantize_nvfp4_kv
from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4 import quantize_nvfp4_kv
from lightllm.common.attention.kernels.mla.nvfp4 import _reduce
from unit_tests.common.att_resources.test_nvfp4 import dequantize, reference_attention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "operation,wide",
    [(op, operand) for op in ("prefill", "decode") for operand in ("query", "output", "table")] + [("gather", "table")],
)
def test_gqa_addresses_above_int32_range(operation, wide):
    if torch.cuda.mem_get_info()[0] < 12 * 1024 ** 3:
        pytest.skip("Large-address regression requires 12 GiB of available GPU memory")
    torch.manual_seed(410)
    q = torch.randn(3, 2, 16, dtype=torch.bfloat16, device="cuda")
    raw = torch.randn(3, 2, 16, dtype=q.dtype, device=q.device)
    k = torch.empty((3, 1, 9), dtype=torch.uint8, device=q.device)
    v = torch.empty_like(k)
    ids = torch.arange(3, dtype=torch.int32, device=q.device)
    quantize_nvfp4_kv(raw, ids, k, v)
    table = ids.repeat(3, 1)
    output = torch.empty_like(q)

    def widen(tensor):
        result = torch.empty_strided(
            tensor.shape, (2 ** 30, *tensor.stride()[1:]), dtype=tensor.dtype, device=tensor.device
        )
        result.copy_(tensor)
        return result

    if wide == "query":
        q = widen(q)
    elif wide == "output":
        output = widen(output)
    else:
        table = widen(table)
    request = torch.tensor([2], dtype=torch.int32, device=q.device)
    lengths = torch.tensor([3], dtype=torch.int32, device=q.device)
    positions = torch.arange(1, 4, dtype=torch.int32, device=q.device)
    kd, vd = [dequantize(value, 16, q.dtype) for value in (k, v)]
    if operation == "gather":
        outputs = [torch.empty((3, 1, 16), dtype=q.dtype, device=q.device) for _ in range(2)]
        cu = torch.tensor([0, 3], dtype=torch.int32, device=q.device)
        dequantize_nvfp4_kv(k, v, table, request, lengths, cu, 3, *outputs)
        for actual, expected in zip(outputs, (kd, vd)):
            torch.testing.assert_close(actual.float(), expected, atol=0, rtol=0)
        return
    if operation == "prefill":
        zero = torch.zeros_like(request)
        context_attention_nvfp4(q, k, v, output, request, zero, lengths, zero, table, 3)
    else:

        def allocate(shape, dtype, device):
            return output if tuple(shape) == tuple(q.shape) else torch.empty(shape, dtype=dtype, device=device)

        output = decode_attention_nvfp4(q, k, v, table, request.repeat(3), positions, positions, 3, alloc_func=allocate)
    expected = reference_attention(q, kd, vd, table, request.repeat(3), positions, positions - 1)
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)


def test_mla_split_reduction_offsets_above_int32_range():
    if torch.cuda.mem_get_info()[0] < 18 * 1024 ** 3:
        pytest.skip("Large-address regression requires 18 GiB of available GPU memory")
    heads, dim = 128, 512
    boundary = 2 ** 31 // (heads * dim)
    rows = boundary + 1
    partial = torch.empty((rows, 1, heads, dim), dtype=torch.float32, device="cuda")
    partial[:, :, 0, :128].fill_(1.25)
    lse = torch.zeros((rows, 1, heads), dtype=torch.float32, device="cuda")
    output = torch.empty((rows, heads, dim), dtype=torch.bfloat16, device="cuda")
    _reduce[(rows, 1, 1)](partial, lse, output, heads, dim, 1, 1, 128, num_warps=4)
    actual = output[[0, boundary - 1, boundary], 0, :128]
    torch.testing.assert_close(actual, torch.full_like(actual, 1.25), atol=0, rtol=0)
