import pytest
import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import _eplb_replica_index


@triton.jit
def _hash_periodic_tokens(output, STRIDE: tl.constexpr, COPIES: tl.constexpr, COUNT: tl.constexpr):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    replica = _eplb_replica_index(index * STRIDE, tl.full((256,), 7, tl.int32), tl.full((256,), COPIES, tl.int32))
    tl.store(output + index, replica, index < COUNT)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("copies", [2, 3, 4, 8])
@pytest.mark.parametrize("stride", [1, 2, 4, 8, 256])
def test_replica_hash_breaks_periodic_token_aliasing(copies, stride):
    count = 4096
    result = torch.empty(count, device="cuda", dtype=torch.int32)
    _hash_periodic_tokens[(triton.cdiv(count, 256),)](result, stride, copies, count)
    histogram = torch.bincount(result.long(), minlength=copies).cpu()
    # The old linear hash sends all stride-4 indices to one of four replicas.
    # This tests distribution, independently of an implementation-shaped oracle.
    expected = count / copies
    assert histogram.min() > 0.80 * expected
    assert histogram.max() < 1.20 * expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_original_periodic_hash_regression():
    result = torch.empty(1024, device="cuda", dtype=torch.int32)
    _hash_periodic_tokens[(4,)](result, 4, 4, 1024)
    assert torch.bincount(result.long(), minlength=4).tolist() == [266, 258, 244, 256]
