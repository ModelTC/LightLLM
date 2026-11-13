import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.softmax_topk import softmax_topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("experts,topk", [(1024, 8), (256, 5), (127, 5)])
@pytest.mark.parametrize("renorm", [False, True])
def test_softmax_topk(experts, topk, renorm):
    torch.manual_seed(42)
    gating = torch.randn(32, experts, device="cuda")
    expected_weights, expected_ids = torch.softmax(gating, dim=-1).topk(topk, dim=-1)
    if renorm:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    weights, ids = softmax_topk(gating, topk, renorm)

    torch.testing.assert_close(ids.long(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=1e-3)
