import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.append_shared_expert_topk import (
    append_fused_shared_experts,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def test_append_fused_shared_experts_without_gate():
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 0]], dtype=torch.int32, device="cuda")
    topk_weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]], dtype=torch.float32, device="cuda")

    out_weights, out_ids = append_fused_shared_experts(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert_start_id=4,
        num_fused_shared_experts=1,
        shared_expert_gate=None,
    )

    expect_ids = torch.tensor([[0, 2, 4], [1, 3, 4], [2, 0, 4]], dtype=torch.int32, device="cuda")
    expect_weights = torch.tensor(
        [[0.2, 0.8, 1.0], [0.4, 0.6, 1.0], [0.7, 0.3, 1.0]], dtype=torch.float32, device="cuda"
    )
    assert torch.equal(out_ids, expect_ids)
    assert torch.allclose(out_weights, expect_weights)


def test_append_fused_shared_experts_with_gate():
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 0]], dtype=torch.int32, device="cuda")
    topk_weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]], dtype=torch.float32, device="cuda")
    shared_expert_gate = torch.tensor([[0.0], [2.0], [-2.0]], dtype=torch.float32, device="cuda")

    out_weights, out_ids = append_fused_shared_experts(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert_start_id=4,
        num_fused_shared_experts=1,
        shared_expert_gate=shared_expert_gate,
    )

    expect_ids = torch.tensor([[0, 2, 4], [1, 3, 4], [2, 0, 4]], dtype=torch.int32, device="cuda")
    expect_weights = torch.cat([topk_weights, torch.sigmoid(shared_expert_gate)], dim=1)
    assert torch.equal(out_ids, expect_ids)
    assert torch.allclose(out_weights, expect_weights)


def test_append_fused_shared_experts_multiple_tokens_per_grid():
    token_num = 4097
    topk_ids = torch.stack(
        [
            torch.arange(token_num, dtype=torch.int32, device="cuda") % 4,
            (torch.arange(token_num, dtype=torch.int32, device="cuda") + 1) % 4,
        ],
        dim=1,
    )
    topk_weights = torch.rand((token_num, 2), dtype=torch.float32, device="cuda")
    shared_expert_gate = torch.randn((token_num, 1), dtype=torch.float32, device="cuda")

    out_weights, out_ids = append_fused_shared_experts(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert_start_id=4,
        num_fused_shared_experts=1,
        shared_expert_gate=shared_expert_gate,
    )

    expect_ids = torch.cat(
        [topk_ids, torch.full((token_num, 1), 4, dtype=torch.int32, device="cuda")],
        dim=1,
    )
    expect_weights = torch.cat([topk_weights, torch.sigmoid(shared_expert_gate)], dim=1)
    assert torch.equal(out_ids, expect_ids)
    assert torch.allclose(out_weights, expect_weights)


def test_append_fused_shared_experts_multi_shared_gate():
    token_num = 7
    topk_ids = torch.stack(
        [
            torch.arange(token_num, dtype=torch.int32, device="cuda") % 4,
            (torch.arange(token_num, dtype=torch.int32, device="cuda") + 1) % 4,
        ],
        dim=1,
    )
    topk_weights = torch.rand((token_num, 2), dtype=torch.float32, device="cuda")
    shared_expert_gate = torch.randn((token_num, 2), dtype=torch.float32, device="cuda")

    out_weights, out_ids = append_fused_shared_experts(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert_start_id=4,
        num_fused_shared_experts=2,
        shared_expert_gate=shared_expert_gate,
    )

    expect_ids = torch.cat(
        [
            topk_ids,
            torch.tensor([[4, 5]], dtype=torch.int32, device="cuda").repeat(token_num, 1),
        ],
        dim=1,
    )
    expect_weights = torch.cat([topk_weights, torch.sigmoid(shared_expert_gate)], dim=1)
    assert torch.equal(out_ids, expect_ids)
    assert torch.allclose(out_weights, expect_weights)
