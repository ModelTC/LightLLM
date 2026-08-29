import pytest
import torch
import torch.nn.functional as F

from lightllm.models.deepseek_v4.triton_kernel.moe_topk import deepseek_v4_eplb_topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_hash", [False, True])
@pytest.mark.parametrize("record_load", [False, True])
@pytest.mark.parametrize("token_num", [1, 4])
def test_deepseek_v4_eplb_topk_matches_reference(is_hash, record_load, token_num):
    torch.manual_seed(0)
    device = "cuda"
    num_experts, topk = 256, 6
    logits = torch.randn((token_num, num_experts), device=device, dtype=torch.float32)
    bias = None if is_hash else torch.randn((num_experts,), device=device, dtype=torch.float32)
    input_tokens = torch.arange(token_num, device=device, dtype=torch.long)
    hash_indices = torch.randint(num_experts, (token_num + 4, topk), device=device) if is_hash else None
    logical_to_physical = torch.arange(num_experts * 2, device=device, dtype=torch.int32).view(num_experts, 2)
    replica_count = torch.full((num_experts,), 2, device=device, dtype=torch.int32)
    counter = torch.zeros((3, num_experts), device=device, dtype=torch.int64)

    weights, physical_ids, logical_ids = deepseek_v4_eplb_topk(
        logits=logits,
        bias=bias,
        input_tokens=input_tokens if is_hash else None,
        hash_indices_table=hash_indices,
        topk=topk,
        routed_scaling_factor=1.7,
        logical_to_physical_map=logical_to_physical,
        logical_replica_count=replica_count,
        expert_counter=counter,
        sample_index=1,
        record_load=record_load,
        return_logical_ids=True,
    )
    scores = F.softplus(logits).sqrt()
    expected_logical_ids = hash_indices[input_tokens] if is_hash else (scores + bias).topk(topk, dim=-1).indices
    expected_weights = scores.gather(1, expected_logical_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 1.7
    token_indices = torch.arange(token_num, device=device, dtype=torch.int64).view(-1, 1)
    if token_num == 1:
        replica_indices = torch.zeros_like(expected_logical_ids)
    else:
        token_phase = (token_indices * 2654435769) & 0xFFFFFFFF
        expert_phase = (expected_logical_ids * 2246822519) & 0xFFFFFFFF
        replica_indices = ((token_phase + expert_phase) & 0xFFFFFFFF) % 2
    expected_physical_ids = logical_to_physical[expected_logical_ids, replica_indices].to(torch.long)
    torch.cuda.synchronize()

    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(logical_ids, expected_logical_ids)
    torch.testing.assert_close(physical_ids, expected_physical_ids)
    expected_counter = torch.zeros_like(counter)
    if record_load:
        expected_counter[1].scatter_add_(
            0,
            expected_logical_ids.reshape(-1),
            torch.ones_like(expected_logical_ids.reshape(-1), dtype=torch.int64),
        )
    torch.testing.assert_close(counter, expected_counter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_deepseek_v4_eplb_topk_empty_input():
    empty_logits = torch.empty((0, 256), device="cuda", dtype=torch.float32)
    map_ = torch.arange(256, device="cuda", dtype=torch.int32).view(256, 1)
    counter = torch.zeros((1, 256), device="cuda", dtype=torch.int64)
    weights_out = torch.empty((0, 6), device="cuda", dtype=torch.float32)
    physical_ids_out = torch.empty((0, 6), device="cuda", dtype=torch.long)
    logical_ids_out = torch.empty((0, 6), device="cuda", dtype=torch.long)
    weights, physical_ids, logical_ids = deepseek_v4_eplb_topk(
        logits=empty_logits,
        bias=torch.zeros((256,), device="cuda"),
        input_tokens=None,
        hash_indices_table=None,
        topk=6,
        routed_scaling_factor=1.0,
        logical_to_physical_map=map_,
        logical_replica_count=torch.ones((256,), device="cuda", dtype=torch.int32),
        expert_counter=counter,
        sample_index=0,
        record_load=True,
        return_logical_ids=True,
        weights_out=weights_out,
        physical_ids_out=physical_ids_out,
        logical_ids_out=logical_ids_out,
    )
    assert weights is weights_out
    assert physical_ids is physical_ids_out
    assert logical_ids is logical_ids_out
    assert counter.sum().item() == 0
