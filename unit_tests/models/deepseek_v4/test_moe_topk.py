import pytest
import torch
import torch.nn.functional as F

from lightllm.models.deepseek_v4.triton_kernel.moe_topk import deepseek_v4_eplb_topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("copies", [2, 3, 4, 8])
def test_deepseek_v4_replica_hash_balances_periodic_expert_selections(copies):
    tokens, experts, topk = 4096, 256, 6
    logits = torch.zeros((tokens, experts), device="cuda")
    table = torch.tensor([[0, 1, 2, 3, 4, 5], [7, 1, 2, 3, 4, 5]], device="cuda")
    input_tokens = (torch.arange(tokens, device="cuda") % 4 == 0).long()
    maps = torch.arange(experts * copies, device="cuda", dtype=torch.int32).reshape(experts, copies)
    counter = torch.zeros((1, experts), device="cuda", dtype=torch.int64)
    weights, physical, logical = deepseek_v4_eplb_topk(
        logits=logits,
        bias=None,
        input_tokens=input_tokens,
        hash_indices_table=table,
        topk=topk,
        routed_scaling_factor=1.0,
        logical_to_physical_map=maps,
        logical_replica_count=torch.full((experts,), copies, device="cuda", dtype=torch.int32),
        expert_counter=counter,
        sample_index=0,
        record_load=True,
        return_logical_ids=True,
    )
    torch.testing.assert_close(logical, table[input_tokens])
    torch.testing.assert_close(physical // copies, logical)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(tokens, device="cuda"))
    assert counter[0, 7] == tokens // 4
    histogram = torch.bincount(physical[::4, 0] % copies, minlength=copies)
    assert histogram.min() > 0.75 * (tokens // 4 / copies)
    assert histogram.max() < 1.25 * (tokens // 4 / copies)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_hash", [False, True])
@pytest.mark.parametrize("record_load", [False, True])
@pytest.mark.parametrize("token_num", [1, 4])
@pytest.mark.parametrize("return_logical_ids", [False, True])
def test_deepseek_v4_eplb_topk_matches_reference(is_hash, record_load, token_num, return_logical_ids):
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
        return_logical_ids=return_logical_ids,
    )
    scores = F.softplus(logits).sqrt()
    expected_logical_ids = hash_indices[input_tokens] if is_hash else (scores + bias).topk(topk, dim=-1).indices
    expected_weights = scores.gather(1, expected_logical_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 1.7
    token_indices = torch.arange(token_num, device=device, dtype=torch.int64).view(-1, 1)
    if token_num == 1:
        replica_indices = torch.zeros_like(expected_logical_ids)
    else:
        value = token_indices ^ (((expected_logical_ids + 1) * 0x9E3779B9) & 0xFFFFFFFF)
        value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
        value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
        replica_indices = (value ^ (value >> 16)) % 2
    expected_physical_ids = logical_to_physical[expected_logical_ids, replica_indices].to(torch.long)
    torch.cuda.synchronize()

    torch.testing.assert_close(weights, expected_weights)
    if return_logical_ids:
        torch.testing.assert_close(logical_ids, expected_logical_ids)
    else:
        assert logical_ids is None
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
    )
    assert weights.shape == (0, 6) and weights.dtype == torch.float32
    assert physical_ids.shape == (0, 6) and physical_ids.dtype == torch.long
    assert logical_ids.shape == (0, 6) and logical_ids.dtype == torch.long
    assert counter.sum().item() == 0
