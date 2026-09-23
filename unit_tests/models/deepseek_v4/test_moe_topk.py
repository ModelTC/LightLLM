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


def _expected_replica(token_num, logical_ids, replica_count):
    if token_num == 1:
        return torch.zeros_like(logical_ids)
    token_indices = torch.arange(token_num, device=logical_ids.device, dtype=torch.int64).view(-1, 1)
    value = token_indices ^ (((logical_ids + 1) * 0x9E3779B9) & 0xFFFFFFFF)
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) % replica_count[logical_ids].to(torch.int64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_hash", [False, True])
@pytest.mark.parametrize("record_load", [False, True])
@pytest.mark.parametrize("return_logical_ids", [False, True])
def test_deepseek_v4_eplb_topk_mixed_vision_matches_independent_reference(is_hash, record_load, return_logical_ids):
    """Vision IDs at and past vocab must route by bias_vl without hash-table OOB."""
    torch.manual_seed(17)
    device, vocab, experts, topk = "cuda", 32, 256, 6
    # `vocab` is deliberately the first image ID; the last ID is well beyond
    # the hash table and must still use the vision bias safely.
    input_tokens = torch.tensor([2, vocab, vocab + 100_000], dtype=torch.long, device=device)
    logits = torch.randn((len(input_tokens), experts), dtype=torch.float32, device=device)
    text_bias = torch.zeros(experts, dtype=torch.float32, device=device)
    vision_bias = torch.zeros(experts, dtype=torch.float32, device=device)
    text_bias[20:26] = torch.tensor([60.0, 55.0, 50.0, 45.0, 40.0, 35.0], device=device)
    vision_bias[200:206] = torch.tensor([60.0, 55.0, 50.0, 45.0, 40.0, 35.0], device=device)
    table = torch.zeros((vocab, topk), dtype=torch.long, device=device)
    table[2] = torch.tensor([1, 4, 7, 10, 13, 16], dtype=torch.long, device=device)
    maps = torch.arange(experts * 3, dtype=torch.int32, device=device).view(experts, 3)
    replica_count = torch.full((experts,), 3, dtype=torch.int32, device=device)
    counter = torch.zeros((2, experts), dtype=torch.int64, device=device)

    weights, physical, logical = deepseek_v4_eplb_topk(
        logits=logits,
        bias=None if is_hash else text_bias,
        input_tokens=input_tokens,
        hash_indices_table=table if is_hash else None,
        topk=topk,
        routed_scaling_factor=1.5,
        logical_to_physical_map=maps,
        logical_replica_count=replica_count,
        expert_counter=counter,
        sample_index=1,
        record_load=record_load,
        return_logical_ids=return_logical_ids,
        bias_vl=vision_bias,
        image_token_start=vocab,
    )
    scores = F.softplus(logits).sqrt()
    text_ids = table[input_tokens[:1]] if is_hash else (scores[:1] + text_bias).topk(topk, dim=-1).indices
    vision_ids = (scores[1:] + vision_bias).topk(topk, dim=-1).indices
    expected_logical = torch.cat((text_ids, vision_ids))
    expected_weights = scores.gather(1, expected_logical)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 1.5
    expected_replica = _expected_replica(len(input_tokens), expected_logical, replica_count)
    expected_physical = maps[expected_logical, expected_replica].to(torch.long)
    torch.cuda.synchronize()

    torch.testing.assert_close(weights, expected_weights, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(physical, expected_physical)
    # The physical map is unique, so this also checks logical routes when the
    # optional logical output is disabled.
    torch.testing.assert_close(physical // 3, expected_logical)
    if return_logical_ids:
        torch.testing.assert_close(logical, expected_logical)
    else:
        assert logical is None
    expected_counter = torch.zeros_like(counter)
    if record_load:
        expected_counter[1].scatter_add_(
            0, expected_logical.flatten(), torch.ones(expected_logical.numel(), dtype=torch.int64, device=device)
        )
    torch.testing.assert_close(counter, expected_counter)
    if is_hash:
        assert set(expected_logical[0].tolist()) == set(table[2].tolist())
    else:
        assert set(expected_logical[0].tolist()) == set(range(20, 26))
    assert set(expected_logical[1].tolist()).isdisjoint(set(expected_logical[0].tolist()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_hash", [False, True])
def test_deepseek_v4_eplb_topk_single_vision_token_uses_replica_zero(is_hash):
    device, experts, topk = "cuda", 256, 6
    logits = torch.zeros((1, experts), dtype=torch.float32, device=device)
    vision_bias = torch.zeros(experts, dtype=torch.float32, device=device)
    vision_bias[200:206] = torch.arange(topk, 0, -1, dtype=torch.float32, device=device)
    maps = torch.arange(experts * 4, dtype=torch.int32, device=device).view(experts, 4)
    counter = torch.zeros((1, experts), dtype=torch.int64, device=device)
    _, physical, logical = deepseek_v4_eplb_topk(
        logits=logits,
        bias=None if is_hash else torch.zeros(experts, dtype=torch.float32, device=device),
        input_tokens=torch.tensor([32], dtype=torch.long, device=device),
        hash_indices_table=torch.zeros((32, topk), dtype=torch.long, device=device) if is_hash else None,
        topk=topk,
        routed_scaling_factor=1.0,
        logical_to_physical_map=maps,
        logical_replica_count=torch.full((experts,), 4, dtype=torch.int32, device=device),
        expert_counter=counter,
        sample_index=0,
        record_load=True,
        return_logical_ids=True,
        bias_vl=vision_bias,
        image_token_start=32,
    )
    torch.testing.assert_close(physical, maps[logical, 0].to(torch.long))
    assert counter.sum().item() == topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_deepseek_v4_eplb_topk_validates_vision_arguments():
    kwargs = dict(
        logits=torch.zeros((1, 256), dtype=torch.float32, device="cuda"),
        bias=None,
        input_tokens=torch.tensor([32], dtype=torch.long, device="cuda"),
        hash_indices_table=torch.zeros((32, 6), dtype=torch.long, device="cuda"),
        topk=6,
        routed_scaling_factor=1.0,
        logical_to_physical_map=torch.arange(256, dtype=torch.int32, device="cuda").view(256, 1),
        logical_replica_count=torch.ones(256, dtype=torch.int32, device="cuda"),
        expert_counter=torch.zeros((1, 256), dtype=torch.int64, device="cuda"),
        sample_index=0,
        record_load=False,
    )
    with pytest.raises(RuntimeError, match="vision routing requires"):
        deepseek_v4_eplb_topk(**kwargs, bias_vl=torch.zeros(256, device="cuda"), image_token_start=0)
