import os
from types import SimpleNamespace

import pytest
import torch

from lightllm.utils import envs_utils
from lightllm.models.deepseek_v4 import model as dsv4_model


@pytest.fixture(autouse=True)
def _clear_tile_routing_cache():
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    yield
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()


def test_tile_routing_env_default_and_validation(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", raising=False)
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    assert not envs_utils.get_dsv4_eplb_tile_routing()
    monkeypatch.setenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", "1")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    assert envs_utils.get_dsv4_eplb_tile_routing()
    monkeypatch.setenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", "bad")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    with pytest.raises(ValueError, match="must be 0 or 1"):
        envs_utils.get_dsv4_eplb_tile_routing()
    monkeypatch.delenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()


def test_tile_routing_peak_bytes(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", "1")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    from lightllm.models.deepseek_v4.triton_kernel.eplb_tile_route import tile_routing_peak_nbytes

    expected = 8192 * 6 * 12 + 256 * 4 + 8 * 256 * 4 + 256 * 8 * 4 + 12
    assert tile_routing_peak_nbytes(8192, 6) == expected


def test_tile_routing_model_guard_accepts_dp8_and_rejects_other_topologies(monkeypatch):
    expert = SimpleNamespace(expert_parallel_state=SimpleNamespace(eplb=SimpleNamespace()))
    model = dsv4_model.DeepseekV4TpPartModel.__new__(dsv4_model.DeepseekV4TpPartModel)
    model.is_mtp_draft_model = False
    model.args = SimpleNamespace(enable_prefill_eplb=True, run_mode="prefill")
    model.trans_layers_weight = [SimpleNamespace(experts_=expert)]
    monkeypatch.setenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", "1")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    monkeypatch.setattr(dsv4_model, "get_dp_world_size", lambda: 1)
    monkeypatch.setattr(dsv4_model, "get_node_world_size", lambda: 8)
    assert model._get_eplb_weights() == [expert]
    monkeypatch.setattr(dsv4_model, "get_dp_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="per-DP world size 1"):
        model._get_eplb_weights()
    monkeypatch.setattr(dsv4_model, "get_dp_world_size", lambda: 1)
    monkeypatch.setattr(dsv4_model, "get_node_world_size", lambda: 4)
    with pytest.raises(RuntimeError, match="node world size 8"):
        model._get_eplb_weights()


def test_tile_routing_draft_skips_runtime_guard(monkeypatch):
    model = dsv4_model.DeepseekV4TpPartModel.__new__(dsv4_model.DeepseekV4TpPartModel)
    model.is_mtp_draft_model = True
    model.args = SimpleNamespace(enable_prefill_eplb=True, run_mode="prefill")
    model.trans_layers_weight = []
    monkeypatch.setenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", "1")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    assert model._get_eplb_weights() == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_hash", [False, True])
@pytest.mark.parametrize("capture,record_load", [(False, False), (True, True)])
def test_select_experts_tile_flag_preserves_logical_gates_and_counters(monkeypatch, is_hash, capture, record_load):
    """Tile routing changes only physical ids after the existing EPLB top-k."""
    from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import (
        DeepseekV4TransformerLayerInfer,
    )
    from lightllm.models.deepseek_v4.triton_kernel import eplb_tile_route

    device, vocab, topk = "cuda", 32, 6
    logits = torch.randn((3, 256), dtype=torch.float32, device=device)
    input_ids = torch.tensor([2, vocab, vocab + 4096], dtype=torch.long, device=device)
    mapping = torch.arange(256 * 2, dtype=torch.int32, device=device).view(256, 2)
    replicas = torch.full((256,), 2, dtype=torch.int32, device=device)
    counter = torch.zeros((2, 256), dtype=torch.int64, device=device)
    eplb = SimpleNamespace(
        full_layout=True,
        logical_to_physical_map=mapping,
        logical_replica_count=replicas,
        route_counter=counter,
        recording=record_load,
        num_redundant_experts_per_rank=2,
        next_sample_index=lambda: 1,
    )
    experts = SimpleNamespace(
        expert_parallel_state=SimpleNamespace(
            eplb=eplb, num_logical_experts=256, world_size=8, num_primary_experts_per_rank=32
        ),
        global_rank_=0,
    )
    text_bias = torch.zeros(256, dtype=torch.float32, device=device)
    vision_bias = torch.zeros(256, dtype=torch.float32, device=device)
    text_bias[20:26] = 100
    vision_bias[200:206] = 100
    table = torch.zeros((vocab, topk), dtype=torch.long, device=device)
    table[2] = torch.tensor([1, 4, 7, 10, 13, 16], device=device)
    layer_weight = SimpleNamespace(
        experts_=experts,
        gate_bias_=SimpleNamespace(weight=text_bias),
        gate_bias_vl_=SimpleNamespace(weight=vision_bias),
        gate_tid2eid_=SimpleNamespace(weight=table),
    )
    infer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
    infer.is_hash = is_hash
    infer.has_vision = True
    infer.vocab_size = vocab
    infer.num_experts_per_tok = topk
    infer.routed_scaling_factor = 1.0
    infer.alloc_tensor = torch.empty
    infer_state = SimpleNamespace(is_prefill=True, input_ids=input_ids)

    monkeypatch.setenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", "0")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    weights0, physical0, logical0 = infer._select_experts(logits, infer_state, layer_weight, capture)
    counter0 = counter.clone()
    counter.zero_()
    seen = []

    def fake_route(logical, physical, *args):
        seen.append(logical.clone())
        # Simulate a valid physical remap while leaving the logical choices intact.
        physical.copy_(mapping[logical, 0].to(torch.long))
        return physical

    monkeypatch.setenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", "1")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    monkeypatch.setattr(eplb_tile_route, "route", fake_route)
    weights1, physical1, logical1 = infer._select_experts(logits, infer_state, layer_weight, capture)
    torch.testing.assert_close(weights1, weights0)
    torch.testing.assert_close(seen[0], logical0 if capture else physical0 // 2)
    torch.testing.assert_close(counter, counter0)
    assert logical1 is None if not capture else torch.equal(logical1, logical0)
    assert physical1.shape == physical0.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_production_tile_solver_boundary_and_random_maps():
    """Production JIT solver matches an independent Hall bound on legal maps."""
    import random

    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
        build_logical_to_physical_maps_for_layers,
        validate_physical_placement,
    )
    from lightllm.models.deepseek_v4.triton_kernel.eplb_tile_route import _load_cuda

    monkeypatch_env = "LIGHTLLM_DSV4_EPLB_TILE_ROUTING"
    previous = os.environ.get(monkeypatch_env)
    os.environ[monkeypatch_env] = "1"
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    module = _load_cuda()
    generator = random.Random(20260923)

    def make_layout(redundancy):
        primary = list(range(256))
        generator.shuffle(primary)
        rows = [primary[index * 32 : (index + 1) * 32] for index in range(8)]
        for rank in range(8):
            choices = [expert for expert in range(256) if expert not in rows[rank]]
            generator.shuffle(choices)
            rows[rank].extend(choices[:redundancy])
        placement = torch.tensor(rows, dtype=torch.int32)
        validate_physical_placement(placement.unsqueeze(0), 256)
        mapping, replicas = build_logical_to_physical_maps_for_layers(placement.unsqueeze(0), 256, full_layout=True)
        return mapping[0].cuda().contiguous(), replicas[0].cuda().contiguous(), 32 + redundancy

    def hall(counts, mapping, replicas, slots):
        buckets = [0] * 256
        for expert, count in enumerate(counts):
            mask = 0
            for physical in mapping[expert, : int(replicas[expert])].cpu().tolist():
                mask |= 1 << (physical // slots)
            buckets[mask] += (count + 127) // 128
        for bit in range(8):
            for subset in range(256):
                if subset & (1 << bit):
                    buckets[subset] += buckets[subset ^ (1 << bit)]
        return max((buckets[subset] + subset.bit_count() - 1) // subset.bit_count() for subset in range(1, 256))

    cases = [[0] * 256, [1] + [0] * 255, [127] + [0] * 255, [128] + [0] * 255, [129] + [0] * 255]
    cases.extend([[generator.randrange(0, 512) for _ in range(256)] for _ in range(10)])
    checks = 0
    for redundancy in (0, 1, 2):
        mapping, replicas, slots = make_layout(redundancy)
        for counts in cases:
            gathered = torch.zeros((8, 256), dtype=torch.int32, device="cuda")
            gathered[0].copy_(torch.tensor(counts, dtype=torch.int32, device="cuda"))
            quota, status, maximum, _ = module.solve(gathered, mapping, replicas, 8, slots)
            assert int(status.cpu()) == 0
            assert int(maximum.cpu()) == hall(counts, mapping, replicas, slots)
            assert torch.equal(quota.sum(1).cpu(), torch.tensor([(value + 127) // 128 for value in counts]))
            checks += 1
    assert checks == 45
    if previous is None:
        os.environ.pop(monkeypatch_env)
    else:
        os.environ[monkeypatch_env] = previous
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
