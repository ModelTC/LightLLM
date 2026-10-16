"""Check exact EPLB expert selection and normal FP32/BF16 rounding bounds."""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _routed_expert_outputs(hidden, physical_ids, w13, w2):
    """Keep the same BF16 expert MLP when physical expert rows move."""
    selected_w13 = w13[physical_ids.long()].float()
    gate_up = torch.einsum("tkih,th->tki", selected_w13, hidden.float()).to(torch.bfloat16)
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    return torch.einsum("tkhi,tki->tkh", w2[physical_ids.long()].float(), intermediate.float()).to(torch.bfloat16)


def _gather_by_placement(expert_outputs, weights, physical_ids, capacity):
    from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import ep_gather_chunk

    tokens, topk, hidden = expert_outputs.shape
    partials = []
    for rank in range(2):
        routes = torch.nonzero((physical_ids // capacity).flatten() == rank).flatten()
        routes = routes[physical_ids.flatten()[routes].argsort(stable=True)]
        metadata = torch.full((tokens, topk + 2), -1, dtype=torch.int32, device="cuda")
        metadata[routes // topk, routes % topk + 2] = torch.arange(routes.numel(), dtype=torch.int32, device="cuda")
        partial = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device="cuda")
        if routes.numel():
            ep_gather_chunk(
                expert_outputs.reshape(-1, hidden)[routes].contiguous(),
                0,
                weights.flatten()[routes].contiguous(),
                metadata,
                partial,
            )
        partials.append(partial)
    # Isolate placement-dependent local gather; transport preserves these values.
    return torch.stack(partials).float().sum(0).to(torch.bfloat16)


@pytest.mark.parametrize(
    "experts,groups,group_topk,grouped", [(288, 1, 1, True), (256, 8, 4, True), (128, 1, 1, False)]
)
@pytest.mark.parametrize("tokens", [1, 8, 32])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("migrated", [False, True])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_eplb_preserves_routing_and_moe_output(
    experts, groups, group_topk, grouped, tokens, tied, migrated, is_prefill
):
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
        build_initial_redundant_expert_ids,
        build_logical_to_physical_map,
        expand_redundant_placement,
    )
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import EPLBState
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM

    generator = torch.Generator(device="cuda").manual_seed(20261022)
    hidden = torch.randn((tokens, 32), device="cuda", generator=generator).to(torch.bfloat16)
    logits = torch.randn((tokens, experts), device="cuda", generator=generator)
    bias = (torch.arange(experts, device="cuda") % 11 / 64).to(torch.bfloat16) if grouped else None
    if tied:
        logits.zero_()
        if bias is not None:
            bias.zero_()
    w13 = (torch.randn((experts, 64, 32), device="cuda", generator=generator) / 8).to(torch.bfloat16)
    w2 = (torch.randn((experts, 32, 32), device="cuda", generator=generator) / 8).to(torch.bfloat16)
    moe = FuseMoeDeepGEMM.__new__(FuseMoeDeepGEMM)
    moe.eplb = None
    moe.routed_scaling_factor = 2.5
    arguments = dict(
        input_tensor=hidden,
        router_logits=logits,
        correction_bias=bias,
        top_k=8,
        renormalize=True,
        use_grouped_topk=grouped,
        topk_group=group_topk,
        num_expert_group=groups,
        scoring_func="sigmoid" if grouped else "softmax",
        is_prefill=is_prefill,
        preserve_logical_ids=True,
    )
    off_weights, off_physical, off_logical = moe._select_experts(**arguments)
    off_experts = _routed_expert_outputs(hidden, off_physical, w13, w2)
    off_output = _gather_by_placement(off_experts, off_weights, off_physical, experts // 2)

    placement = expand_redundant_placement(build_initial_redundant_expert_ids(experts, 2, 2), experts)
    if migrated:
        placement = placement.flip(0).roll(placement.shape[1] // 2, dims=1)
    mapping, counts = build_logical_to_physical_map(placement, experts, source_rank=0, node_world_size=2)
    moe.eplb = EPLBState(
        num_redundant_experts_per_rank=2,
        logical_to_physical_map=mapping.cuda(),
        logical_replica_count=counts.cuda(),
        route_counter=torch.zeros((1, experts), dtype=torch.int64, device="cuda"),
    )
    on_weights, on_physical, on_logical = moe._select_experts(**arguments)
    torch.testing.assert_close(on_logical.long(), off_logical.long(), rtol=0, atol=0)
    # Different FP32 reduction orders are valid; verify the actual weight formula
    # independently instead of requiring bitwise-identical normalized weights.
    scores = torch.sigmoid(logits.double()) if grouped else logits.double().softmax(dim=-1)
    reference_weights = scores.gather(1, off_logical.long())
    reference_weights = reference_weights / reference_weights.sum(dim=-1, keepdim=True) * moe.routed_scaling_factor
    unit_fp32 = torch.finfo(torch.float32).eps / 2
    rounding_steps = arguments["top_k"] + 4
    weight_rtol = rounding_steps * unit_fp32 / (1 - rounding_steps * unit_fp32)
    for weights in (off_weights, on_weights):
        torch.testing.assert_close(weights.double(), reference_weights, rtol=weight_rtol, atol=0)
    rows = placement.flatten().cuda()
    torch.testing.assert_close(rows[on_physical.long()], off_logical.long(), rtol=0, atol=0)
    on_experts = _routed_expert_outputs(hidden, on_physical, w13[rows], w2[rows])
    torch.testing.assert_close(on_experts, off_experts, rtol=0, atol=0)
    on_output = _gather_by_placement(on_experts, on_weights, on_physical, placement.shape[1])

    contributions = off_experts.double() * reference_weights.unsqueeze(-1)
    reference = contributions.sum(dim=1)
    magnitude = contributions.abs().sum(dim=1)
    # One BF16 local cast and one final cast, plus FP32 routing/product/sum rounding.
    unit_bf16 = torch.finfo(torch.bfloat16).eps / 2
    rounding_bound = 2 * unit_bf16 / (1 - 2 * unit_bf16) + 10 * unit_fp32 / (1 - 10 * unit_fp32)
    bound = (rounding_bound + (1 + rounding_bound) * weight_rtol) * magnitude
    for output in (off_output, on_output):
        assert torch.all((output.double() - reference).abs() <= bound), "MoE output exceeds BF16 rounding bound"
    assert torch.all((on_output.double() - off_output.double()).abs() <= 2 * bound)
