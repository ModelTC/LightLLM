from contextlib import nullcontext

import pytest
import torch
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
from lightllm.common.basemodel.triton_kernel.fused_moe import topk_select

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("route", ["global", "grouped", "custom"])
@pytest.mark.parametrize("phase", [None, AutotuneKernelType.GENERAL, AutotuneKernelType.DECODE_ATTENTION])
def test_plain_moe_preserves_warmup_routing(monkeypatch, route, phase):
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    batch, experts, top_k = 7, 8, 2
    logits = torch.arange(experts, device="cuda", dtype=torch.float32).expand(batch, -1).contiguous()
    hidden = torch.ones((batch, 16), device="cuda", dtype=torch.bfloat16)

    def custom_routing(**kwargs):
        scores = kwargs["gating_output"].softmax(-1)
        return torch.topk(scores, k=kwargs["topk"], dim=1)

    kwargs = dict(
        hidden_states=hidden,
        router_logits=logits,
        correction_bias=None,
        top_k=top_k,
        use_grouped_topk=route == "grouped",
        renormalize=False,
        topk_group=1,
        num_expert_group=2,
        custom_routing_function=custom_routing if route == "custom" else None,
    )
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        expected_weights, original_ids = topk_select.select_experts(**kwargs)
    context = Autotuner.autotune_warmup(phase) if phase is not None else nullcontext()
    with context:
        weights, ids = topk_select.select_experts(**kwargs)
    torch.testing.assert_close(weights, expected_weights)
    if phase == AutotuneKernelType.GENERAL:
        generator = torch.Generator(device="cuda").manual_seed(batch)
        reference_logits = torch.randn((batch, experts), generator=generator, device="cuda", dtype=torch.float32)
        expected_ids = reference_logits.topk(top_k, dim=1).indices
        assert not torch.equal(expected_ids, original_ids.long())
    else:
        expected_ids = original_ids.long()
    torch.testing.assert_close(ids.long(), expected_ids)
