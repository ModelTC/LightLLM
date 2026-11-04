import torch


class MoeRouter:
    def __init__(self, offer, device, correction_bias=None, expert_scale=None):
        self.offer, self.spec, self.device = offer, offer.spec, device
        self.correction_bias, self.expert_scale = correction_bias, expert_scale

    def run(self, logits, shared_gate=None, capture=None):
        spec = self.spec
        if (
            logits.ndim != 2
            or logits.shape[1] != spec.experts
            or logits.device != self.device
            or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError("MoE router logits do not match the bound experts")
        if spec.gated_shared:
            if shared_gate is None or shared_gate.shape != (logits.shape[0], self.offer.fused_shared_experts):
                raise ValueError("MoE requires one shared gate logit per token and shared expert")
            if shared_gate.device != logits.device or shared_gate.dtype != spec.dtype:
                raise ValueError("MoE shared gate dtype/device disagrees with the input")
        elif shared_gate is not None:
            raise ValueError("MoE model did not declare a shared gate")
        if logits.shape[0] == 0:
            count = spec.top_k + self.offer.fused_shared_experts
            ids = logits.new_empty((0, count), dtype=torch.int64)
            if capture is not None:
                capture(ids[:, : spec.top_k])
            dtype = spec.dtype if spec.round_intermediates else torch.float32
            return logits.new_empty((0, count), dtype=dtype), ids
        if spec.scoring == "topk_softmax":
            selected, ids = logits.topk(spec.top_k, dim=-1)
            weights = selected.softmax(dim=-1)
        elif spec.groups == 1 and spec.scoring == "softmax" and self.correction_bias is None:
            from lightllm.common.basemodel.triton_kernel.fused_moe.topk_select import fused_topk

            weights, ids = fused_topk(logits, logits, spec.top_k, spec.renormalize)
        else:
            from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import triton_grouped_topk

            weights, ids = triton_grouped_topk(
                logits,
                logits,
                self.correction_bias,
                spec.top_k,
                spec.renormalize,
                spec.groups,
                spec.selected_groups,
                spec.scoring,
                spec.group_score_topk,
            )
        # Autotuning measures a spread of expert loads rather than an accidental hot expert.
        from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType

        if Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL):
            generator = torch.Generator(device=logits.device).manual_seed(logits.shape[0])
            random_logits = torch.randn(logits.shape, generator=generator, dtype=torch.float32, device=logits.device)
            ids = random_logits.topk(spec.top_k, dim=1).indices
        if capture is not None:
            capture(ids)
        if spec.routed_scale != 1:
            weights.mul_(spec.routed_scale)
        if self.expert_scale is not None:
            weights = weights * self.expert_scale[ids.long()].to(weights.dtype)
        if spec.round_intermediates:
            weights = weights.to(spec.dtype)
        if self.offer.fused_shared_experts:
            from lightllm.common.basemodel.triton_kernel.fused_moe.append_shared_expert_topk import (
                append_fused_shared_experts,
            )

            weights, ids = append_fused_shared_experts(
                weights, ids, spec.experts, self.offer.fused_shared_experts, shared_gate
            )
        return weights, ids
