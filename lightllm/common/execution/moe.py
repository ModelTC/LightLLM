"""MoE routing and expert execution over bound native tensors."""

import torch

from .specs import MoeOffer, ProbeResult


class MoeBackend:
    name = "triton.moe"

    def probe(self, spec, format, device, *, fused_shared_experts=0):
        if device.device_type != "cuda" or spec.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return ProbeResult(reason="Triton MoE requires CUDA floating-point inputs")
        if format.encoding not in ("plain", "fp8_block128"):
            return ProbeResult(reason="unsupported MoE weight encoding")
        quantized = format.encoding == "fp8_block128"
        fields = ("gate_up", "down", "gate_up_scale", "down_scale") if quantized else ("gate_up", "down")
        if (
            format.layout not in ("expert_output_input", "expert_output_input_interleaved")
            or format.fields not in (fields, fields + ("gate_up_bias", "down_bias"))
            or format.version != 1
        ):
            return ProbeResult(reason="MoE needs explicit expert/output/input weight fields")
        if spec.intermediate_size % format.shard[1]:
            return ProbeResult(reason="MoE intermediate dimension needs complete TP shards")
        local_width = spec.intermediate_size // format.shard[1]
        if format.dtype != (torch.float8_e4m3fn if quantized else spec.dtype):
            return ProbeResult(reason="MoE weight dtype disagrees with encoding")
        if format.group_size != (128 if quantized else None) or format.scale_dtype != (
            torch.float32 if quantized else None
        ):
            return ProbeResult(reason="MoE scale format disagrees with encoding")
        if quantized and (
            device.capability is None
            or device.capability < (8, 9)
            or spec.dtype == torch.float32
            or spec.hidden_size % 128
            or local_width % 128
        ):
            return ProbeResult(reason="FP8 MoE needs SM89+, FP16/BF16 and complete 128-element groups")
        if spec.group_score_topk not in (1, 2) or spec.groups & (spec.groups - 1):
            return ProbeResult(reason="MoE grouped routing needs power-of-two groups and top-1/top-2 scores")
        if not 0 <= fused_shared_experts <= spec.shared_experts:
            return ProbeResult(reason="fused shared experts must be declared by the model")
        if spec.gated_shared and fused_shared_experts != spec.shared_experts:
            return ProbeResult(reason="gated shared experts require their fused weight views")
        return ProbeResult(offers=(MoeOffer(self.name, spec, format, fused_shared_experts),))

    def bind(
        self,
        offer,
        gate_up,
        down,
        *,
        gate_up_scale=None,
        down_scale=None,
        correction_bias=None,
        expert_scale=None,
        gate_up_bias=None,
        down_bias=None,
    ):
        if offer.backend != self.name:
            raise ValueError("MoE offer belongs to another backend")
        spec, format = offer.spec, offer.format
        from .specs import DeviceCaps

        device = gate_up.device
        caps = DeviceCaps(
            device.type, device.index, torch.cuda.get_device_capability(device) if gate_up.is_cuda else None
        )
        checked = self.probe(spec, format, caps, fused_shared_experts=offer.fused_shared_experts)
        if not checked.offers:
            raise ValueError(checked.reason)
        count = spec.experts + offer.fused_shared_experts
        width = spec.intermediate_size // format.shard[1]
        for tensor, shape in (
            (gate_up, (count, 2 * width, spec.hidden_size)),
            (down, (count, spec.hidden_size, width)),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != format.dtype
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError("MoE native weight view does not match the selected format")
        for scale, weight in ((gate_up_scale, gate_up), (down_scale, down)):
            if format.encoding == "plain":
                if scale is not None:
                    raise ValueError("plain MoE must not have quantization scales")
            elif (
                scale is None
                or scale.shape != (count, weight.shape[1] // 128, weight.shape[2] // 128)
                or scale.dtype != torch.float32
                or scale.device != device
                or not scale.is_contiguous()
            ):
                raise ValueError("MoE block scale view does not match the weight")
        for tensor in (correction_bias, expert_scale):
            if tensor is not None and (
                tensor.shape != (spec.experts,)
                or tensor.device != device
                or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or not tensor.is_contiguous()
            ):
                raise ValueError("MoE routing parameter must have one floating-point value per routed expert")
        if spec.scoring == "topk_softmax" and correction_bias is not None:
            raise ValueError("top-k softmax does not support correction bias")
        biased = "gate_up_bias" in format.fields
        for tensor, shape in ((gate_up_bias, (count, 2 * width)), (down_bias, (count, spec.hidden_size))):
            if not biased:
                if tensor is not None:
                    raise ValueError("expert bias was not declared by the weight format")
            elif (
                tensor is None
                or tensor.shape != shape
                or tensor.dtype not in (spec.dtype, torch.float32)
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError("expert bias does not match the bound weight format")
        return BoundMoe(
            offer, gate_up, down, gate_up_scale, down_scale, correction_bias, expert_scale, gate_up_bias, down_bias
        )


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


class BoundMoe:
    def __init__(
        self,
        offer,
        gate_up,
        down,
        gate_up_scale,
        down_scale,
        correction_bias,
        expert_scale,
        gate_up_bias=None,
        down_bias=None,
    ):
        self.offer, self.spec = offer, offer.spec
        self.gate_up, self.down = gate_up, down
        self.gate_up_scale, self.down_scale = gate_up_scale, down_scale
        self.correction_bias, self.expert_scale = correction_bias, expert_scale
        self.gate_up_bias, self.down_bias = gate_up_bias, down_bias
        self.layout = "interleaved" if offer.format.layout.endswith("_interleaved") else "blocked"
        self.router = MoeRouter(offer, gate_up.device, correction_bias, expert_scale)

    @staticmethod
    def _quantize(inputs, group_size, dtype):
        from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
            lightllm_per_token_group_quant_fp8,
        )

        encoded = torch.empty_like(inputs, dtype=dtype)
        scale = torch.empty((inputs.shape[0], inputs.shape[1] // group_size), dtype=torch.float32, device=inputs.device)
        lightllm_per_token_group_quant_fp8(inputs, group_size, encoded, scale)
        return encoded, scale

    def route(self, logits, shared_gate=None, capture=None):
        return self.router.run(logits, shared_gate, capture)

    def run(self, inputs, router_logits, *, shared_gate=None, capture=None):
        if (
            inputs.ndim != 2
            or router_logits.ndim != 2
            or inputs.shape != (router_logits.shape[0], self.spec.hidden_size)
            or inputs.dtype != self.spec.dtype
            or inputs.device != self.gate_up.device
            or not inputs.is_contiguous()
        ):
            raise ValueError("MoE input does not match the bound execution")
        weights, ids = self.route(router_logits, shared_gate, capture)
        if not inputs.shape[0]:
            return inputs
        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import fused_experts_impl

        return fused_experts_impl(
            inputs,
            self.gate_up,
            self.down,
            weights,
            ids,
            inplace=True,
            use_fp8_w8a8=self.offer.format.encoding == "fp8_block128",
            w1_scale=self.gate_up_scale,
            w2_scale=self.down_scale,
            w1_bias=self.gate_up_bias,
            w2_bias=self.down_bias,
            layout=self.layout,
            alpha=self.spec.alpha if self.spec.limit is not None else None,
            limit=self.spec.limit,
            clamp_up_add_one=self.spec.clamp_up_add_one,
            use_tanh_approximate_gelu=False,
            quantize_input=self._quantize,
            round_intermediates=self.spec.round_intermediates,
            routed_top_k=self.spec.top_k if self.spec.round_intermediates and self.offer.fused_shared_experts else None,
        )
