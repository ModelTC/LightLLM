"""MoE routing and expert execution over bound native tensors."""

import torch

from lightllm.core.contracts.provider import MoeOffer, ProbeResult
from lightllm.core.compute.moe.spec import MoeSpec
from lightllm.core.compute.moe.routing import MoeRouter
from lightllm.core.formats.weights.moe import validate_moe_weights
from lightllm.core.formats.activations.fp8 import quantize_fp8


class MoeBackend:
    name = "triton.moe"
    implementation = "triton"

    def probe(self, spec, format, device, *, fused_shared_experts=0):
        if not isinstance(spec, MoeSpec):
            return ProbeResult(reason="Triton MoE requires standard routed expert semantics")
        if not device.is_nvidia or spec.dtype not in (torch.float16, torch.bfloat16, torch.float32):
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
        from lightllm.core.contracts.device import DeviceCaps

        device = gate_up.device
        caps = DeviceCaps(
            device.type, device.index, torch.cuda.get_device_capability(device) if gate_up.is_cuda else None
        )
        checked = self.probe(spec, format, caps, fused_shared_experts=offer.fused_shared_experts)
        if not checked.offers:
            raise ValueError(checked.reason)
        validate_moe_weights(
            offer, gate_up, down, gate_up_scale, down_scale, correction_bias, expert_scale, gate_up_bias, down_bias
        )
        return BoundMoe(
            offer, gate_up, down, gate_up_scale, down_scale, correction_bias, expert_scale, gate_up_bias, down_bias
        )


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

    _quantize = staticmethod(quantize_fp8)

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
