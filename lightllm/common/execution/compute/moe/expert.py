"""Routed expert math over an EP shard; communication is owned by Transfer."""

from dataclasses import replace

import torch

from lightllm.common.execution.compute.moe.backend import MoeBackend
from lightllm.common.execution.compute.moe.routing import MoeRouter
from lightllm.common.execution.contracts.device import DeviceCaps
from lightllm.common.execution.contracts.provider import MoeOffer, ProbeResult


class ExpertComputeBackend:
    name = "deepgemm.moe.ep"

    def probe(self, spec, format, device):
        if device.capability != (9, 0) or spec.dtype != torch.bfloat16:
            return ProbeResult(reason="native DeepGEMM EP requires Hopper and BF16 activations")
        if format.layout != "local_expert_output_input" or spec.experts % format.shard[1]:
            return ProbeResult(reason="EP ranks must own complete expert shards")
        if format.encoding != "fp8_block128":
            return ProbeResult(reason="native DeepGEMM EP requires block128 FP8 weights")
        # Shared experts stay in the model recipe, outside routed EP execution.
        routed = replace(spec, shared_experts=0, gated_shared=False)
        checked = MoeBackend().probe(routed, replace(format, layout="expert_output_input", shard=(0, 1)), device)
        if not checked.offers:
            return checked
        return ProbeResult(offers=(MoeOffer(self.name, routed, format),))

    def bind(
        self,
        offer,
        gate_up,
        down,
        *,
        gate_up_scale,
        down_scale,
        workspaces,
        correction_bias=None,
        expert_scale=None,
        redundancy_ids=None,
        expert_counter=None,
        count_routes=False,
        round_scales=False,
    ):
        if offer.backend != self.name:
            raise ValueError("expert offer belongs to another backend")
        spec, fmt, device = offer.spec, offer.format, gate_up.device
        caps = DeviceCaps(
            device.type, device.index, torch.cuda.get_device_capability(device) if gate_up.is_cuda else None
        )
        result = self.probe(spec, fmt, caps)
        if not result.offers:
            raise ValueError(result.reason)
        import deep_gemm

        for names in (
            ("m_grouped_gemm_fp8_fp8_bf16_nt_contiguous", "m_grouped_fp8_gemm_nt_contiguous"),
            ("m_grouped_gemm_fp8_fp8_bf16_nt_masked", "m_grouped_fp8_gemm_nt_masked"),
        ):
            if not any(hasattr(deep_gemm, name) for name in names):
                raise ValueError(f"DeepGEMM lacks expert execution API: {names}")
        replicas = 0 if redundancy_ids is None else redundancy_ids.numel()
        count = spec.experts // fmt.shard[1] + replicas
        for tensor, shape in (
            (gate_up, (count, 2 * spec.intermediate_size, spec.hidden_size)),
            (down, (count, spec.hidden_size, spec.intermediate_size)),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != fmt.dtype
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError("EP weight view disagrees with local expert placement")
        for scale, weight in ((gate_up_scale, gate_up), (down_scale, down)):
            if (
                scale is None
                or scale.shape != (count, weight.shape[1] // 128, weight.shape[2] // 128)
                or scale.dtype != torch.float32
                or scale.device != device
            ):
                raise ValueError("EP block scales disagree with the native weight view")
        for tensor in (correction_bias, expert_scale):
            if tensor is not None and (
                tensor.shape != (spec.experts,)
                or tensor.device != device
                or not tensor.is_floating_point()
                or not tensor.is_contiguous()
            ):
                raise ValueError("EP routing parameters need one value per logical expert")
        if replicas:
            if (
                redundancy_ids.dtype != torch.int64
                or redundancy_ids.device != device
                or not redundancy_ids.is_contiguous()
            ):
                raise ValueError("EP redundancy mapping must be a contiguous device int64 view")
            if (
                expert_counter is None
                or expert_counter.shape != (spec.experts,)
                or expert_counter.dtype != torch.int64
                or expert_counter.device != device
            ):
                raise ValueError("EP redundancy requires a device expert counter")
        workspaces = tuple(workspaces)
        if not workspaces or any(
            w.dtype != torch.uint8 or w.ndim != 1 or w.device != device or not w.is_contiguous() or not w.numel()
            for w in workspaces
        ):
            raise ValueError("EP requires bound byte workspace for each execution lane")
        spans = sorted((w.data_ptr(), w.data_ptr() + w.numel()) for w in workspaces)
        if any(end > start for (_, end), (start, _) in zip(spans, spans[1:])):
            raise ValueError("concurrent EP lanes must not share workspace")
        return BoundExpertCompute(
            offer,
            gate_up,
            down,
            gate_up_scale,
            down_scale,
            correction_bias,
            expert_scale,
            workspaces,
            redundancy_ids,
            expert_counter,
            count_routes,
            round_scales,
        )


class BoundExpertCompute:
    def __init__(
        self,
        offer,
        gate_up,
        down,
        gate_up_scale,
        down_scale,
        correction_bias,
        expert_scale,
        workspaces,
        redundancy_ids,
        expert_counter,
        count_routes,
        round_scales,
    ):
        self.offer, self.spec = offer, offer.spec
        self.gate_up, self.down = gate_up, down
        self.gate_up_scale, self.down_scale = gate_up_scale, down_scale
        self.router = MoeRouter(offer, gate_up.device, correction_bias, expert_scale)
        self.workspaces = workspaces
        self.redundancy_ids, self.expert_counter = redundancy_ids, expert_counter
        self.count_routes, self.round_scales = count_routes, round_scales
        self.rank, self.world_size = offer.format.shard
        self.local_experts = gate_up.shape[0]
        self.total_experts = self.local_experts * self.world_size

    def route(self, logits, shared_gate=None, capture=None):
        weights, ids = self.router.run(logits, shared_gate, capture)
        if self.redundancy_ids is not None and self.redundancy_ids.numel():
            from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import redundancy_topk_ids_repair

            # Captured route IDs describe logical experts, before physical replicas.
            ids = ids.clone()
            redundancy_topk_ids_repair(
                ids,
                self.redundancy_ids,
                self.spec.experts // self.world_size,
                self.rank,
                self.expert_counter,
                self.count_routes,
            )
        # DeepEP requires FP32 weights; preserve any model-declared BF16 rounding.
        return weights.float(), ids.long()

    def quantize(self, inputs):
        from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8

        return per_token_group_quant_fp8(
            inputs, 128, dtype=self.gate_up.dtype, use_ue8m0_scales=self.round_scales, use_triton=True
        )

    def masked(self, inputs, counts, expected_m):
        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import masked_group_gemm

        return masked_group_gemm(
            inputs,
            counts,
            self.spec.dtype,
            self.gate_up,
            self.gate_up_scale,
            self.down,
            self.down_scale,
            expected_m,
            alpha=self.spec.alpha if self.spec.limit is not None else None,
            limit=self.spec.limit,
            clamp_up_add_one=self.spec.clamp_up_add_one,
            use_ue8m0_scales=self.round_scales,
            use_tanh_approximate_gelu=False,
            round_intermediates=self.spec.round_intermediates,
        )

    def expanded(self, counts, unaligned_counts, source_metadata, inputs, weights, lane=0):
        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import chunked_expanded_moe_forward

        if not 0 <= lane < len(self.workspaces):
            raise ValueError("EP execution lane has no bound workspace")
        if sum(counts):
            return chunked_expanded_moe_forward(
                counts,
                unaligned_counts,
                inputs,
                weights,
                source_metadata,
                self.gate_up,
                self.gate_up_scale,
                self.down,
                self.down_scale,
                128,
                self.workspaces[lane],
                self.spec.dtype,
                alpha=self.spec.alpha if self.spec.limit is not None else None,
                limit=self.spec.limit,
                clamp_up_add_one=self.spec.clamp_up_add_one,
                use_ue8m0_scales=self.round_scales,
                use_tanh_approximate_gelu=False,
                use_triton_quant=True,
                round_intermediates=self.spec.round_intermediates,
            )
        from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType

        if Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL):
            from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd

            branch = self.down.new_zeros((1, 2 * self.spec.intermediate_size), dtype=self.spec.dtype)
            silu_and_mul_fwd(
                branch,
                torch.empty_like(branch[:, : self.spec.intermediate_size]),
                alpha=self.spec.alpha if self.spec.limit is not None else None,
                limit=self.spec.limit,
                clamp_up_add_one=self.spec.clamp_up_add_one,
                use_tanh_approximate_gelu=False,
                round_intermediates=self.spec.round_intermediates,
            )
        shape = (source_metadata.shape[0], self.spec.hidden_size)
        if self.spec.round_intermediates:
            return self.down.new_zeros((self.spec.top_k, *shape), dtype=self.spec.dtype)
        return self.down.new_empty(shape, dtype=self.spec.dtype)

    def reduce(self, contributions):
        return torch.stack(contributions).sum(dim=0, dtype=torch.float32).to(self.spec.dtype)
