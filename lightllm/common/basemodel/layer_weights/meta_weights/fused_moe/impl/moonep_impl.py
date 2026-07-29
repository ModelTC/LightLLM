from typing import Optional

import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.triton_impl import (
    FuseMoeTriton,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import (
    silu_and_mul_fwd,
)
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.distributed import dist_group_manager


class FuseMoeMoonEP(FuseMoeTriton):
    def _pad_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        capacity: int,
    ):
        num_tokens = hidden_states.shape[0]
        if num_tokens > capacity:
            raise ValueError(
                f"MoonEP received {num_tokens} tokens, exceeding its configured capacity {capacity}. "
                "Increase NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL or "
                "NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE."
            )
        if num_tokens == capacity:
            return (
                hidden_states,
                topk_weights.float().contiguous(),
                topk_ids.int().contiguous(),
            )

        padding = capacity - num_tokens
        padded_hidden = torch.cat(
            (hidden_states, hidden_states.new_zeros((padding, hidden_states.shape[1]))),
            dim=0,
        )
        padded_weights = torch.cat(
            (
                topk_weights.float(),
                torch.zeros(
                    (padding, topk_weights.shape[1]),
                    dtype=torch.float32,
                    device=topk_weights.device,
                ),
            ),
            dim=0,
        )
        dummy_ids = torch.arange(
            topk_ids.shape[1],
            dtype=torch.int32,
            device=topk_ids.device,
        ).expand(padding, -1)
        padded_ids = torch.cat((topk_ids.int(), dummy_ids), dim=0)
        return (
            padded_hidden.contiguous(),
            padded_weights.contiguous(),
            padded_ids.contiguous(),
        )

    @staticmethod
    def _vm_group_indices(cu_seqlens: torch.Tensor, num_rows: int, num_experts: int):
        row_ids = torch.arange(num_rows, dtype=torch.int32, device=cu_seqlens.device)
        groups = torch.searchsorted(cu_seqlens, row_ids, right=True, out_int32=True)
        source_groups = torch.where(groups < num_experts, groups, -1)
        slot_groups = torch.where(
            (groups >= num_experts) & (groups < cu_seqlens.numel()),
            groups - num_experts,
            -1,
        )
        return source_groups.contiguous(), slot_groups.contiguous()

    @staticmethod
    def _grouped_gemm(
        hidden_states: torch.Tensor,
        source_weight: torch.Tensor,
        slot_weight: torch.Tensor,
        source_groups: torch.Tensor,
        slot_groups: torch.Tensor,
        output: torch.Tensor,
    ):
        try:
            import deep_gemm
        except ImportError as exc:
            raise RuntimeError("MoonEP requires DeepGEMM for its VM grouped GEMMs") from exc

        safe_source_groups = source_groups.clamp_min(0)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            hidden_states,
            source_weight,
            output,
            safe_source_groups,
        )
        output.masked_fill_(source_groups[:, None] < 0, 0)
        safe_slot_groups = slot_groups.clamp_min(0)
        slot_output = torch.empty_like(output)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            hidden_states,
            slot_weight,
            slot_output,
            safe_slot_groups,
        )
        slot_output.masked_fill_(slot_groups[:, None] < 0, 0)
        output.add_(slot_output)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        correction_bias: Optional[torch.Tensor],
        scoring_func: str,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        is_prefill: Optional[bool] = None,
        per_expert_scale: Optional[torch.Tensor] = None,
        shared_expert_gate: Optional[torch.Tensor] = None,
    ):
        if is_prefill is None:
            raise ValueError("MoonEP requires the caller to identify prefill versus decode")
        if input_tensor.dtype != torch.bfloat16:
            raise ValueError(f"MoonEP requires BF16 hidden states, got {input_tensor.dtype}")
        if shared_expert_gate is not None:
            raise ValueError("MoonEP does not support a fused shared-expert gate")

        topk_weights, topk_ids = self._select_experts(
            input_tensor=input_tensor,
            router_logits=router_logits,
            correction_bias=correction_bias,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            per_expert_scale=per_expert_scale,
        )

        buffer = dist_group_manager.get_moonep_buffer(is_prefill=bool(is_prefill))
        capacity = int(buffer._ctx["S"])
        hidden, route_weights, expert_ids = self._pad_inputs(
            input_tensor,
            topk_weights,
            topk_ids,
            capacity,
        )
        tokens_per_expert = torch.bincount(
            expert_ids.reshape(-1).long(),
            minlength=self.n_routed_experts,
        ).int()
        dispatched, dispatched_weights, cu_seqlens, plan = buffer.dispatch(
            hidden,
            route_weights,
            expert_ids,
            tokens_per_expert,
            zero_copy=True,
        )

        from moonep.prefetch import launch_prefetch

        experts_to_copy = plan.experts_to_copy[self.global_rank_]
        num_sms = int(buffer._ctx["num_sms"])
        launch_prefetch(
            w13.weight,
            w13.moonep_prefetch_slots,
            experts_to_copy,
            num_sms=num_sms,
        )
        launch_prefetch(
            w2.weight,
            w2.moonep_prefetch_slots,
            experts_to_copy,
            num_sms=num_sms,
        )

        source_groups, slot_groups = self._vm_group_indices(
            cu_seqlens,
            dispatched.shape[0],
            self.n_routed_experts,
        )
        intermediate_size = w2.weight.shape[-1]
        gate_up = torch.empty(
            (dispatched.shape[0], 2 * intermediate_size),
            dtype=torch.bfloat16,
            device=dispatched.device,
        )
        self._grouped_gemm(
            dispatched,
            w13.weight,
            w13.moonep_prefetch_slots,
            source_groups,
            slot_groups,
            gate_up,
        )
        activated = torch.empty(
            (dispatched.shape[0], intermediate_size),
            dtype=torch.bfloat16,
            device=dispatched.device,
        )
        silu_and_mul_fwd(
            gate_up,
            activated,
            activation=getattr(self, "activation", "silu"),
            activation_situ_beta=getattr(self, "activation_situ_beta", None),
            activation_situ_linear_beta=getattr(self, "activation_situ_linear_beta", None),
        )
        self._grouped_gemm(
            activated,
            w2.weight,
            w2.moonep_prefetch_slots,
            source_groups,
            slot_groups,
            dispatched,
        )
        dispatched.mul_(dispatched_weights[:, None])
        combined, _, _ = buffer.combine(
            plan=plan,
            hidden_nvsh=dispatched,
            zero_copy=True,
        )
        return combined[: input_tensor.shape[0]]
