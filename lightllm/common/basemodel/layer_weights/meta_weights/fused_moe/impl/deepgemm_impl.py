import torch
from typing import Optional, Tuple, Any
from .triton_impl import FuseMoeTriton
from lightllm.distributed import dist_group_manager
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import (
    fused_experts,
    get_ep_num_sms,
    masked_group_gemm,
    chunked_expanded_moe_forward,
    legacy_normal_moe_forward,
    quantize_fused_experts_input,
    use_sm90_mega_moe,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import redundancy_topk_ids_repair


class FuseMoeDeepGEMM(FuseMoeTriton):
    def _select_experts(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: Optional[torch.Tensor],
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        scoring_func: str,
        per_expert_scale: Optional[torch.Tensor] = None,
        shared_expert_gate: Optional[torch.Tensor] = None,
    ):
        """Select experts and return topk weights and ids."""
        assert shared_expert_gate is None, "fused shared expert as MoE is not supported by DeepGEMM fused MoE"
        from lightllm.common.basemodel.triton_kernel.fused_moe.topk_select import select_experts

        topk_weights, topk_ids = select_experts(
            hidden_states=input_tensor,
            router_logits=router_logits,
            correction_bias=correction_bias,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
        )
        if self.routed_scaling_factor != 1.0:
            topk_weights.mul_(self.routed_scaling_factor)
        if per_expert_scale is not None:
            topk_weights = topk_weights * per_expert_scale[topk_ids.to(torch.long)].to(topk_weights.dtype)
        origin_topk_ids = topk_ids
        if self.redundancy_expert_num > 0:
            # 因为 redundancy_topk_ids_repair 会修改 topk_ids，所以需要先复制一份
            origin_topk_ids = topk_ids.clone()
            redundancy_topk_ids_repair(
                topk_ids=topk_ids,
                redundancy_expert_ids=self.redundancy_expert_ids_tensor,
                ep_expert_num=self.ep_n_routed_experts,
                global_rank=self.global_rank_,
                expert_counter=self.routed_expert_counter_tensor,
                enable_counter=self.auto_update_redundancy_expert,
            )
        return topk_weights, topk_ids, origin_topk_ids

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = None,
    ):
        fused_topk_ids = topk_ids if use_sm90_mega_moe(self.quant_method) else topk_ids.to(torch.long)
        output = fused_experts(
            hidden_states=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_idx=fused_topk_ids,
            num_experts=self.total_expert_num_contain_redundancy,  # number of all experts contain redundancy
            quant_method=self.quant_method,
            is_prefill=is_prefill,
            previous_event=None,  # for overlap
            swiglu_limit=self.swiglu_limit,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_clamp_up_add_one=self.swiglu_clamp_up_add_one,
        )
        return output

    def low_latency_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        use_grouped_topk: bool,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        topk_group: int,
        n_group: int,
        scoring_func: str,
    ):
        topk_weights, topk_idx, _ = self._select_experts(
            input_tensor=hidden_states,
            router_logits=router_logits,
            correction_bias=e_score_correction_bias,
            use_grouped_topk=use_grouped_topk,
            top_k=num_experts_per_tok,
            renormalize=norm_topk_prob,
            topk_group=topk_group,
            num_expert_group=n_group,
            scoring_func=scoring_func,
        )

        topk_idx = topk_idx.to(torch.long)
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_decode()
        use_fp8_w8a8 = self.quant_method.method_name != "none"
        recv_x, masked_m, handle, event, hook = dist_group_manager.ep_low_latency_buffer.low_latency_dispatch(
            topk_idx=topk_idx,
            x=hidden_states,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_experts=self.total_expert_num_contain_redundancy,
            use_fp8=use_fp8_w8a8,
            async_finish=False,
            return_recv_hook=True,
        )
        return recv_x, masked_m, topk_idx, topk_weights, handle, hook

    def select_experts_and_quant_input(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        w13: WeightPack,
        use_grouped_topk: bool,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        topk_group: int,
        n_group: int,
        scoring_func: str,
    ):
        topk_weights, topk_idx, _ = self._select_experts(
            input_tensor=hidden_states,
            router_logits=router_logits,
            correction_bias=e_score_correction_bias,
            use_grouped_topk=use_grouped_topk,
            top_k=num_experts_per_tok,
            renormalize=norm_topk_prob,
            topk_group=topk_group,
            num_expert_group=n_group,
            scoring_func=scoring_func,
        )
        qinput_tensor = quantize_fused_experts_input(hidden_states, w13, self.quant_method)
        return topk_weights, topk_idx.to(torch.long), qinput_tensor

    def dispatch(
        self,
        qinput_tensor: Tuple[torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_event: Optional[Any] = None,
    ):
        buffer = dist_group_manager.ep_buffer
        if dist_group_manager.ep_prefill_uses_legacy_buffer:
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                layout_event,
            ) = buffer.get_dispatch_layout(
                topk_idx,
                self.total_expert_num_contain_redundancy,
                previous_event=overlap_event,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
            (recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, _,) = buffer.dispatch(
                qinput_tensor,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                previous_event=layout_event,
                async_finish=False,
                allocate_on_comm_stream=False,
                expert_alignment=128,
            )
            return (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                lambda: None,
            )

        num_max_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_prefill()
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
            qinput_tensor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=self.total_expert_num_contain_redundancy,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            expert_alignment=128,
            num_sms=get_ep_num_sms(),
            previous_event=overlap_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
            do_cpu_sync=True,
            do_handle_copy=False,
            do_expand=True,
            use_tma_aligned_col_major_sf=True,
        )

        def hook():
            event.current_stream_wait()

        return recv_x, recv_topk_idx, recv_topk_weights, handle.num_recv_tokens_per_expert_list, handle, hook

    def masked_group_gemm(
        self,
        recv_x: Tuple[torch.Tensor],
        w13: WeightPack,
        w2: WeightPack,
        masked_m: torch.Tensor,
        dtype: torch.dtype,
        expected_m: int,
    ):
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        return masked_group_gemm(
            recv_x,
            masked_m,
            dtype,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            expected_m=expected_m,
            swiglu_limit=self.swiglu_limit,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_clamp_up_add_one=self.swiglu_clamp_up_add_one,
        )

    def prefilled_group_gemm(
        self,
        num_recv_tokens_per_expert_list,
        num_unaligned_recv_tokens_per_expert: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        recv_x: Tuple[torch.Tensor],
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        hidden_dtype=torch.bfloat16,
        microbatch_index: int = 0,
    ):
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        if dist_group_manager.ep_prefill_uses_legacy_buffer:
            return legacy_normal_moe_forward(
                num_recv_tokens_per_expert_list,
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                w13_weight,
                w13_scale,
                w2_weight,
                w2_scale,
                hidden_dtype,
                self.swiglu_limit,
                self.swiglu_alpha,
                self.swiglu_clamp_up_add_one,
            )

        assert recv_topk_idx is None
        all_tokens = sum(num_recv_tokens_per_expert_list)
        if all_tokens > 0:
            gather_out = chunked_expanded_moe_forward(
                num_recv_tokens_per_expert_list=num_recv_tokens_per_expert_list,
                num_unaligned_recv_tokens_per_expert=num_unaligned_recv_tokens_per_expert,
                recv_x=recv_x,
                recv_topk_weights=recv_topk_weights,
                recv_src_metadata=recv_src_metadata,
                w1=w13_weight,
                w1_scale=w13_scale,
                w2=w2_weight,
                w2_scale=w2_scale,
                block_size_k=self.quant_method.block_size,
                workspace=dist_group_manager.get_deep_ep_prefill_moe_workspace(microbatch_index),
                hidden_dtype=hidden_dtype,
                swiglu_limit=self.swiglu_limit,
                swiglu_alpha=self.swiglu_alpha,
                swiglu_clamp_up_add_one=self.swiglu_clamp_up_add_one,
            )
        else:
            gather_out = torch.empty(
                (recv_src_metadata.shape[0], w2_weight.shape[1]),
                device=recv_x[0].device,
                dtype=hidden_dtype,
            )
            ######################################## warning ##################################################
            # A rank may receive no tokens during autotune warmup. Run one dummy token through
            # silu_and_mul_fwd so the empty rank matches the first kernel call made by non-empty ranks.
            # This branch does not synchronize additional calls caused by different positive chunk counts.
            if Autotuner.is_autotune_warmup():
                N = w13_weight.shape[1]
                _gemm_out_a = torch.zeros((1, N), device=recv_x[0].device, dtype=hidden_dtype)
                _silu_out = torch.zeros((1, N // 2), device=recv_x[0].device, dtype=hidden_dtype)
                silu_and_mul_fwd(
                    _gemm_out_a.view(-1, N),
                    _silu_out,
                    limit=self.swiglu_limit,
                    alpha=self.swiglu_alpha,
                    clamp_up_add_one=self.swiglu_clamp_up_add_one,
                )
                _gemm_out_a, _silu_out = None, None
        del recv_x
        return gather_out

    def low_latency_combine(
        self,
        gemm_out_b: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Any,
    ):
        combined_x, event_overlap, hook = dist_group_manager.ep_low_latency_buffer.low_latency_combine(
            gemm_out_b, topk_idx, topk_weights, handle, async_finish=False, return_recv_hook=True
        )
        return combined_x, hook

    def combine(
        self,
        gemm_out_b: torch.Tensor,
        handle: Any,
        overlap_event: Optional[Any] = None,
    ):
        # normal combine
        if dist_group_manager.ep_prefill_uses_legacy_buffer:
            combined_x, _, _ = dist_group_manager.ep_buffer.combine(
                gemm_out_b,
                handle,
                topk_weights=None,
                previous_event=overlap_event,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
            return combined_x, lambda: None

        combined_x, _, event = dist_group_manager.ep_buffer.combine(
            gemm_out_b,
            handle,
            topk_weights=None,
            num_sms=get_ep_num_sms(),
            previous_event=overlap_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )

        def hook():
            event.current_stream_wait()

        return combined_x, hook
