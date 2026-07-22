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
    quantize_fused_experts_input,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import redundancy_topk_ids_repair
from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_kernels import eplb_map


class FuseMoeDeepGEMM(FuseMoeTriton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eplb_experts_logical_to_physical_map = None
        self.eplb_experts_logical_replica_count = None
        self.eplb_experts_record_load_tensor = None
        # These host-side values select ring-buffer rows.  The CUDA flag remains
        # the ordering-safe source of truth for whether the kernel records.
        self.eplb_recorded_sample_count = 0
        self.eplb_recording = False
        self.ep_balance_counters = None
        self._primary_weight_pack_cache = {}

    def configure_eplb(
        self,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_load_tensor: torch.Tensor,
    ) -> None:
        self.eplb_experts_logical_to_physical_map = logical_to_physical_map
        self.eplb_experts_logical_replica_count = logical_replica_count
        self.eplb_experts_record_load_tensor = record_load_tensor

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
        is_prefill: Optional[bool] = None,
        preserve_logical_ids: bool = False,
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
        if self.eplb_experts_logical_to_physical_map is not None:
            if is_prefill is True:
                if preserve_logical_ids:
                    origin_topk_ids = topk_ids.clone()
                sample_index = 0
                if self.eplb_recording:
                    sample_index = self.eplb_recorded_sample_count % self.routed_expert_counter_tensor.shape[0]
                    self.eplb_recorded_sample_count += 1
                eplb_map(
                    topk_ids,
                    self.eplb_experts_logical_to_physical_map,
                    self.eplb_experts_logical_replica_count,
                    self.routed_expert_counter_tensor,
                    self.eplb_experts_record_load_tensor,
                    sample_index,
                )
        elif self.redundancy_expert_num > 0:
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
        if is_prefill is False and self.eplb_experts_logical_to_physical_map is not None:
            w13 = self._primary_weight_pack(w13)
            w2 = self._primary_weight_pack(w2)
            num_experts = self.n_routed_experts
        else:
            num_experts = self.total_expert_num_contain_redundancy
        output = fused_experts(
            hidden_states=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_idx=topk_ids.to(torch.long),
            num_experts=num_experts,
            quant_method=self.quant_method,
            is_prefill=is_prefill,
            previous_event=None,  # for overlap
            ep_balance_counters=self.ep_balance_counters,
        )
        return output

    def _primary_weight_pack(self, weight_pack: WeightPack) -> WeightPack:
        """Return the cached local-primary view used by all decode paths."""
        if self.eplb_experts_logical_to_physical_map is None:
            return weight_pack
        cache = getattr(self, "_primary_weight_pack_cache", None)
        if cache is None:
            cache = self._primary_weight_pack_cache = {}
        cache_key = id(weight_pack)
        primary = cache.get(cache_key)
        if primary is None:
            primary = WeightPack(
                weight=weight_pack.weight[: self.ep_n_routed_experts],
                weight_scale=(
                    weight_pack.weight_scale[: self.ep_n_routed_experts]
                    if weight_pack.weight_scale is not None
                    else None
                ),
                weight_zero_point=(
                    getattr(weight_pack, "weight_zero_point", None)[: self.ep_n_routed_experts]
                    if getattr(weight_pack, "weight_zero_point", None) is not None
                    else None
                ),
            )
            cache[cache_key] = primary
        return primary

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
            is_prefill=False,
        )

        topk_idx = topk_idx.to(torch.long)
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_decode()
        use_fp8_w8a8 = self.quant_method.method_name != "none"
        recv_x, masked_m, handle, event, hook = dist_group_manager.ep_low_latency_buffer.low_latency_dispatch(
            topk_idx=topk_idx,
            x=hidden_states,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_experts=(
                self.n_routed_experts
                if self.eplb_experts_logical_to_physical_map is not None
                else self.total_expert_num_contain_redundancy
            ),
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
            is_prefill=True,
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

        counters = self.ep_balance_counters
        if counters is None:

            def hook():
                event.current_stream_wait()

        else:
            # Sent routes are globally conserved by all-to-all; recv_x[0] is the 128-aligned expanded compute load.
            route_load = topk_idx.numel()
            compute_load = recv_x[0].shape[0]

            def hook():
                event.current_stream_wait()
                counters.accumulate(
                    route_load=route_load,
                    compute_load=compute_load,
                )

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
        if self.eplb_experts_logical_to_physical_map is not None:
            w13, w2 = self._primary_weight_pack(w13), self._primary_weight_pack(w2)
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
                silu_and_mul_fwd(_gemm_out_a.view(-1, N), _silu_out)
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
