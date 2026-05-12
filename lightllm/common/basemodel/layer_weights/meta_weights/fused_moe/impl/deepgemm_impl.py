import torch
from typing import Optional, Tuple, Any
from .triton_impl import FuseMoeTriton
from lightllm.distributed import dist_group_manager
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import (
    fused_experts_impl,
    masked_group_gemm,
    deepgemm_grouped_fp8_fp4_nt_contiguous,
    deepgemm_grouped_fp8_nt_contiguous,
)
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
    tma_align_input_scale,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_scatter_gather import ep_scatter, ep_gather
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import redundancy_topk_ids_repair
from lightllm.utils.device_utils import is_sm100_gpu


class FuseMoeDeepGEMM(FuseMoeTriton):
    def _get_ep_num_sms(self) -> int:
        return getattr(dist_group_manager, "ep_num_sms", None) or 0

    def _use_sm100_fp4_moe(self) -> bool:
        return is_sm100_gpu() and self.quant_method.method_name == "deepgemm-fp8fp4-b32"

    def _get_mega_moe_weights(self, w13: WeightPack, w2: WeightPack):
        cache_key = (
            w13.weight.data_ptr(),
            w13.weight_scale.data_ptr(),
            w2.weight.data_ptr(),
            w2.weight_scale.data_ptr(),
        )
        if getattr(self, "_mega_moe_weight_cache_key", None) != cache_key:
            import deep_gemm

            self._mega_moe_weight_cache = deep_gemm.transform_weights_for_mega_moe(
                (w13.weight, w13.weight_scale),
                (w2.weight, w2.weight_scale),
            )
            self._mega_moe_weight_cache_key = cache_key
        return self._mega_moe_weight_cache

    def _get_mega_moe_stats(self, num_local_experts: int, device: torch.device):
        stats = getattr(self, "_mega_moe_stats", None)
        if stats is None or stats.numel() != num_local_experts or stats.device != device:
            stats = torch.zeros((num_local_experts,), device=device, dtype=torch.int32)
            self._mega_moe_stats = stats
        return stats

    def _mega_moe(
        self,
        hidden_states: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp8

        buffer = getattr(dist_group_manager, "ep_mega_moe_buffer", None)
        if buffer is None:
            raise RuntimeError("SM100 Mega MoE requires dist_group_manager.ep_mega_moe_buffer to be initialized")

        num_tokens = hidden_states.shape[0]
        if num_tokens > buffer.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE got {num_tokens} tokens, exceeding num_max_tokens_per_rank={buffer.num_max_tokens_per_rank}"
            )

        qinput_tensor = per_token_cast_to_fp8(
            hidden_states,
            use_ue8m0=True,
            gran_k=self.quant_method.block_size,
            use_packed_ue8m0=True,
        )
        l1_weights, l2_weights = self._get_mega_moe_weights(w13, w2)
        cumulative_stats = self._get_mega_moe_stats(w13.weight.shape[0], hidden_states.device)
        buffer.x[:num_tokens].copy_(qinput_tensor[0])
        buffer.x_sf[:num_tokens].copy_(qinput_tensor[1])
        buffer.topk_idx[:num_tokens].copy_(topk_ids)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

        output = torch.empty_like(hidden_states)
        deep_gemm.fp8_fp4_mega_moe(
            output,
            l1_weights,
            l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_stats,
        )
        return output

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
    ):
        """Select experts and return topk weights and ids."""
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
        if self.redundancy_expert_num > 0:
            redundancy_topk_ids_repair(
                topk_ids=topk_ids,
                redundancy_expert_ids=self.redundancy_expert_ids_tensor,
                ep_expert_num=self.ep_n_routed_experts,
                global_rank=self.global_rank_,
                expert_counter=self.routed_expert_counter_tensor,
                enable_counter=self.auto_update_redundancy_expert,
            )
        return topk_weights, topk_ids

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

        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        if self._use_sm100_fp4_moe():
            return self._mega_moe(input_tensor, w13, w2, topk_weights, topk_ids.to(torch.long))

        use_fp8_w8a8 = self.quant_method.method_name != "none"
        buffer = dist_group_manager.ep_buffer if is_prefill else dist_group_manager.ep_low_latency_buffer
        output = fused_experts_impl(
            hidden_states=input_tensor,
            w1=w13_weight,
            w2=w2_weight,
            topk_weights=topk_weights,
            topk_idx=topk_ids.to(torch.long),
            num_experts=self.total_expert_num_contain_redundancy,  # number of all experts contain redundancy
            buffer=buffer,
            is_prefill=is_prefill,
            use_fp8_w8a8=use_fp8_w8a8,
            use_fp8_all2all=use_fp8_w8a8,
            use_int8_w8a16=False,  # default to False
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            previous_event=None,  # for overlap
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
        topk_weights, topk_idx = self._select_experts(
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
        topk_weights, topk_idx = self._select_experts(
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
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        if self._use_sm100_fp4_moe():
            from deep_gemm.utils import per_token_cast_to_fp8

            qinput_tensor = per_token_cast_to_fp8(
                hidden_states,
                use_ue8m0=True,
                gran_k=self.quant_method.block_size,
                use_packed_ue8m0=True,
            )
            return topk_weights, topk_idx.to(torch.long), qinput_tensor

        block_size_k = 0
        if w13_weight.ndim == 3:
            block_size_k = w13_weight.shape[2] // w13_scale.shape[2]
        assert block_size_k == 128, "block_size_k must be 128"
        qinput_tensor, input_scale = per_token_group_quant_fp8(hidden_states, block_size_k, dtype=w13_weight.dtype)
        return topk_weights, topk_idx.to(torch.long), (qinput_tensor, input_scale)

    def dispatch(
        self,
        qinput_tensor: Tuple[torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_event: Optional[Any] = None,
    ):
        buffer = dist_group_manager.ep_buffer
        num_max_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_prefill()
        if self._use_sm100_fp4_moe():
            recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
                qinput_tensor,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=self.total_expert_num_contain_redundancy,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                expert_alignment=128,
                num_sms=self._get_ep_num_sms(),
                previous_event=overlap_event,
                async_with_compute_stream=True,
                allocate_on_comm_stream=True,
                do_cpu_sync=False,
                do_handle_copy=False,
                do_expand=True,
                use_tma_aligned_col_major_sf=True,
            )

            def hook():
                event.current_stream_wait()

            return recv_x, recv_topk_idx, recv_topk_weights, handle.psum_num_recv_tokens_per_expert, handle, hook

        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
            qinput_tensor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=self.total_expert_num_contain_redundancy,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            expert_alignment=128,
            num_sms=self._get_ep_num_sms(),
            previous_event=overlap_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
            do_cpu_sync=True,
            do_handle_copy=False,
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
            recv_x, masked_m, dtype, w13_weight, w13_scale, w2_weight, w2_scale, expected_m=expected_m
        )

    def prefilled_group_gemm(
        self,
        num_recv_tokens_per_expert_list,
        recv_x: Tuple[torch.Tensor],
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        hidden_dtype=torch.bfloat16,
    ):
        device = recv_x[0].device
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        _, K = recv_x[0].shape
        _, N, _ = w13_weight.shape
        block_size = self.quant_method.block_size
        if self._use_sm100_fp4_moe():
            n = recv_x[0].shape[0]
            l1_y = torch.empty((n, N), device=device, dtype=hidden_dtype)
            deepgemm_grouped_fp8_fp4_nt_contiguous(
                recv_x,
                (w13_weight, w13_scale),
                l1_y,
                num_recv_tokens_per_expert_list,
                use_psum_layout=True,
            )
            silu_out = torch.empty((n, N // 2), device=device, dtype=hidden_dtype)
            silu_and_mul_fwd(l1_y.view(-1, N), silu_out)
            if recv_topk_weights is not None:
                recv_topk_weights = recv_topk_weights.reshape(-1)[:n]
                silu_out.mul_(recv_topk_weights.view(-1, 1))

            from deep_gemm.utils import per_token_cast_to_fp8

            qsilu_out = per_token_cast_to_fp8(
                silu_out,
                use_ue8m0=True,
                gran_k=block_size,
                use_packed_ue8m0=True,
            )
            l2_y = torch.empty((n, K), device=device, dtype=hidden_dtype)
            deepgemm_grouped_fp8_fp4_nt_contiguous(
                qsilu_out,
                (w2_weight, w2_scale),
                l2_y,
                num_recv_tokens_per_expert_list,
                use_psum_layout=True,
            )
            return l2_y

        # scatter
        all_tokens = sum(num_recv_tokens_per_expert_list)  # calcu padding all nums.
        # gather_out shape [recive_num_tokens, hidden]
        gather_out = torch.empty_like(recv_x[0], device=device, dtype=hidden_dtype)
        if all_tokens > 0:
            input_tensor = [
                torch.empty((all_tokens, K), device=device, dtype=recv_x[0].dtype),
                torch.empty((all_tokens, K // 128), device=device, dtype=torch.float32),
            ]
            # when m_indices is filled ok.
            # m_indices show token use which expert, example, [0, 0, 0, 0, .... 1, 1, 1, 1,...., cur_expert_num - 1, ..]
            # the count of 0 is num_recv_tokens_per_expert_list[0], the count of 1 is num_recv_tokens_per_expert_list[1]
            # ...
            m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
            # output_index shape [recive_num_tokens, topk_num]
            # output_index use to show the token index in input_tensor
            output_index = torch.empty_like(recv_topk_idx)

            num_recv_tokens_per_expert = torch.tensor(
                num_recv_tokens_per_expert_list, dtype=torch.int32, pin_memory=True, device="cpu"
            ).cuda(non_blocking=True)

            expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)

            ep_scatter(
                recv_x[0],
                recv_x[1],
                recv_topk_idx,
                num_recv_tokens_per_expert,
                expert_start_loc,
                input_tensor[0],
                input_tensor[1],
                m_indices,
                output_index,
            )
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
            # groupgemm (contiguous layout)
            gemm_out_a = torch.empty((all_tokens, N), device=device, dtype=hidden_dtype)

            deepgemm_grouped_fp8_nt_contiguous(input_tensor, (w13_weight, w13_scale), gemm_out_a, m_indices)

            # silu_and_mul_fwd + qaunt
            # TODO fused kernel
            silu_out = torch.empty((all_tokens, N // 2), device=device, dtype=hidden_dtype)

            silu_and_mul_fwd(gemm_out_a.view(-1, N), silu_out)
            qsilu_out, qsilu_out_scale = per_token_group_quant_fp8(
                silu_out, block_size, dtype=w13_weight.dtype, column_major_scales=True, scale_tma_aligned=True
            )

            # groupgemm (contiguous layout)
            gemm_out_b = torch.empty((all_tokens, K), device=device, dtype=hidden_dtype)

            deepgemm_grouped_fp8_nt_contiguous(
                (qsilu_out, qsilu_out_scale), (w2_weight, w2_scale), gemm_out_b, m_indices
            )
            # gather and local reduce
            ep_gather(gemm_out_b, recv_topk_idx, recv_topk_weights, output_index, gather_out)
        else:
            ######################################## warning ##################################################
            # here is used to match autotune feature, make moe model run same triton kernel in different rank.
            # in some special case, one rank will recv 0 token, so add a token to make it run triton kernel.
            if Autotuner.is_autotune_warmup():
                _gemm_out_a = torch.zeros((1, N), device=device, dtype=hidden_dtype)
                _silu_out = torch.zeros((1, N // 2), device=device, dtype=hidden_dtype)
                silu_and_mul_fwd(_gemm_out_a.view(-1, N), _silu_out)
                _gemm_out_a, _silu_out = None, None

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
            num_sms=self._get_ep_num_sms(),
            previous_event=overlap_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )

        def hook():
            event.current_stream_wait()

        return combined_x, hook
