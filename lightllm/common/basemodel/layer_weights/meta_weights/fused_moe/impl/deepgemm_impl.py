import torch
from typing import Optional, Tuple, Any
from .base_impl import FuseMoeBaseImpl
from lightllm.distributed import dist_group_manager
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
    get_env_start_args,
    get_redundancy_expert_ids,
    get_redundancy_expert_num,
)
from lightllm.utils.dist_utils import get_global_rank, get_global_world_size
from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import (
    fused_experts,
    get_ep_num_sms,
    decode_masked_group_gemm,
    chunked_expanded_moe_forward,
    quantize_fused_experts_input,
    get_mk_alignment_for_contiguous_layout,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import (
    ep_reduce_decode_output,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import redundancy_topk_ids_repair


class FuseMoeDeepGEMM(FuseMoeBaseImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        world_size = get_global_world_size()
        assert self.n_routed_experts % world_size == 0
        global_rank = get_global_rank()
        self.global_rank_ = global_rank
        self.global_world_size_ = world_size
        self.ep_n_routed_experts = self.n_routed_experts // world_size
        self.redundancy_expert_num = get_redundancy_expert_num()
        self.auto_update_redundancy_expert = get_env_start_args().auto_update_redundancy_expert
        redundancy_ids = list(get_redundancy_expert_ids(self.layer_index))
        self.redundancy_expert_ids_tensor = torch.tensor(redundancy_ids, dtype=torch.int64, device="cuda")
        self.routed_expert_counter_tensor = torch.zeros((self.n_routed_experts,), dtype=torch.int64, device="cuda")
        self.total_expert_num_contain_redundancy = self.n_routed_experts + self.redundancy_expert_num * world_size
        first_local_expert_id = global_rank * self.ep_n_routed_experts
        self.local_logics_expert_ids_list = (
            list(range(first_local_expert_id, first_local_expert_id + self.ep_n_routed_experts)) + redundancy_ids
        )

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
    ):
        """只选择逻辑专家，冗余专家 ID 留到 ``_prepare_expert_execution``。"""
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
        return topk_weights, topk_ids

    def _prepare_expert_execution(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_prefill: bool,
        shared_expert_gate: Optional[torch.Tensor] = None,
    ):
        """在逻辑路由采集之后，生成 DeepEP 实际使用的专家 ID。"""
        assert is_prefill is not None, "is_prefill must be explicitly specified for fused MoE execution"
        assert shared_expert_gate is None, "fused shared expert as MoE is not supported by DeepGEMM fused MoE"
        if self.redundancy_expert_num > 0:
            # repair 会原地改写 ID。复制后再改，保留已经交给 capture 的逻辑 ID。
            topk_ids = topk_ids.clone()
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
        is_prefill: bool,
        router_logits: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        limit: Optional[float] = None,
        clamp_up_add_one: bool = True,
        shared_expert_func=None,
    ):
        return fused_experts(
            hidden_states=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_idx=topk_ids.to(torch.long),
            num_experts=self.total_expert_num_contain_redundancy,  # number of all experts contain redundancy
            quant_method=self.quant_method,
            is_prefill=is_prefill,
            previous_event=None,  # for overlap
            alpha=alpha,
            limit=limit,
            clamp_up_add_one=clamp_up_add_one,
            shared_expert_func=shared_expert_func,
        )

    def decode_dispatch(
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
        topk_weights, topk_idx = self._prepare_expert_execution(topk_weights, topk_idx, is_prefill=False)
        return self.decode_dispatch_with_topk(
            hidden_states=hidden_states,
            w13=w13,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )

    def decode_dispatch_with_topk(
        self,
        hidden_states: torch.Tensor,
        w13: WeightPack,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        topk_idx = topk_idx.to(torch.long)
        qinput_tensor = quantize_fused_experts_input(hidden_states, w13, self.quant_method)
        recv_x, _, recv_topk_weights, ep_handle, event = dist_group_manager.ep_buffer.dispatch(
            qinput_tensor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=self.total_expert_num_contain_redundancy,
            num_max_tokens_per_rank=get_deepep_num_max_dispatch_tokens_per_rank_decode(),
            expert_alignment=get_mk_alignment_for_contiguous_layout(),
            num_sms=get_ep_num_sms(overlap_with_compute=True),
            async_with_compute_stream=True,
            # Decode 不传 previous_event，输出直接归属计算流；DeepEP 会为
            # 通信流的异步访问登记 allocator 生命周期。
            allocate_on_comm_stream=False,
            do_cpu_sync=False,
            do_handle_copy=False,
            do_expand=True,
            do_zero_padding=True,
            use_tma_aligned_col_major_sf=True,
        )

        def hook():
            event.current_stream_wait()

        return recv_x, ep_handle, recv_topk_weights, hook

    def quantize_dispatch_input(self, hidden_states: torch.Tensor, w13: WeightPack):
        return quantize_fused_experts_input(hidden_states, w13, self.quant_method)

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
        topk_weights, topk_idx = self._prepare_expert_execution(topk_weights, topk_idx, is_prefill=True)
        qinput_tensor = quantize_fused_experts_input(hidden_states, w13, self.quant_method)
        return topk_weights, topk_idx.to(torch.long), qinput_tensor

    def prefill_dispatch(
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
            expert_alignment=get_mk_alignment_for_contiguous_layout(),
            num_sms=get_ep_num_sms(overlap_with_compute=True),
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

    def decode_masked_group_gemm(
        self,
        recv_x: Tuple[torch.Tensor],
        w13: WeightPack,
        w2: WeightPack,
        ep_handle: Any,
        dtype: torch.dtype,
        expected_m: int,
        alpha: Optional[float] = None,
        limit: Optional[float] = None,
        clamp_up_add_one: bool = True,
    ):
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        return decode_masked_group_gemm(
            recv_x=recv_x,
            expert_token_psum=ep_handle.psum_num_recv_tokens_per_expert,
            expert_alignment=ep_handle.expert_alignment,
            dtype=dtype,
            w1=w13_weight,
            w1_scale=w13_scale,
            w2=w2_weight,
            w2_scale=w2_scale,
            expected_m=expected_m,
            alpha=alpha,
            limit=limit,
            clamp_up_add_one=clamp_up_add_one,
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
        alpha: Optional[float] = None,
        limit: Optional[float] = None,
        clamp_up_add_one: bool = True,
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
                hidden_dtype=hidden_dtype,
                alpha=alpha,
                limit=limit,
                clamp_up_add_one=clamp_up_add_one,
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
            if Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL):
                N = w13_weight.shape[1]
                _gemm_out_a = torch.zeros((1, N), device=recv_x[0].device, dtype=hidden_dtype)
                _silu_out = torch.zeros((1, N // 2), device=recv_x[0].device, dtype=hidden_dtype)
                silu_and_mul_fwd(
                    _gemm_out_a.view(-1, N),
                    _silu_out,
                    alpha=alpha,
                    limit=limit,
                    clamp_up_add_one=clamp_up_add_one,
                )
                _gemm_out_a, _silu_out = None, None
        # 只为占显存主体的 recv_x 登记计算流；较小的路由 metadata 继续由
        # handle 和当前调用栈维持生命周期。
        compute_stream = torch.cuda.current_stream()
        recv_x[0].record_stream(compute_stream)
        recv_x[1].record_stream(compute_stream)
        del recv_x
        return gather_out

    def decode_combine(
        self,
        expert_output: torch.Tensor,
        ep_handle: Any,
        recv_topk_weights: torch.Tensor,
    ):
        # 阶段 1：把按 expert 展开的输出归约回去重接收 token 布局。
        # psum_num_recv_tokens_per_scaleup_rank 的最后一项是去重接收 token 数，
        # 不是按 expert 展开后的有效行数。
        dense_output, compact_metadata = ep_reduce_decode_output(
            expert_output=expert_output,
            route_weights=recv_topk_weights,
            recv_src_metadata=ep_handle.recv_src_metadata,
            num_valid_recv_tokens=ep_handle.psum_num_recv_tokens_per_scaleup_rank[-1:],
        )
        ep_handle.recv_src_metadata = compact_metadata

        # 阶段 2：compact metadata 的第一个 top-k 槽位指向同序 dense row，
        # 其余槽位置为 -1。异步 combine 返回 event，由调用方在消费输出前等待。
        combined_x, _, event = dist_group_manager.ep_buffer.combine(
            dense_output,
            ep_handle,
            topk_weights=None,
            num_sms=get_ep_num_sms(overlap_with_compute=True),
            async_with_compute_stream=True,
            # Decode combine 没有 previous_event，输出归属计算流；通信流的异步
            # 生命周期由 DeepEP 内部统一登记。
            allocate_on_comm_stream=False,
        )

        def hook():
            event.current_stream_wait()

        return combined_x, hook

    def prefill_combine(
        self,
        gemm_out_b: torch.Tensor,
        handle: Any,
        overlap_event: Optional[Any] = None,
    ):
        combined_x, _, event = dist_group_manager.ep_buffer.combine(
            gemm_out_b,
            handle,
            topk_weights=None,
            num_sms=get_ep_num_sms(overlap_with_compute=True),
            previous_event=overlap_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )

        def hook():
            event.current_stream_wait()

        return combined_x, hook
