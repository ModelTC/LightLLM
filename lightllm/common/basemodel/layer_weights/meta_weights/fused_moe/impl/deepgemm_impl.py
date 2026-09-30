import torch
from typing import Optional, Tuple, Any

from .base_impl import FuseMoeBaseImpl
from lightllm.distributed import dist_group_manager
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.utils.envs_utils import (
    get_env_start_args,
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
)
from lightllm.utils.dist_utils import (
    get_global_rank,
    get_global_world_size,
    get_node_world_size,
)
from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
    eplb_repair_topk_ids,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import (
    ep_reduce_decode_output,
)
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType


class FuseMoeDeepGEMM(FuseMoeBaseImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        world_size = get_global_world_size()
        assert self.n_routed_experts % world_size == 0
        global_rank = get_global_rank()
        start_args = get_env_start_args()
        self.num_redundant_experts_per_rank = start_args.eplb_num_redundant_experts_per_rank

        if self.num_redundant_experts_per_rank > 0:
            self._init_eplb_runtime(
                start_args=start_args,
                world_size=world_size,
                global_rank=global_rank,
            )
        else:
            self._init_standard_expert_layout(
                world_size=world_size,
                global_rank=global_rank,
            )

    def _init_eplb_runtime(self, start_args: Any, world_size: int, global_rank: int) -> None:
        """初始化本地物理槽位以及可更新的 EPLB 路由运行态。

        ``local_logics_expert_ids_list`` 始终描述全部本地物理行。初始化时主专家
        在前、冗余专家在后；负载均衡运行后允许替换任意物理行，并在同一个
        安全推理边界同时更新专家权重和 ``logical_to_physical_map``。
        """
        # 延迟导入：顶层导入会经 mode_backend 包形成 meta_weights -> server 的循环依赖。
        from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
            build_initial_local_expert_ids,
            build_logical_to_physical_map,
            load_layer_placement,
        )

        self.num_total_physical_experts = self.n_routed_experts + world_size * self.num_redundant_experts_per_rank

        # 阶段 1：先构造确定性的默认布局。未指定配置文件，或配置读取、校验失败时，
        # 后续权重初始化会继续使用这份布局。
        initial_local_expert_ids_by_rank = build_initial_local_expert_ids(
            self.n_routed_experts,
            world_size,
            self.num_redundant_experts_per_rank,
        )

        # 阶段 2：如果指定了配置文件，尝试读取与当前层及部署拓扑匹配的历史布局。
        # load_layer_placement 会负责记录 warning，并在任何异常或配置无效时返回 None。
        config_path = start_args.eplb_config_path
        if config_path is not None:
            saved_placement = load_layer_placement(
                config_path,
                layer_index=self.layer_index,
                num_logical_experts=self.n_routed_experts,
                world_size=world_size,
                num_redundant_experts_per_rank=self.num_redundant_experts_per_rank,
            )

            # 阶段 3：只有完整校验通过的历史布局才会替换默认布局，使专家权重在
            # 初始化时直接加载到上一次优化后的物理槽位中。
            if saved_placement is not None:
                initial_local_expert_ids_by_rank = saved_placement
        self.local_logics_expert_ids_list = initial_local_expert_ids_by_rank[global_rank]
        self.logical_to_physical_map = torch.tensor(
            build_logical_to_physical_map(
                initial_local_expert_ids_by_rank,
                self.n_routed_experts,
                current_rank=global_rank,
                node_world_size=get_node_world_size(),
            ),
            dtype=torch.int32,
        ).cuda()
        if start_args.eplb_run_mode == "prefill":
            # 环形缓冲区保留最近 24 次 prefill 路由采样，每次采样写入独立的一行；
            # 始终按 logical expert 统计，冗余副本不会拆散规划器观察到的负载信号。
            self.prefill_route_counter = torch.zeros(
                (24, self.n_routed_experts),
                dtype=torch.int64,
                device="cuda",
            )
            # [0] 是单调递增的 sample index；[1] 用于在同一个 kernel 内协调
            # 目标行清零，并从所有 program 中选出最后完成者。
            self.prefill_route_sample_index = torch.zeros(2, dtype=torch.int64, device="cuda")
            self.decode_route_counter = None
        else:
            self.prefill_route_counter = None
            self.prefill_route_sample_index = None
            # decode 共现矩阵只写包含主对角线的上三角。对角线记录单个
            # logical expert 的精确负载，非对角位置记录无序 expert pair
            # 在同一个 token 的 top-k 中共同出现的次数。
            self.decode_route_counter = torch.zeros(
                (self.n_routed_experts, self.n_routed_experts),
                dtype=torch.int64,
                device="cuda",
            )

    def _init_standard_expert_layout(self, world_size: int, global_rank: int) -> None:
        """初始化未启用 EPLB 时连续均分的本地专家布局。"""
        self.num_total_physical_experts = self.n_routed_experts
        num_local_experts = self.n_routed_experts // world_size
        first_local_expert_id = global_rank * num_local_experts
        self.local_logics_expert_ids_list = list(
            range(
                first_local_expert_id,
                first_local_expert_id + num_local_experts,
            )
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
        """只选择逻辑专家，不在此阶段应用 EPLB 物理布局。"""
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
        assert is_prefill is not None, "is_prefill must be explicitly specified for fused MoE execution"
        assert shared_expert_gate is None, "fused shared expert as MoE is not supported by DeepGEMM fused MoE"
        if self.num_redundant_experts_per_rank > 0:
            # 延迟导入以避免 meta_weights -> server 的循环依赖。prefill 和
            # decode 的分发策略统一由 EPLB 模块管理。
            from lightllm.server.router.model_infer.mode_backend.eplb.eplb_utils import (
                get_eplb_dispatch_mode,
                should_record_decode_route,
                should_record_prefill_route,
            )

            dispatch_mode = get_eplb_dispatch_mode(is_prefill=is_prefill)
            update_prefill_route_counter = should_record_prefill_route(is_prefill=is_prefill)
            update_decode_route_counter = should_record_decode_route(is_prefill=is_prefill)
            topk_ids = eplb_repair_topk_ids(
                logical_topk_ids=topk_ids,
                logical_to_physical_map=self.logical_to_physical_map,
                prefill_route_counter=self.prefill_route_counter,
                prefill_route_sample_index=self.prefill_route_sample_index,
                update_prefill_route_counter=update_prefill_route_counter,
                decode_route_counter=self.decode_route_counter,
                update_decode_route_counter=update_decode_route_counter,
                mode=dispatch_mode,
            )
        return topk_weights, topk_ids

    # ==================== Prefill / Decode 公共接口 ====================

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_prefill: bool,
        router_logits: Optional[torch.Tensor] = None,
    ):
        output = grouped_fused_moe_ep.fused_experts(
            hidden_states=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_idx=topk_ids.to(torch.long),
            num_experts=self.num_total_physical_experts,
            quant_method=self.quant_method,
            is_prefill=is_prefill,
        )
        return output

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
        qinput_tensor = grouped_fused_moe_ep.quantize_fused_experts_input(hidden_states, w13, self.quant_method)
        return topk_weights, topk_idx.to(torch.long), qinput_tensor

    # ==================== Decode 接口 ====================

    def decode_dispatch(
        self,
        hidden_states: torch.Tensor,
        w13: WeightPack,
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
        topk_weights, topk_idx = self._prepare_expert_execution(topk_weights, topk_idx, is_prefill=False)

        topk_idx = topk_idx.to(torch.long)
        qinput_tensor = grouped_fused_moe_ep.quantize_fused_experts_input(hidden_states, w13, self.quant_method)
        recv_x, _, recv_topk_weights, ep_handle, event = dist_group_manager.ep_buffer.dispatch(
            qinput_tensor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=self.num_total_physical_experts,
            num_max_tokens_per_rank=get_deepep_num_max_dispatch_tokens_per_rank_decode(),
            expert_alignment=grouped_fused_moe_ep.get_mk_alignment_for_contiguous_layout(),
            num_sms=grouped_fused_moe_ep.get_ep_num_sms(overlap_with_compute=True),
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

    def decode_masked_group_gemm(
        self,
        recv_x: Tuple[torch.Tensor, torch.Tensor],
        w13: WeightPack,
        w2: WeightPack,
        ep_handle: Any,
        dtype: torch.dtype,
        expected_m: int,
    ):
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        moe_output = grouped_fused_moe_ep.decode_masked_group_gemm(
            recv_x=recv_x,
            expert_token_psum=ep_handle.psum_num_recv_tokens_per_expert,
            expert_alignment=ep_handle.expert_alignment,
            dtype=dtype,
            w1=w13_weight,
            w1_scale=w13_scale,
            w2=w2_weight,
            w2_scale=w2_scale,
            expected_m=expected_m,
        )
        return moe_output

    def decode_combine(
        self,
        expert_output: torch.Tensor,
        ep_handle: Any,
        recv_topk_weights: torch.Tensor,
    ):
        # 阶段 1：把按 expert 展开的输出归约回去重接收 token 布局。
        # expert_output:    [num_expanded_rows, hidden_size]
        # recv_topk_weights:[num_expanded_rows]
        # recv_src_metadata:[num_recv_tokens_capacity, topk + 2]
        # dense_output:     [num_recv_tokens_capacity, hidden_size]
        # compact_metadata: [num_recv_tokens_capacity, topk + 2]
        # psum_num_recv_tokens_per_scaleup_rank 的 shape 为 [num_scaleup_ranks]，
        # 保存各来源 rank 的去重接收 token 数的 inclusive prefix sum。最后一项
        # 是 recv_src_metadata 的有效行数，不是按 expert 展开后的有效行数。
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
            num_sms=grouped_fused_moe_ep.get_ep_num_sms(overlap_with_compute=True),
            async_with_compute_stream=True,
            # Decode combine 没有 previous_event，输出归属计算流；通信流的异步
            # 生命周期由 DeepEP 内部统一登记。
            allocate_on_comm_stream=False,
        )

        def hook():
            event.current_stream_wait()

        return combined_x, hook

    # ==================== Prefill 接口 ====================

    def prefill_dispatch(
        self,
        qinput_tensor: Tuple[torch.Tensor, torch.Tensor],
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
            num_experts=self.num_total_physical_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            expert_alignment=grouped_fused_moe_ep.get_mk_alignment_for_contiguous_layout(),
            num_sms=grouped_fused_moe_ep.get_ep_num_sms(overlap_with_compute=True),
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

    def prefilled_group_gemm(
        self,
        num_recv_tokens_per_expert_list,
        num_unaligned_recv_tokens_per_expert: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        recv_x: Tuple[torch.Tensor, torch.Tensor],
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        hidden_dtype=torch.bfloat16,
    ):
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        assert recv_topk_idx is None
        all_tokens = sum(num_recv_tokens_per_expert_list)
        if all_tokens > 0:
            gather_out = grouped_fused_moe_ep.chunked_expanded_moe_forward(
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
                silu_and_mul_fwd(_gemm_out_a.view(-1, N), _silu_out)
                _gemm_out_a, _silu_out = None, None
        # 与 decode 一致，只为占显存主体的 recv_x 登记计算流；较小的路由
        # metadata 继续由 handle 和当前调用栈维持生命周期。
        compute_stream = torch.cuda.current_stream()
        recv_x[0].record_stream(compute_stream)
        recv_x[1].record_stream(compute_stream)
        del recv_x
        return gather_out

    def prefill_combine(
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
            num_sms=grouped_fused_moe_ep.get_ep_num_sms(overlap_with_compute=True),
            previous_event=overlap_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )

        def hook():
            event.current_stream_wait()

        return combined_x, hook
