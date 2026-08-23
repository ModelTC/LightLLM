import threading

import torch
from typing import Optional, Tuple, Any
from deep_ep import ElasticBuffer
from .base_impl import FuseMoeBaseImpl
from ..expert_parallel_state import ExpertParallelState
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
from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_kernels import eplb_map_to_physical_long_fast


# Dispatch policy mapping is launched after the caller has captured its
# overlap event.  One routing stream per device lets that mapping wait for the
# captured inputs without serializing subsequent compute-stream microbatches.
_PREFILL_ROUTING_STREAMS = {}
_PREFILL_ROUTING_STREAMS_LOCK = threading.Lock()


def _get_prefill_routing_stream(device: torch.device):
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device.type, device_index)
    stream = _PREFILL_ROUTING_STREAMS.get(key)
    if stream is not None:
        return stream
    with _PREFILL_ROUTING_STREAMS_LOCK:
        stream = _PREFILL_ROUTING_STREAMS.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            _PREFILL_ROUTING_STREAMS[key] = stream
        return stream


class FuseMoeDeepGEMM(FuseMoeBaseImpl):
    def __init__(self, *args, expert_parallel_state: ExpertParallelState, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_parallel_state = expert_parallel_state
        self.eplb_state = expert_parallel_state.eplb
        self.ep_balance_counters = None
        self._primary_weight_pack_cache = {}

    def _next_eplb_sample_index(self) -> int:
        eplb = self.eplb_state
        if not eplb.recording:
            return 0
        sample_index = eplb.recorded_sample_count % eplb.route_counter.shape[0]
        eplb.recorded_sample_count += 1
        return sample_index

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
        route_prefill: bool = True,
    ):
        """Select experts and return topk weights and ids."""
        assert shared_expert_gate is None, "fused shared expert as MoE is not supported by DeepGEMM fused MoE"
        eplb = self.eplb_state
        eplb_active = eplb is not None
        # For grouped prefill, selecting logical IDs, then launching a second
        # kernel to count and remap them is avoidable.  Keep every observable
        # logical-ID path on the generic implementation: callbacks and
        # autotune need logical routing IDs, and per-expert scales index them.
        fused_eplb_grouped_topk = (
            is_prefill is True
            and route_prefill
            and eplb_active
            and use_grouped_topk
            and per_expert_scale is None
            and not preserve_logical_ids
            and not Autotuner.is_autotune_warmup()
        )
        if fused_eplb_grouped_topk:
            from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import triton_grouped_topk_eplb

            group_score_topk_num = 2 if topk_group == 4 and num_expert_group == 8 and top_k == 8 else 1
            sample_index = self._next_eplb_sample_index()
            topk_weights, topk_ids = triton_grouped_topk_eplb(
                hidden_states=input_tensor,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                logical_to_physical_map=eplb.logical_to_physical_map,
                logical_replica_count=eplb.logical_replica_count,
                expert_counter=eplb.route_counter,
                sample_index=sample_index,
                record_load=eplb.recording,
                group_score_used_topk_num=group_score_topk_num,
            )
        else:
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
        if route_prefill and is_prefill is True and eplb_active and not fused_eplb_grouped_topk:
            sample_index = self._next_eplb_sample_index()
            topk_ids = eplb_map_to_physical_long_fast(
                topk_ids,
                eplb.logical_to_physical_map,
                eplb.logical_replica_count,
                eplb.route_counter,
                sample_index,
                record_load=eplb.recording,
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
        clamp_limit: Optional[float] = None,
        alloc_tensor_func=torch.empty,
    ):
        if is_prefill is False:
            w13 = self._primary_weight_pack(w13)
            w2 = self._primary_weight_pack(w2)
            num_experts = self.n_routed_experts
        else:
            num_experts = self.expert_parallel_state.total_physical_experts
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
            clamp_limit=clamp_limit,
            alloc_tensor_func=alloc_tensor_func,
            ep_balance_counters=self.ep_balance_counters,
        )
        return output

    def fused_experts_with_topk(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_prefill: Optional[bool] = None,
        clamp_limit: Optional[float] = None,
        alloc_tensor_func=torch.empty,
    ):
        if is_prefill is True:
            topk_ids = self._prepare_prefill_topk_ids(topk_ids)
        return self._fused_experts(
            input_tensor=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            is_prefill=is_prefill,
            clamp_limit=clamp_limit,
            alloc_tensor_func=alloc_tensor_func,
        )

    def _prepare_prefill_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        eplb = self.eplb_state
        if eplb is None:
            return topk_ids
        if not topk_ids.is_contiguous():
            topk_ids = topk_ids.contiguous()
        sample_index = self._next_eplb_sample_index()
        return eplb_map_to_physical_long_fast(
            topk_ids,
            eplb.logical_to_physical_map,
            eplb.logical_replica_count,
            eplb.route_counter,
            sample_index,
            record_load=eplb.recording,
        )

    def _prepare_prefill_dispatch_topk_ids(
        self,
        topk_ids: torch.Tensor,
        overlap_event: Optional[Any],
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Prepare dispatch IDs and the event which makes them visible to DeepEP.

        Policy-free dispatch deliberately retains the original path: it only
        performs the existing long conversion and forwards the caller event.
        A policy maps logical IDs directly to a freshly allocated int64
        physical-ID tensor on a shared routing stream after the caller's input
        event, then hands DeepEP an event captured after that map.
        """
        if self.eplb_state is None:
            return topk_ids.to(torch.long), overlap_event

        source_event = overlap_event
        if source_event is None:
            source_event = ElasticBuffer.capture()
        routing_stream = _get_prefill_routing_stream(topk_ids.device)
        with torch.cuda.stream(routing_stream):
            source_event.current_stream_wait()
            topk_ids = self._prepare_prefill_topk_ids(topk_ids)
            dispatch_event = ElasticBuffer.capture()
        return topk_ids, dispatch_event

    def _primary_weight_pack(self, weight_pack: WeightPack) -> WeightPack:
        """Return the cached local-primary view used by all decode paths."""
        if self.eplb_state is None:
            return weight_pack
        cache = getattr(self, "_primary_weight_pack_cache", None)
        if cache is None:
            cache = self._primary_weight_pack_cache = {}
        cache_key = id(weight_pack)
        primary = cache.get(cache_key)
        if primary is None:
            primary_experts_per_rank = self.expert_parallel_state.primary_experts_per_rank
            primary = WeightPack(
                weight=weight_pack.weight[:primary_experts_per_rank],
                weight_scale=(
                    weight_pack.weight_scale[:primary_experts_per_rank]
                    if weight_pack.weight_scale is not None
                    else None
                ),
                weight_zero_point=(
                    getattr(weight_pack, "weight_zero_point", None)[:primary_experts_per_rank]
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

        return self.low_latency_dispatch_with_topk(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )

    def low_latency_dispatch_with_topk(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        topk_idx = topk_idx.to(torch.long)
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_decode()
        use_fp8_w8a8 = self.quant_method.method_name != "none"
        recv_x, masked_m, handle, event, hook = dist_group_manager.ep_low_latency_buffer.low_latency_dispatch(
            topk_idx=topk_idx,
            x=hidden_states,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            # Decode is deliberately isolated from EPLB's physical redundant
            # rows: DeepEP sees the original logical expert IDs.
            num_experts=self.n_routed_experts,
            use_fp8=use_fp8_w8a8,
            async_finish=False,
            return_recv_hook=True,
        )
        return recv_x, masked_m, topk_idx, topk_weights, handle, hook

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
            route_prefill=False,
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
        topk_idx, dispatch_event = self._prepare_prefill_dispatch_topk_ids(topk_idx, overlap_event)
        buffer = dist_group_manager.ep_buffer
        num_max_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_prefill()
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
            qinput_tensor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=self.expert_parallel_state.total_physical_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            expert_alignment=128,
            num_sms=get_ep_num_sms(),
            previous_event=dispatch_event,
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
        clamp_limit: Optional[float] = None,
    ):
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
            clamp_limit=clamp_limit,
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
        clamp_limit: Optional[float] = None,
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
                clamp_limit=clamp_limit,
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
                silu_and_mul_fwd(_gemm_out_a.view(-1, N), _silu_out, limit=clamp_limit)
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
