import os
import torch
import threading
from typing import Optional, Tuple, List, Dict, Any
from lightllm.utils.dist_utils import get_global_world_size, get_global_rank, get_current_device_id
from .base_weight import BaseWeight
from lightllm.common.fused_moe.grouped_fused_moe_ep import (
    fused_experts_impl,
    masked_group_gemm,
    _deepgemm_grouped_fp8_nt_contiguous,
)
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.distributed import dist_group_manager
from lightllm.common.fused_moe.topk_select import select_experts
from lightllm.utils.envs_utils import get_deepep_num_max_dispatch_tokens_per_rank
from lightllm.utils.envs_utils import get_redundancy_expert_ids, get_redundancy_expert_num
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
    tma_align_input_scale,
)
from lightllm.common.fused_moe.deepep_scatter_gather import ep_scatter, ep_gather
from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import redundancy_topk_ids_repair
from lightllm.utils.log_utils import init_logger
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.common.quantization.quantize_method import QuantizedWeightPack


logger = init_logger(__name__)


class FusedMoeWeightEP(BaseWeight):
    def __init__(
        self,
        gate_proj_name: str,
        down_proj_name: str,
        up_proj_name: str,
        e_score_correction_bias_name: str,
        weight_prefix: str,
        n_routed_experts: int,
        data_type: torch.dtype,
        network_config: Dict[str, Any],
        layer_num: int,
        quant_cfg=None,
        hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.layer_num = layer_num
        self.quant_method = quant_cfg.get_quant_method(layer_num, "fused_moe")
        self.quantized_weight = quant_cfg.quantized_weight
        if self.quant_method is not None:
            self.weight_scale_suffix = self.quant_method.weight_scale_suffix
            self.quant_method.is_moe = True
            block_size = 1
            if hasattr(self.quant_method, "block_size"):
                block_size = self.quant_method.block_size
            self.block_size = block_size

        self.weight_prefix = weight_prefix
        self.w1_weight_name = gate_proj_name
        self.w2_weight_name = down_proj_name
        self.w3_weight_name = up_proj_name
        self.e_score_correction_bias_name = e_score_correction_bias_name
        self.n_routed_experts = n_routed_experts
        self.data_type_ = data_type
        self.hidden_size = hidden_size

        global_world_size = get_global_world_size()
        self.global_rank_ = get_global_rank()
        self.redundancy_expert_num = get_redundancy_expert_num()
        self.redundancy_expert_ids = get_redundancy_expert_ids(layer_num)
        logger.info(
            f"global_rank {self.global_rank_} layerindex {layer_num} redundancy_expertids: {self.redundancy_expert_ids}"
        )
        self.redundancy_expert_ids_tensor = torch.tensor(self.redundancy_expert_ids, dtype=torch.int64, device="cuda")
        self.routed_expert_counter_tensor = torch.zeros((self.n_routed_experts,), dtype=torch.int64, device="cuda")
        self.total_expert_num_contain_redundancy = (
            self.n_routed_experts + self.redundancy_expert_num * global_world_size
        )
        assert self.n_routed_experts % global_world_size == 0
        self.ep_n_routed_experts = self.n_routed_experts // global_world_size
        ep_load_expert_num = self.ep_n_routed_experts + self.redundancy_expert_num
        self.ep_load_expert_num = ep_load_expert_num
        self.experts_up_projs = [None] * ep_load_expert_num
        self.experts_gate_projs = [None] * ep_load_expert_num
        self.experts_up_proj_scales = [None] * ep_load_expert_num
        self.experts_gate_proj_scales = [None] * ep_load_expert_num
        self.e_score_correction_bias = None
        self.w2_list = [None] * ep_load_expert_num
        self.w2_scale_list = [None] * ep_load_expert_num
        self.scoring_func = network_config.get("scoring_func", "softmax")
        self.w1 = [None, None]  # weight, weight_scale
        self.w2 = [None, None]  # weight, weight_scale
        self.use_fp8_w8a8 = self.quant_method is not None
        network_config["n_group"] = network_config.get("n_group", 0)
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.use_grouped_topk = network_config["n_group"] > 0
        self.norm_topk_prob = network_config["norm_topk_prob"]
        self.n_group = network_config["n_group"]
        network_config["topk_group"] = network_config.get("topk_group", 0)
        self.topk_group = network_config["topk_group"]
        network_config["routed_scaling_factor"] = network_config.get("routed_scaling_factor", 1.0)
        self.routed_scaling_factor = network_config["routed_scaling_factor"]

        self.lock = threading.Lock()
        # init buffer

        # auto update redundancy expert vars
        self.auto_update_redundancy_expert: bool = get_env_start_args().auto_update_redundancy_expert

        # Pre-allocate memory if hidden_size is provided
        if self.hidden_size is not None:
            self._create_weight()

    def _create_weight(self):
        """Pre-allocate GPU memory for fused MoE weights"""
        if self.hidden_size is None:
            return

        total_expert_num = self.ep_load_expert_num
        # We need to determine intermediate size from network config or use a default
        # This will be updated when first weight is loaded if needed
        intermediate_size = getattr(self, "intermediate_size", None)
        if intermediate_size is None:
            # Default fallback - this will be corrected during load
            intermediate_size = self.hidden_size * 4

        device_id = get_current_device_id()

        if not self.quantized_weight and self.quant_method is not None:
            # Quantized weights
            w1_pack = self.quant_method.create_weight(
                total_expert_num * intermediate_size * 2, self.hidden_size, dtype=self.data_type_, device_id=device_id
            )
            self.w1[0] = w1_pack.weight.view(total_expert_num, intermediate_size * 2, self.hidden_size)
            self.w1[1] = w1_pack.weight_scale.view(total_expert_num, intermediate_size * 2, self.hidden_size)

            w2_pack = self.quant_method.create_weight(
                total_expert_num * self.hidden_size, intermediate_size, dtype=self.data_type_, device_id=device_id
            )
            self.w2[0] = w2_pack.weight.view(total_expert_num, self.hidden_size, intermediate_size)
            self.w2[1] = w2_pack.weight_scale.view(total_expert_num, self.hidden_size, intermediate_size)
        else:
            # Regular weights
            self.w1[0] = torch.empty(
                (total_expert_num, intermediate_size * 2, self.hidden_size),
                dtype=self.data_type_,
                device=f"cuda:{device_id}",
            )
            self.w2[0] = torch.empty(
                (total_expert_num, self.hidden_size, intermediate_size),
                dtype=self.data_type_,
                device=f"cuda:{device_id}",
            )

    def experts(
        self,
        input_tensor,
        router_logits,
        top_k,
        renormalize,
        use_grouped_topk,
        topk_group,
        num_expert_group,
        is_prefill,
    ):
        topk_weights, topk_ids = select_experts(
            hidden_states=input_tensor,
            router_logits=router_logits,
            correction_bias=self.e_score_correction_bias,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=self.scoring_func,
        )
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

        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        return fused_experts_impl(
            hidden_states=input_tensor,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_idx=topk_ids.to(torch.long),
            num_experts=self.total_expert_num_contain_redundancy,  # number of all experts contain redundancy
            buffer=dist_group_manager.ep_buffer,
            is_prefill=is_prefill,
            use_fp8_w8a8=self.use_fp8_w8a8,
            use_fp8_all2all=self.use_fp8_w8a8,
            use_int8_w8a16=False,  # default to False
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            previous_event=None,  # for overlap
        )

    def low_latency_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):

        topk_weights, topk_idx = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=self.e_score_correction_bias,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
            scoring_func=self.scoring_func,
        )
        topk_weights.mul_(self.routed_scaling_factor)

        if self.redundancy_expert_num > 0:
            redundancy_topk_ids_repair(
                topk_ids=topk_idx,
                redundancy_expert_ids=self.redundancy_expert_ids_tensor,
                ep_expert_num=self.ep_n_routed_experts,
                global_rank=self.global_rank_,
                expert_counter=self.routed_expert_counter_tensor,
                enable_counter=self.auto_update_redundancy_expert,
            )

        topk_idx = topk_idx.to(torch.long)
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank()
        recv_x, masked_m, handle, event, hook = dist_group_manager.ep_buffer.low_latency_dispatch(
            hidden_states,
            topk_idx,
            num_max_dispatch_tokens_per_rank,
            self.total_expert_num_contain_redundancy,
            use_fp8=self.use_fp8_w8a8,
            async_finish=False,
            return_recv_hook=True,
        )
        return recv_x, masked_m, topk_idx, topk_weights, handle, hook

    def select_experts_and_quant_input(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        topk_weights, topk_idx = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=self.e_score_correction_bias,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
            scoring_func=self.scoring_func,
        )
        topk_weights.mul_(self.routed_scaling_factor)
        if self.redundancy_expert_num > 0:
            redundancy_topk_ids_repair(
                topk_ids=topk_idx,
                redundancy_expert_ids=self.redundancy_expert_ids_tensor,
                ep_expert_num=self.ep_n_routed_experts,
                global_rank=self.global_rank_,
                expert_counter=self.routed_expert_counter_tensor,
                enable_counter=self.auto_update_redundancy_expert,
            )
        M, K = hidden_states.shape
        w1, w1_scale = self.w1
        block_size_k = 0
        if w1.ndim == 3:
            block_size_k = w1.shape[2] // w1_scale.shape[2]
        assert block_size_k == 128, "block_size_k must be 128"
        qinput_tensor, input_scale = per_token_group_quant_fp8(hidden_states, block_size_k, dtype=w1.dtype)
        return topk_weights, topk_idx.to(torch.long), (qinput_tensor, input_scale)

    def dispatch(
        self,
        qinput_tensor: Tuple[torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_event: Optional[Any] = None,
    ):
        buffer = dist_group_manager.ep_buffer
        # get_dispatch_layout
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            self.total_expert_num_contain_redundancy,
            previous_event=overlap_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = buffer.dispatch(
            qinput_tensor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
            expert_alignment=128,
        )

        def hook():
            event.current_stream_wait()

        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, hook

    def masked_group_gemm(
        self, recv_x: Tuple[torch.Tensor], masked_m: torch.Tensor, dtype: torch.dtype, expected_m: int
    ):
        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        return masked_group_gemm(recv_x, masked_m, dtype, w1, w1_scale, w2, w2_scale, expected_m=expected_m)

    def prefilled_group_gemm(
        self,
        num_recv_tokens_per_expert_list,
        recv_x: Tuple[torch.Tensor],
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        hidden_dtype=torch.bfloat16,
    ):
        device = recv_x[0].device
        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        _, K = recv_x[0].shape
        _, N, _ = w1.shape
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

            _deepgemm_grouped_fp8_nt_contiguous(input_tensor, (w1, w1_scale), gemm_out_a, m_indices)

            # silu_and_mul_fwd + qaunt
            # TODO fused kernel
            silu_out = torch.empty((all_tokens, N // 2), device=device, dtype=hidden_dtype)

            silu_and_mul_fwd(gemm_out_a.view(-1, N), silu_out)
            qsilu_out, qsilu_out_scale = per_token_group_quant_fp8(
                silu_out, self.block_size, dtype=w1.dtype, column_major_scales=True, scale_tma_aligned=True
            )

            # groupgemm (contiguous layout)
            gemm_out_b = torch.empty((all_tokens, K), device=device, dtype=hidden_dtype)

            _deepgemm_grouped_fp8_nt_contiguous((qsilu_out, qsilu_out_scale), (w2, w2_scale), gemm_out_b, m_indices)
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
        combined_x, event_overlap, hook = dist_group_manager.ep_buffer.low_latency_combine(
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
            async_finish=True,
            previous_event=overlap_event,
            allocate_on_comm_stream=True,
        )

        def hook():
            event.current_stream_wait()

        return combined_x, hook

    def _fuse(self):
        if self.quantized_weight:
            self._fuse_weight_scale()
        with self.lock:
            if (
                hasattr(self, "experts_up_projs")
                and None not in self.experts_up_projs
                and None not in self.experts_gate_projs
                and None not in self.w2_list
            ):
                gate_out_dim, gate_in_dim = self.experts_gate_projs[0].shape
                up_out_dim, up_in_dim = self.experts_up_projs[0].shape
                assert gate_in_dim == up_in_dim
                dtype = self.experts_gate_projs[0].dtype
                total_expert_num = self.ep_n_routed_experts + self.redundancy_expert_num

                w1 = torch.empty((total_expert_num, gate_out_dim + up_out_dim, gate_in_dim), dtype=dtype, device="cpu")

                for i_experts in range(self.ep_n_routed_experts + self.redundancy_expert_num):
                    w1[i_experts, 0:gate_out_dim:, :] = self.experts_gate_projs[i_experts]
                    w1[i_experts, gate_out_dim:, :] = self.experts_up_projs[i_experts]

                inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(len(self.w2_list), inter_shape, hidden_size)
                if not self.quantized_weight and self.quant_method is not None:
                    self.w1 = self.quant_method.quantize(w1)
                    self.w2 = self.quant_method.quantize(w2)
                else:
                    self.w1[0] = self._cuda(w1)
                    self.w2[0] = self._cuda(w2)
                delattr(self, "w2_list")
                delattr(self, "experts_up_projs")
                delattr(self, "experts_gate_projs")

    def _fuse_weight_scale(self):
        with self.lock:
            if (
                hasattr(self, "experts_up_proj_scales")
                and None not in self.experts_up_proj_scales
                and None not in self.experts_gate_proj_scales
                and None not in self.w2_scale_list
            ):
                gate_out_dim, gate_in_dim = self.experts_gate_proj_scales[0].shape
                up_out_dim, up_in_dim = self.experts_up_proj_scales[0].shape
                assert gate_in_dim == up_in_dim
                dtype = self.experts_gate_proj_scales[0].dtype
                total_expert_num = self.ep_n_routed_experts + self.redundancy_expert_num

                w1_scale = torch.empty(
                    (total_expert_num, gate_out_dim + up_out_dim, gate_in_dim), dtype=dtype, device="cpu"
                )

                for i_experts in range(self.ep_n_routed_experts + self.redundancy_expert_num):
                    w1_scale[i_experts, 0:gate_out_dim:, :] = self.experts_gate_proj_scales[i_experts]
                    w1_scale[i_experts, gate_out_dim:, :] = self.experts_up_proj_scales[i_experts]

                inter_shape, hidden_size = self.w2_scale_list[0].shape[0], self.w2_scale_list[0].shape[1]
                w2_scale = torch._utils._flatten_dense_tensors(self.w2_scale_list).view(
                    len(self.w2_scale_list), inter_shape, hidden_size
                )
                self.w1[1] = self._cuda(w1_scale)
                self.w2[1] = self._cuda(w2_scale)
                delattr(self, "w2_scale_list")
                delattr(self, "experts_up_proj_scales")
                delattr(self, "experts_gate_proj_scales")

    def load_hf_weights(self, weights):
        n_expert_ep = self.ep_n_routed_experts

        # Load bias
        if self.e_score_correction_bias_name in weights:
            self.e_score_correction_bias = self._cuda(weights[self.e_score_correction_bias_name])

        # Get weight shapes from first expert to determine intermediate size
        first_expert_idx = 0 + n_expert_ep * self.global_rank_
        w1_weight_name = f"{self.weight_prefix}.{first_expert_idx}.{self.w1_weight_name}.weight"
        if w1_weight_name in weights:
            intermediate_size = weights[w1_weight_name].shape[0]
            self.intermediate_size = intermediate_size

            # Re-create weights with correct size if needed
            if self.w1[0].shape[1] != intermediate_size * 2:
                self._create_weight()

        # Load regular experts
        for i_experts_ep in range(n_expert_ep):
            i_experts = i_experts_ep + n_expert_ep * self.global_rank_
            self._copy_expert_weights(i_experts_ep, i_experts, weights)

        # Load redundant experts
        for i, redundant_expert_id in enumerate(self.redundancy_expert_ids):
            self._copy_expert_weights(n_expert_ep + i, redundant_expert_id, weights)

        if self.quantized_weight:
            self._load_weight_scale_direct(weights)

    def _copy_expert_weights(self, target_idx, expert_id, weights):
        """Copy a single expert's weights to pre-allocated GPU memory"""
        w1_weight = f"{self.weight_prefix}.{expert_id}.{self.w1_weight_name}.weight"
        w2_weight = f"{self.weight_prefix}.{expert_id}.{self.w2_weight_name}.weight"
        w3_weight = f"{self.weight_prefix}.{expert_id}.{self.w3_weight_name}.weight"

        intermediate_size = self.intermediate_size

        if w1_weight in weights and w3_weight in weights:
            # Combine gate and up projections into w1
            gate_weight = weights[w1_weight]  # [intermediate_size, hidden_size]
            up_weight = weights[w3_weight]  # [intermediate_size, hidden_size]

            # Copy to pre-allocated memory
            if not self.quantized_weight and self.quant_method is not None:
                # Quantized path
                combined_cpu = torch.empty((intermediate_size * 2, self.hidden_size), dtype=gate_weight.dtype)
                combined_cpu[:intermediate_size, :] = gate_weight
                combined_cpu[intermediate_size:, :] = up_weight
                quantized_pack = self.quant_method.quantize(combined_cpu)
                self.w1[0][target_idx].copy_(quantized_pack.weight.view(intermediate_size * 2, self.hidden_size))
                if quantized_pack.weight_scale is not None:
                    self.w1[1][target_idx].copy_(
                        quantized_pack.weight_scale.view(intermediate_size * 2, self.hidden_size)
                    )
            else:
                # Regular path
                self.w1[0][target_idx, :intermediate_size, :].copy_(gate_weight)
                self.w1[0][target_idx, intermediate_size:, :].copy_(up_weight)

        if w2_weight in weights:
            # Copy w2 (down projection)
            w2_weight_tensor = weights[w2_weight]  # [hidden_size, intermediate_size] - already the correct shape
            if not self.quantized_weight and self.quant_method is not None:
                quantized_pack = self.quant_method.quantize(w2_weight_tensor)
                self.w2[0][target_idx].copy_(quantized_pack.weight)
                if quantized_pack.weight_scale is not None:
                    self.w2[1][target_idx].copy_(quantized_pack.weight_scale)
            else:
                self.w2[0][target_idx].copy_(w2_weight_tensor)

    def _load_weight_scale(self, weights: Dict[str, torch.Tensor]) -> None:
        n_expert_ep = self.ep_n_routed_experts
        for i_experts_ep in range(n_expert_ep):
            i_experts = i_experts_ep + n_expert_ep * self.global_rank_
            w1_scale = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.{self.weight_scale_suffix}"
            w2_scale = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.{self.weight_scale_suffix}"
            w3_scale = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.{self.weight_scale_suffix}"
            if w1_scale in weights:
                self.experts_gate_proj_scales[i_experts_ep] = weights[w1_scale]
            if w3_scale in weights:
                self.experts_up_proj_scales[i_experts_ep] = weights[w3_scale]

            if w2_scale in weights:
                self.w2_scale_list[i_experts_ep] = weights[w2_scale]

        # Load scale parameters for redundant experts
        for i, redundant_expert_id in enumerate(self.redundancy_expert_ids):
            i_experts = redundant_expert_id
            w1_scale = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.{self.weight_scale_suffix}"
            w2_scale = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.{self.weight_scale_suffix}"
            w3_scale = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.{self.weight_scale_suffix}"
            if w1_scale in weights:
                self.experts_gate_proj_scales[n_expert_ep + i] = weights[w1_scale]
            if w3_scale in weights:
                self.experts_up_proj_scales[n_expert_ep + i] = weights[w3_scale]
            if w2_scale in weights:
                self.w2_scale_list[n_expert_ep + i] = weights[w2_scale]

    def _load_weight_scale_direct(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load weight scales directly to pre-allocated GPU memory"""
        n_expert_ep = self.ep_n_routed_experts

        # Load regular expert scales
        for i_experts_ep in range(n_expert_ep):
            i_experts = i_experts_ep + n_expert_ep * self.global_rank_
            self._copy_expert_scales(i_experts_ep, i_experts, weights)

        # Load redundant expert scales
        for i, redundant_expert_id in enumerate(self.redundancy_expert_ids):
            self._copy_expert_scales(n_expert_ep + i, redundant_expert_id, weights)

    def _copy_expert_scales(self, target_idx, expert_id, weights):
        """Copy a single expert's weight scales to pre-allocated GPU memory"""
        w1_scale = f"{self.weight_prefix}.{expert_id}.{self.w1_weight_name}.{self.weight_scale_suffix}"
        w2_scale = f"{self.weight_prefix}.{expert_id}.{self.w2_weight_name}.{self.weight_scale_suffix}"
        w3_scale = f"{self.weight_prefix}.{expert_id}.{self.w3_weight_name}.{self.weight_scale_suffix}"

        intermediate_size = self.intermediate_size

        if w1_scale in weights and w3_scale in weights:
            # Combine gate and up projection scales into w1 scale
            gate_scale = weights[w1_scale]  # [intermediate_size, hidden_size]
            up_scale = weights[w3_scale]  # [intermediate_size, hidden_size]

            # Copy to pre-allocated memory
            self.w1[1][target_idx, :intermediate_size, :].copy_(gate_scale)
            self.w1[1][target_idx, intermediate_size:, :].copy_(up_scale)

        if w2_scale in weights:
            # Copy w2 scale (down projection)
            w2_scale_tensor = weights[w2_scale]  # [hidden_size, intermediate_size]
            self.w2[1][target_idx].copy_(w2_scale_tensor)

    def _cuda(self, cpu_tensor):
        device_id = get_current_device_id()
        if self.quantized_weight:
            return cpu_tensor.contiguous().cuda(device_id)
        return cpu_tensor.contiguous().to(self.data_type_).cuda(device_id)

    def verify_load(self):
        return self.w1 is not None and self.w2 is not None
