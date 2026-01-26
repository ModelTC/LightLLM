import torch
import threading
from typing import Dict, Any, Optional, Tuple, List
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_slicer import (
    get_row_slice_mixin,
    get_col_slice_mixin,
    SliceMixinTpl,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import select_fuse_moe_impl
from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.utils.envs_utils import get_redundancy_expert_ids, get_redundancy_expert_num, get_env_start_args
from lightllm.utils.dist_utils import get_global_world_size, get_global_rank
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class FusedMoeWeight(BaseWeightTpl):
    def __init__(
        self,
        gate_proj_name: str,
        down_proj_name: str,
        up_proj_name: str,
        e_score_correction_bias_name: str,
        weight_prefix: str,
        n_routed_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        data_type: torch.dtype,
        quant_method: QuantizationMethod = None,
        num_fused_shared_experts: int = 0,
        layer_num: int = 0,
        network_config: Dict[str, Any] = None,
    ) -> None:
        super().__init__(data_type=data_type)
        self.w1_weight_name = gate_proj_name
        self.w2_weight_name = down_proj_name
        self.w3_weight_name = up_proj_name
        self.e_score_correction_bias_name = e_score_correction_bias_name
        self.weight_prefix = weight_prefix
        self.layer_num_ = layer_num
        self.global_rank_ = get_global_rank()
        self.global_world_size = get_global_world_size()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.quant_method = quant_method
        self.row_slicer = get_row_slice_mixin(
            self.quant_method.method_name, tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_
        )
        self.col_slicer = get_col_slice_mixin(
            self.quant_method.method_name, tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_
        )
        assert num_fused_shared_experts in [0, 1], "num_fused_shared_experts can only support 0 or 1 now."
        self.enable_ep_moe = get_env_start_args().enable_ep_moe
        self.n_routed_experts = n_routed_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        self._init_config(network_config)
        self._init_redundancy_expert_params()
        self._init_parallel_params()
        self.fuse_moe_impl = select_fuse_moe_impl(self.quant_method, self.enable_ep_moe)(
            n_routed_experts=self.n_routed_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            quant_method=self.quant_method,
            redundancy_expert_num=self.redundancy_expert_num,
            redundancy_expert_ids_tensor=self.redundancy_expert_ids_tensor,
            routed_expert_counter_tensor=self.routed_expert_counter_tensor,
            auto_update_redundancy_expert=self.auto_update_redundancy_expert,
        )
        self.lock = threading.Lock()
        self._create_weight()

    def _init_config(self, network_config: Dict[str, Any]):
        self.n_group = network_config.get("n_group", 0)
        self.use_grouped_topk = self.n_group > 0
        self.norm_topk_prob = network_config["norm_topk_prob"]
        self.topk_group = network_config.get("topk_group", 0)
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.routed_scaling_factor = network_config.get("routed_scaling_factor", 1.0)
        self.scoring_func = network_config.get("scoring_func", "softmax")

    def _init_redundancy_expert_params(self):
        self.redundancy_expert_num = get_redundancy_expert_num()
        self.redundancy_expert_ids = get_redundancy_expert_ids(self.layer_num_)
        self.auto_update_redundancy_expert: bool = get_env_start_args().auto_update_redundancy_expert
        self.redundancy_expert_ids_tensor = torch.tensor(self.redundancy_expert_ids, dtype=torch.int64, device="cuda")
        self.routed_expert_counter_tensor = torch.zeros((self.n_routed_experts,), dtype=torch.int64, device="cuda")
        # TODO: find out the reason of failure of deepep when redundancy_expert_num is 1.
        assert self.redundancy_expert_num != 1, "redundancy_expert_num can not be 1 for some unknown hang of deepep."

    def _init_parallel_params(self):
        self.local_n_routed_experts = self.n_routed_experts + self.num_fused_shared_experts
        self.split_inter_size = self.moe_intermediate_size // self.tp_world_size_
        if self.enable_ep_moe:
            assert self.num_fused_shared_experts == 0, "num_fused_shared_experts must be 0 when enable_ep_moe"
            logger.info(
                f"global_rank {self.global_rank_} layerindex {self.layer_num_} "
                f"redundancy_expertids: {self.redundancy_expert_ids}"
            )
            self.local_n_routed_experts = self.n_routed_experts // self.global_world_size + self.redundancy_expert_num
            self.split_inter_size = self.moe_intermediate_size
            n_experts_per_rank = self.n_routed_experts // self.global_world_size
            start_expert_id = self.global_rank_ * n_experts_per_rank
            self.local_expert_ids = (
                list(range(start_expert_id, start_expert_id + n_experts_per_rank)) + self.redundancy_expert_ids
            )
            self.expert_idx_to_local_idx = {
                expert_idx: expert_idx - start_expert_id for expert_idx in self.local_expert_ids[:n_experts_per_rank]
            }
            self.redundancy_expert_idx_to_local_idx = {
                redundancy_expert_idx: n_experts_per_rank + i
                for (i, redundancy_expert_idx) in enumerate(self.redundancy_expert_ids)
            }
        else:
            self.local_expert_ids = list(range(self.n_routed_experts + self.num_fused_shared_experts))
            self.expert_idx_to_local_idx = {expert_idx: i for (i, expert_idx) in enumerate(self.local_expert_ids)}
            self.rexpert_idx_to_local_idx = {}

    def experts(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        is_prefill: Optional[bool] = None,
    ):
        """Backward compatible method that routes to platform-specific implementation."""
        return self.fuse_moe_impl(
            input_tensor=input_tensor,
            router_logits=router_logits,
            w13=self.w13,
            w2=self.w2,
            correction_bias=self.e_score_correction_bias,
            scoring_func=self.scoring_func,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            is_prefill=is_prefill,
        )

    def low_latency_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        assert self.enable_ep_moe, "low_latency_dispatch is only supported when enable_ep_moe is True"
        return self.fuse_moe_impl.low_latency_dispatch(
            hidden_states=hidden_states,
            router_logits=router_logits,
            e_score_correction_bias=self.e_score_correction_bias,
            use_grouped_topk=self.use_grouped_topk,
            num_experts_per_tok=self.num_experts_per_tok,
            norm_topk_prob=self.norm_topk_prob,
            topk_group=self.topk_group,
            n_group=self.n_group,
            scoring_func=self.scoring_func,
        )

    def select_experts_and_quant_input(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        assert self.enable_ep_moe, "select_experts_and_quant_input is only supported when enable_ep_moe is True"
        return self.fuse_moe_impl.select_experts_and_quant_input(
            hidden_states=hidden_states,
            router_logits=router_logits,
            e_score_correction_bias=self.e_score_correction_bias,
            w13=self.w13,
            use_grouped_topk=self.use_grouped_topk,
            num_experts_per_tok=self.num_experts_per_tok,
            norm_topk_prob=self.norm_topk_prob,
            topk_group=self.topk_group,
            n_group=self.n_group,
            scoring_func=self.scoring_func,
        )

    def dispatch(
        self,
        qinput_tensor: Tuple[torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_event: Optional[Any] = None,
    ):
        assert self.enable_ep_moe, "dispatch is only supported when enable_ep_moe is True"
        return self.fuse_moe_impl.dispatch(
            qinput_tensor=qinput_tensor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            overlap_event=overlap_event,
        )

    def masked_group_gemm(
        self, recv_x: Tuple[torch.Tensor], masked_m: torch.Tensor, dtype: torch.dtype, expected_m: int
    ):
        assert self.enable_ep_moe, "masked_group_gemm is only supported when enable_ep_moe is True"
        return self.fuse_moe_impl.masked_group_gemm(
            recv_x=recv_x,
            w13=self.w13,
            w2=self.w2,
            masked_m=masked_m,
            dtype=dtype,
            expected_m=expected_m,
        )

    def prefilled_group_gemm(
        self,
        num_recv_tokens_per_expert_list,
        recv_x: Tuple[torch.Tensor],
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        hidden_dtype=torch.bfloat16,
    ):
        assert self.enable_ep_moe, "prefilled_group_gemm is only supported when enable_ep_moe is True"
        return self.fuse_moe_impl.prefilled_group_gemm(
            num_recv_tokens_per_expert_list=num_recv_tokens_per_expert_list,
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            w13=self.w13,
            w2=self.w2,
            hidden_dtype=hidden_dtype,
        )

    def low_latency_combine(
        self,
        gemm_out_b: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Any,
    ):
        assert self.enable_ep_moe, "low_latency_combine is only supported when enable_ep_moe is True"
        return self.fuse_moe_impl.low_latency_combine(
            gemm_out_b=gemm_out_b,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=handle,
        )

    def combine(
        self,
        gemm_out_b: torch.Tensor,
        handle: Any,
        overlap_event: Optional[Any] = None,
    ):
        assert self.enable_ep_moe, "combine is only supported when enable_ep_moe is True"
        return self.fuse_moe_impl.combine(
            gemm_out_b=gemm_out_b,
            handle=handle,
            overlap_event=overlap_event,
        )

    def load_hf_weights(self, weights):
        # Load bias
        if self.e_score_correction_bias_name in weights:
            self.e_score_correction_bias.copy_(weights[self.e_score_correction_bias_name])
        self._load_weight(self.expert_idx_to_local_idx, weights)
        if self.redundancy_expert_num > 0:
            self._load_weight(self.redundancy_expert_idx_to_local_idx, weights)

    def verify_load(self):
        return True

    def _create_weight(self):
        intermediate_size = self.split_inter_size
        self.e_score_correction_bias = None
        # Create e_score_correction_bias
        if self.e_score_correction_bias_name:
            self.e_score_correction_bias = torch.empty(
                (self.n_routed_experts,),
                dtype=self.data_type_,
                device=f"cuda:{self.device_id_}",
            )

        self.w13: WeightPack = self.quant_method.create_weight(
            out_dim=intermediate_size * 2,
            in_dim=self.hidden_size,
            dtype=self.data_type_,
            device_id=self.device_id_,
            num_experts=self.local_n_routed_experts,
        )
        self.w13_list: List[WeightPack] = self._get_expert_weight_list(self.w13, 2)
        self.w2: WeightPack = self.quant_method.create_weight(
            out_dim=self.hidden_size,
            in_dim=intermediate_size,
            dtype=self.data_type_,
            device_id=self.device_id_,
            num_experts=self.local_n_routed_experts,
        )
        self.w2_list: List[WeightPack] = self._get_expert_weight_list(self.w2, 1)
        self.load_cnt = 0

    def _get_expert_weight_list(self, weight_pack: WeightPack, weight_num: int = 1):
        weight_list = []
        for idx in range(self.local_n_routed_experts):
            expert_weight = weight_pack.get_expert(idx)
            expert_weight.create_cpu_buffer(weight_num)
            weight_list.append(expert_weight)
        return weight_list

    def _load_weight(self, expert_idx_to_local_idx: Dict[int, int], weights: Dict[str, torch.Tensor]):

        # Load each expert with TP slicing
        for expert_idx, local_expert_idx in expert_idx_to_local_idx.items():
            with self.lock:
                self._load_expert(
                    expert_idx, local_expert_idx, weights, type="weight", suffix=self.quant_method.weight_suffix
                )
            if self.w13.weight_scale is not None:
                with self.lock:
                    self._load_expert(
                        expert_idx,
                        local_expert_idx,
                        weights,
                        type="weight_scale",
                        suffix=self.quant_method.weight_scale_suffix,
                    )
            if self.w13.weight_zero_point is not None:
                with self.lock:
                    self._load_expert(
                        expert_idx,
                        local_expert_idx,
                        weights,
                        type="weight_zero_point",
                        suffix=self.quant_method.weight_zero_point_suffix,
                    )

    def _load_expert(
        self,
        expert_idx: int,
        local_expert_idx: int,
        weights: Dict[str, torch.Tensor],
        type: str,
        suffix: str = "weight",
    ):
        w1_weight = f"{self.weight_prefix}.{expert_idx}.{self.w1_weight_name}.{suffix}"
        w2_weight = f"{self.weight_prefix}.{expert_idx}.{self.w2_weight_name}.{suffix}"
        w3_weight = f"{self.weight_prefix}.{expert_idx}.{self.w3_weight_name}.{suffix}"
        load_func = self._get_load_func(type)
        row_slice_func = self._get_slice_func(self.row_slicer, type)
        col_slice_func = self._get_slice_func(self.col_slicer, type)
        if w1_weight in weights:
            self.w13_list[local_expert_idx].weight_cpu_buffer[0] = row_slice_func(weights[w1_weight])
        if w3_weight in weights:
            self.w13_list[local_expert_idx].weight_cpu_buffer[1] = row_slice_func(weights[w3_weight])
        w13_weight = self.w13_list[local_expert_idx].get_fused_weight_part(suffix)
        load_func(w13_weight, self.w13_list[local_expert_idx])
        if w2_weight in weights:
            self.w2_list[local_expert_idx].weight_cpu_buffer[0] = col_slice_func(weights[w2_weight])
        w2_weight = self.w2_list[local_expert_idx].get_fused_weight_part(suffix)
        load_func(w2_weight, self.w2_list[local_expert_idx])

    def _load_weight_func(self, weight: torch.Tensor, weight_pack: WeightPack):
        if self.quant_method.weight_need_quanted(weight):
            self.quant_method.quantize(weight, weight_pack)
        else:
            self.quant_method.load_weight(weight, weight_pack)

    def _get_load_func(self, type: str):
        if type == "weight":
            return self._load_weight_func
        elif type == "weight_scale":
            return getattr(self.quant_method, "load_weight_scale")
        elif type == "weight_zero_point":
            return getattr(self.quant_method, "load_weight_zero_point")

    def _get_slice_func(self, slicer: SliceMixinTpl, type: str):
        if type == "weight":
            return slicer._slice_weight
        elif type == "weight_scale":
            return slicer._slice_weight_scale
        elif type == "weight_zero_point":
            return slicer._slice_weight_zero_point
