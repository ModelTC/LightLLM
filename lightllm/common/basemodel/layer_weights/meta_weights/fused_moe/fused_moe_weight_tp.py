import os
import torch
import threading
from typing import Tuple, List, Dict, Any, Union, Callable
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_current_device_id, get_dp_world_size
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_slicer import (
    get_row_slice_mixin,
    get_col_slice_mixin,
)
from threading import Lock


def create_tp_moe_wegiht_obj(
    gate_proj_name: str,
    down_proj_name: str,
    up_proj_name: str,
    e_score_correction_bias_name: str,
    weight_prefix: str,
    n_routed_experts: int,
    num_fused_shared_experts: int,
    split_inter_size: int,
    data_type: torch.dtype,
    network_config: Dict[str, Any],
    layer_num: int,
    quant_cfg: Quantcfg = None,
) -> Union["FusedMoeWeightTP", "FusedAWQMARLINMoeWeightTP"]:
    quant_method = quant_cfg.get_quant_method(layer_num, "fused_moe")
    if quant_method is not None and quant_method.method_name == "awq_marlin":
        return FusedAWQMARLINMoeWeightTP(
            gate_proj_name=gate_proj_name,
            down_proj_name=down_proj_name,
            up_proj_name=up_proj_name,
            e_score_correction_bias_name=e_score_correction_bias_name,
            weight_prefix=weight_prefix,
            n_routed_experts=n_routed_experts,
            num_fused_shared_experts=num_fused_shared_experts,
            split_inter_size=split_inter_size,
            data_type=data_type,
            network_config=network_config,
            layer_num=layer_num,
            quant_cfg=quant_cfg,
        )
    else:
        return FusedMoeWeightTP(
            gate_proj_name=gate_proj_name,
            down_proj_name=down_proj_name,
            up_proj_name=up_proj_name,
            e_score_correction_bias_name=e_score_correction_bias_name,
            weight_prefix=weight_prefix,
            n_routed_experts=n_routed_experts,
            num_fused_shared_experts=num_fused_shared_experts,
            split_inter_size=split_inter_size,
            data_type=data_type,
            network_config=network_config,
            layer_num=layer_num,
            quant_cfg=quant_cfg,
        )


class FusedMoeWeightTP(BaseWeight):
    def __init__(
        self,
        gate_proj_name: str,
        down_proj_name: str,
        up_proj_name: str,
        e_score_correction_bias_name: str,
        weight_prefix: str,
        n_routed_experts: int,
        num_fused_shared_experts: int,
        split_inter_size: int,
        data_type: torch.dtype,
        network_config: Dict[str, Any],
        layer_num: int,
        quant_cfg: Quantcfg = None,
    ) -> None:
        super().__init__()
        self.quant_method = quant_cfg.get_quant_method(layer_num, "fused_moe")
        self.quantized_weight = quant_cfg.quantized_weight
        if self.quant_method.method_name != "none":
            self.weight_scale_suffix = self.quant_method.weight_scale_suffix
            self.quant_method.is_moe = True

        self.w1_weight_name = gate_proj_name
        self.w2_weight_name = down_proj_name
        self.w3_weight_name = up_proj_name

        self.e_score_correction_bias_name = e_score_correction_bias_name
        self.weight_prefix = weight_prefix
        assert num_fused_shared_experts in [0, 1], "num_fused_shared_experts can only support 0 or 1 now."
        self.n_routed_experts = n_routed_experts + num_fused_shared_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        self.routed_scaling_factor = network_config.get("routed_scaling_factor", 1.0)
        self.split_inter_size = split_inter_size
        self.data_type_ = data_type
        self.hidden_size = network_config.get("hidden_size")
        self.tp_rank_ = get_current_rank_in_dp()
        self.e_score_correction_bias = None
        self.scoring_func = network_config.get("scoring_func", "softmax")
        self.row_slicer = get_row_slice_mixin(
            self.quant_method.method_name, tp_rank=self.tp_rank_, tp_world_size=get_dp_world_size()
        )
        self.col_slicer = get_col_slice_mixin(
            self.quant_method.method_name, tp_rank=self.tp_rank_, tp_world_size=get_dp_world_size()
        )
        self.lock = Lock()
        # for online per-tensor quantization
        self.gate_up_buffer = [[None, None] for _ in range(self.n_routed_experts)]
        self._create_weight()

    def _create_weight(self):
        total_expert_num = self.n_routed_experts
        intermediate_size = self.split_inter_size
        device_id = get_current_device_id()

        # Create e_score_correction_bias
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias = torch.empty(
                (total_expert_num,),
                dtype=self.data_type_,
                device=f"cuda:{device_id}",
            )

        self.w13: WeightPack = self.quant_method.create_weight(
            out_dim=intermediate_size * 2,
            in_dim=self.hidden_size,
            dtype=self.data_type_,
            device_id=device_id,
            num_experts=total_expert_num,
        )
        self.w2: WeightPack = self.quant_method.create_weight(
            out_dim=self.hidden_size,
            in_dim=intermediate_size,
            dtype=self.data_type_,
            device_id=device_id,
            num_experts=total_expert_num,
        )

    def experts(self, input_tensor, router_logits, top_k, renormalize, use_grouped_topk, topk_group, num_expert_group):
        from lightllm.common.fused_moe.topk_select import select_experts

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
        if self.num_fused_shared_experts > 0:
            pad_topk_ids = (
                torch.arange(
                    start=self.n_routed_experts - self.num_fused_shared_experts,
                    end=self.n_routed_experts,
                    step=1,
                    dtype=topk_ids.dtype,
                    device="cuda",
                )
                .view(1, self.num_fused_shared_experts)
                .repeat(topk_ids.shape[0], 1)
            )
            pad_topk_weights = torch.full(
                (topk_weights.shape[0], self.num_fused_shared_experts),
                fill_value=1.0,
                device="cuda",
                dtype=topk_weights.dtype,
            )

            topk_ids = torch.cat([topk_ids, pad_topk_ids], dim=1)
            topk_weights = torch.cat([topk_weights, pad_topk_weights], dim=1)

        w13, w13_scale = self.w13.weight, self.w13.weight_scale
        w2, w2_scale = self.w2.weight, self.w2.weight_scale
        use_fp8_w8a8 = self.quant_method.method_name != "none"

        from lightllm.common.fused_moe.grouped_fused_moe import fused_experts

        fused_experts(
            hidden_states=input_tensor,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
        )
        return

    def _cuda(self, cpu_tensor):
        device_id = get_current_device_id()
        if self.quantized_weight:
            return cpu_tensor.cuda(device_id)
        return cpu_tensor.cuda(device_id)

    def verify_load(self):
        return True

    def load_hf_weights(self, weights):
        # Load bias
        if self.e_score_correction_bias_name in weights:
            self.e_score_correction_bias.copy_(weights[self.e_score_correction_bias_name])

        # Load each expert with TP slicing
        for i_experts in range(self.n_routed_experts):
            self._load_expert(i_experts, weights, type="weight", suffix=self.quant_method.weight_suffix)
            if self.w13.weight_scale is not None and self.quantized_weight:
                self._load_expert(i_experts, weights, type="weight_scale", suffix=self.quant_method.weight_scale_suffix)
            if self.w13.weight_zero_point is not None and self.quantized_weight:
                self._load_expert(
                    i_experts, weights, type="weight_zero_point", suffix=self.quant_method.weight_zero_point_suffix
                )

    def _load_weight_func(self, weight: torch.Tensor, weight_pack: WeightPack, start_idx: int = 0):
        if self.quant_method.weight_need_quanted(weight):
            self.quant_method.quantize_moe(weight, weight_pack, start_idx)
        else:
            self.quant_method.load_weight(weight, weight_pack, start_idx)

    def _load_expert(self, expert_idx, weights, type: str, suffix: str = "weight"):
        w1_weight = f"{self.weight_prefix}.{expert_idx}.{self.w1_weight_name}.{suffix}"
        w2_weight = f"{self.weight_prefix}.{expert_idx}.{self.w2_weight_name}.{suffix}"
        w3_weight = f"{self.weight_prefix}.{expert_idx}.{self.w3_weight_name}.{suffix}"

        merge_w1_weight = f"{self.weight_prefix}.{self.w1_weight_name}.{suffix}"
        merge_w2_weight = f"{self.weight_prefix}.{self.w2_weight_name}.{suffix}"
        merge_w3_weight = f"{self.weight_prefix}.{self.w3_weight_name}.{suffix}"

        intermediate_size = self.split_inter_size
        load_func, slice_func = self._get_load_and_slice_func(type, is_row=True)
        if suffix == "weight":
            with self.lock:
                if w1_weight in weights:
                    self.gate_up_buffer[expert_idx][0] = slice_func(weights[w1_weight])
                if merge_w1_weight in weights:
                    self.gate_up_buffer[expert_idx][0] = slice_func(weights[merge_w1_weight][expert_idx])
                if w3_weight in weights:
                    self.gate_up_buffer[expert_idx][1] = slice_func(weights[w3_weight])
                if merge_w3_weight in weights:
                    self.gate_up_buffer[expert_idx][1] = slice_func(weights[merge_w3_weight][expert_idx])
                if None not in self.gate_up_buffer[expert_idx]:
                    tmp_weight = torch.cat(
                        [self.gate_up_buffer[expert_idx][0], self.gate_up_buffer[expert_idx][1]], dim=0
                    )
                    load_func(tmp_weight, self.w13.get_expert(expert_idx), start_idx=0)
                    self.gate_up_buffer[expert_idx] = [None, None]
        else:
            if w1_weight in weights:
                load_func(slice_func(weights[w1_weight]), self.w13.get_expert(expert_idx), start_idx=0)
            if w3_weight in weights:
                load_func(slice_func(weights[w3_weight]), self.w13.get_expert(expert_idx), start_idx=intermediate_size)

        load_func, slice_func = self._get_load_and_slice_func(type, is_row=False)
        if w2_weight in weights:
            load_func(slice_func(weights[w2_weight]), self.w2.get_expert(expert_idx), start_idx=0)
        if merge_w2_weight in weights:
            load_func(slice_func(weights[merge_w2_weight][expert_idx]), self.w2.get_expert(expert_idx), start_idx=0)

    def _get_load_and_slice_func(self, type: str, is_row: bool = True):
        if is_row:
            slicer = self.row_slicer
        else:
            slicer = self.col_slicer
        if type == "weight":
            return self._load_weight_func, slicer._slice_weight
        elif type == "weight_scale":
            return getattr(self.quant_method, "load_weight_scale"), slicer._slice_weight_scale
        elif type == "weight_zero_point":
            return getattr(self.quant_method, "load_weight_zero_point"), slicer._slice_weight_zero_point


class FusedAWQMARLINMoeWeightTP(FusedMoeWeightTP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops

        assert HAS_VLLM, "moe awq marlin quantization requires kernels of vllm"
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
        )

        self.workspace = marlin_make_workspace_new(self.w13.weight.device, 4)

    def experts(self, input_tensor, router_logits, top_k, renormalize, use_grouped_topk, topk_group, num_expert_group):
        from lightllm.common.fused_moe.topk_select import select_experts

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
        if self.num_fused_shared_experts > 0:
            pad_topk_ids = (
                torch.arange(
                    start=self.n_routed_experts - self.num_fused_shared_experts,
                    end=self.n_routed_experts,
                    step=1,
                    dtype=topk_ids.dtype,
                    device="cuda",
                )
                .view(1, self.num_fused_shared_experts)
                .repeat(topk_ids.shape[0], 1)
            )
            pad_topk_weights = torch.full(
                (topk_weights.shape[0], self.num_fused_shared_experts),
                fill_value=1.0,
                device="cuda",
                dtype=topk_weights.dtype,
            )

            topk_ids = torch.cat([topk_ids, pad_topk_ids], dim=1)
            topk_weights = torch.cat([topk_weights, pad_topk_weights], dim=1)

        w1, w1_scale, w1_zero_point = self.w13.weight, self.w13.weight_scale, self.w13.weight_zero_point
        w2, w2_scale, w2_zero_point = self.w2.weight, self.w2.weight_scale, self.w2.weight_zero_point

        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe

        fused_marlin_moe(
            input_tensor,
            w1,
            w2,
            None,
            None,
            w1_scale,
            w2_scale,
            router_logits,
            topk_weights,
            topk_ids,
            quant_type_id=self.quant_method.vllm_quant_type.id,
            apply_router_weight_on_input=False,
            global_num_experts=-1,
            expert_map=None,
            w1_zeros=w1_zero_point,
            w2_zeros=w2_zero_point,
            workspace=self.workspace,
            inplace=True,
        )

        return
