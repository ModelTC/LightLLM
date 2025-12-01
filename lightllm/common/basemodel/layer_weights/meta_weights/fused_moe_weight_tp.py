import os
import torch
import threading
from typing import Optional, Tuple, List, Dict, Any, Union
from .base_weight import BaseWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_current_device_id
from lightllm.common.quantization import Quantcfg


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
        if self.quant_method is not None:
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
        self.tp_rank_ = get_current_rank_in_dp()
        self.experts_up_projs = [None] * self.n_routed_experts
        self.experts_gate_projs = [None] * self.n_routed_experts
        self.experts_up_proj_scales = [None] * self.n_routed_experts
        self.experts_gate_proj_scales = [None] * self.n_routed_experts
        self.e_score_correction_bias = None
        self.w2_list = [None] * self.n_routed_experts
        self.w2_scale_list = [None] * self.n_routed_experts
        self.scoring_func = network_config.get("scoring_func", "softmax")
        self.w1 = [None, None]  # weight, weight_scale
        self.w2 = [None, None]  # weight, weight_scale
        self.lock = threading.Lock()

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

        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        use_fp8_w8a8 = self.quant_method is not None

        from lightllm.common.fused_moe.grouped_fused_moe import fused_experts

        fused_experts(
            hidden_states=input_tensor,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
        return

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
                total_expert_num = self.n_routed_experts

                w1 = torch.empty((total_expert_num, gate_out_dim + up_out_dim, gate_in_dim), dtype=dtype, device="cpu")

                for i_experts in range(self.n_routed_experts):
                    w1[i_experts, 0:gate_out_dim:, :] = self.experts_gate_projs[i_experts]
                    w1[i_experts, gate_out_dim:, :] = self.experts_up_projs[i_experts]

                inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(len(self.w2_list), inter_shape, hidden_size)
                if not self.quantized_weight and self.quant_method is not None:
                    qw1, qw1_scale, qw1_zero_point = self.quant_method.quantize(w1)
                    qw2, qw2_scale, qw2_zero_point = self.quant_method.quantize(w2)
                    self.w1[0] = qw1
                    self.w1[1] = qw1_scale
                    self.w2[0] = qw2
                    self.w2[1] = qw2_scale
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
                total_expert_num = self.n_routed_experts

                w1_scale = torch.empty(
                    (total_expert_num, gate_out_dim + up_out_dim, gate_in_dim), dtype=dtype, device="cpu"
                )

                for i_experts in range(self.n_routed_experts):
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
        if self.e_score_correction_bias_name in weights:
            self.e_score_correction_bias = self._cuda(weights[self.e_score_correction_bias_name])
        for i_experts in range(self.n_routed_experts):
            w1_weight = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.weight"
            w2_weight = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.weight"
            w3_weight = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.weight"

            if w1_weight in weights:
                self.experts_gate_projs[i_experts] = weights[w1_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]
            if w3_weight in weights:
                self.experts_up_projs[i_experts] = weights[w3_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]

            if w2_weight in weights:
                self.w2_list[i_experts] = weights[w2_weight][
                    :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1)
                ]
        if self.quant_method is not None:
            self._load_weight_scale(weights)
        self._fuse()

    def _load_weight_scale(self, weights: Dict[str, torch.Tensor]) -> None:
        block_size = 1
        if hasattr(self.quant_method, "block_size"):
            block_size = self.quant_method.block_size
        for i_experts in range(self.n_routed_experts):
            w1_scale = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.{self.weight_scale_suffix}"
            w2_scale = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.{self.weight_scale_suffix}"
            w3_scale = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.{self.weight_scale_suffix}"
            if w1_scale in weights:
                self.experts_gate_proj_scales[i_experts] = weights[w1_scale][
                    self.split_inter_size
                    // block_size
                    * self.tp_rank_ : self.split_inter_size
                    // block_size
                    * (self.tp_rank_ + 1),
                    :,
                ]
            if w3_scale in weights:
                self.experts_up_proj_scales[i_experts] = weights[w3_scale][
                    self.split_inter_size
                    // block_size
                    * self.tp_rank_ : self.split_inter_size
                    // block_size
                    * (self.tp_rank_ + 1),
                    :,
                ]

            if w2_scale in weights:
                self.w2_scale_list[i_experts] = weights[w2_scale][
                    :,
                    self.split_inter_size
                    // block_size
                    * self.tp_rank_ : self.split_inter_size
                    // block_size
                    * (self.tp_rank_ + 1),
                ]

    def _cuda(self, cpu_tensor):
        device_id = get_current_device_id()
        if self.quantized_weight:
            return cpu_tensor.contiguous().cuda(device_id)
        return cpu_tensor.contiguous().to(self.data_type_).cuda(device_id)

    def verify_load(self):
        return self.w1 is not None and self.w2 is not None


class FusedAWQMARLINMoeWeightTP(BaseWeight):
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
        if self.quant_method is not None:
            self.weight_scale_suffix = self.quant_method.weight_scale_suffix
            self.weight_zero_point_suffix = self.quant_method.weight_zero_point_suffix
            self.quant_method.is_moe = True
        hf_quantization_config = network_config.get("quantization_config", None)
        self.num_bits = hf_quantization_config.get("bits", 4)
        self.group_size = hf_quantization_config.get("group_size", 128)
        self.pack_factor = 32 // self.num_bits
        self.has_processed_weight = False
        assert self.quant_method.method_name == "awq_marlin"

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
        self.tp_rank_ = get_current_rank_in_dp()
        self.experts_up_projs = [None] * self.n_routed_experts
        self.experts_gate_projs = [None] * self.n_routed_experts
        self.experts_up_proj_scales = [None] * self.n_routed_experts
        self.experts_up_proj_zero_points = [None] * self.n_routed_experts
        self.experts_gate_proj_scales = [None] * self.n_routed_experts
        self.experts_gate_proj_zero_points = [None] * self.n_routed_experts
        self.e_score_correction_bias = None
        self.w2_list = [None] * self.n_routed_experts
        self.w2_scale_list = [None] * self.n_routed_experts
        self.w2_zero_point_list = [None] * self.n_routed_experts
        self.scoring_func = network_config.get("scoring_func", "softmax")
        self.w1 = [None, None, None]  # weight, weight_scale, zero_point
        self.w2 = [None, None, None]  # weight, weight_scale, zero_point
        self.lock = threading.Lock()

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

        w1, w1_scale, w1_zero_point = self.w1
        w2, w2_scale, w2_zero_point = self.w2

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

    def _fuse(self):
        self._fuse_weight()
        self._fuse_weight_scale()
        self._fuse_weight_zero_point()

    def _fuse_weight(self):
        with self.lock:
            if (
                hasattr(self, "experts_up_projs")
                and None not in self.experts_up_projs
                and None not in self.experts_gate_projs
                and None not in self.w2_list
            ):
                gate_in_dim, gate_out_dim = self.experts_gate_projs[0].shape
                up_in_dim, up_out_dim = self.experts_up_projs[0].shape
                assert gate_in_dim == up_in_dim
                total_expert_num = self.n_routed_experts

                w1 = torch.empty(
                    (total_expert_num, gate_in_dim, gate_out_dim + up_out_dim), dtype=torch.int32, device="cpu"
                )

                for i_experts in range(self.n_routed_experts):
                    w1[i_experts, :, 0:gate_out_dim] = self.experts_gate_projs[i_experts]
                    w1[i_experts, :, gate_out_dim:] = self.experts_up_projs[i_experts]

                inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(len(self.w2_list), inter_shape, hidden_size)
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
                gate_in_dim, gate_out_dim = self.experts_gate_proj_scales[0].shape
                up_in_dim, up_out_dim = self.experts_up_proj_scales[0].shape
                dtype = self.experts_gate_proj_scales[0].dtype
                assert gate_in_dim == up_in_dim
                total_expert_num = self.n_routed_experts
                w1_scale = torch.empty(
                    (total_expert_num, gate_in_dim, gate_out_dim + up_out_dim), dtype=dtype, device="cpu"
                )
                for i_experts in range(self.n_routed_experts):
                    w1_scale[i_experts, :, 0:gate_out_dim] = self.experts_gate_proj_scales[i_experts]
                    w1_scale[i_experts, :, gate_out_dim:] = self.experts_up_proj_scales[i_experts]
                inter_shape, hidden_size = self.w2_scale_list[0].shape[0], self.w2_scale_list[0].shape[1]
                w2_scale = torch._utils._flatten_dense_tensors(self.w2_scale_list).view(
                    len(self.w2_scale_list), inter_shape, hidden_size
                )
                self.w1[1] = self._cuda(w1_scale).to(self.data_type_)
                self.w2[1] = self._cuda(w2_scale).to(self.data_type_)
                delattr(self, "w2_scale_list")
                delattr(self, "experts_up_proj_scales")
                delattr(self, "experts_gate_proj_scales")

    def _fuse_weight_zero_point(self):
        with self.lock:
            if (
                hasattr(self, "experts_up_proj_zero_points")
                and None not in self.experts_up_proj_zero_points
                and None not in self.experts_gate_proj_zero_points
                and None not in self.w2_zero_point_list
            ):
                gate_in_dim, gate_out_dim = self.experts_gate_proj_zero_points[0].shape
                up_in_dim, up_out_dim = self.experts_up_proj_zero_points[0].shape
                assert gate_in_dim == up_in_dim
                total_expert_num = self.n_routed_experts
                w1_zero_point = torch.empty(
                    (total_expert_num, gate_in_dim, gate_out_dim + up_out_dim), dtype=torch.int32, device="cpu"
                )
                for i_experts in range(self.n_routed_experts):
                    w1_zero_point[i_experts, :, 0:gate_out_dim] = self.experts_gate_proj_zero_points[i_experts]
                    w1_zero_point[i_experts, :, gate_out_dim:] = self.experts_up_proj_zero_points[i_experts]
                inter_shape, hidden_size = self.w2_zero_point_list[0].shape[0], self.w2_zero_point_list[0].shape[1]
                w2_zero_point = torch._utils._flatten_dense_tensors(self.w2_zero_point_list).view(
                    len(self.w2_zero_point_list), inter_shape, hidden_size
                )
                self.w1[2] = self._cuda(w1_zero_point)
                self.w2[2] = self._cuda(w2_zero_point)
                delattr(self, "w2_zero_point_list")
                delattr(self, "experts_up_proj_zero_points")
                delattr(self, "experts_gate_proj_zero_points")

    def load_hf_weights(self, weights):
        self._load_weight(weights)
        self._load_weight_scale(weights)
        self._load_weight_zero_point(weights)
        self._fuse()
        self._process_weight_after_loading()

    def _load_weight(self, weights: Dict[str, torch.Tensor]) -> None:
        # awq quantization weight shape: in x out
        if self.e_score_correction_bias_name in weights:
            self.e_score_correction_bias = self._cuda(weights[self.e_score_correction_bias_name])
        for i_experts in range(self.n_routed_experts):
            w1_weight = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.qweight"
            w2_weight = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.qweight"
            w3_weight = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.qweight"

            if w1_weight in weights:
                self.experts_gate_projs[i_experts] = weights[w1_weight][
                    :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1)
                ]
            if w3_weight in weights:
                self.experts_up_projs[i_experts] = weights[w3_weight][
                    :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1)
                ]

            if w2_weight in weights:
                self.w2_list[i_experts] = weights[w2_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]

    def _load_weight_scale(self, weights: Dict[str, torch.Tensor]) -> None:
        for i_experts in range(self.n_routed_experts):
            w1_scale = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.{self.weight_scale_suffix}"
            w2_scale = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.{self.weight_scale_suffix}"
            w3_scale = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.{self.weight_scale_suffix}"
            split_inter_size = self.split_inter_size * self.pack_factor
            if w1_scale in weights:
                self.experts_gate_proj_scales[i_experts] = weights[w1_scale][
                    :,
                    split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1),
                ]
            if w3_scale in weights:
                self.experts_up_proj_scales[i_experts] = weights[w3_scale][
                    :,
                    split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1),
                ]

            if w2_scale in weights:
                self.w2_scale_list[i_experts] = weights[w2_scale][
                    split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1),
                    :,
                ]

    def _load_weight_zero_point(self, weights: Dict[str, torch.Tensor]) -> None:
        for i_experts in range(self.n_routed_experts):
            w1_zero_point = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.{self.weight_zero_point_suffix}"
            w2_zero_point = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.{self.weight_zero_point_suffix}"
            w3_zero_point = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.{self.weight_zero_point_suffix}"
            if w1_zero_point in weights:
                self.experts_gate_proj_zero_points[i_experts] = weights[w1_zero_point][
                    :,
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1),
                ]
            if w3_zero_point in weights:
                self.experts_up_proj_zero_points[i_experts] = weights[w3_zero_point][
                    :,
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1),
                ]
            if w2_zero_point in weights:
                self.w2_zero_point_list[i_experts] = weights[w2_zero_point][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1),
                    :,
                ]

    def _process_weight_after_loading(self):
        with self.lock:
            if None in self.w1 or None in self.w2 or self.has_processed_weight:
                return
        self.has_processed_weight = True
        from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops

        assert HAS_VLLM, "moe awq marlin quantization requires kernels of vllm"

        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_moe_permute_scales,
            moe_awq_to_marlin_zero_points,
            marlin_make_workspace_new,
        )

        num_experts = self.n_routed_experts
        device = self.w1[0].device

        self.w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        self.w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        self.w1[0] = vllm_ops.awq_marlin_moe_repack(
            self.w1[0],
            self.w13_g_idx_sort_indices,
            size_k=self.w1[0].shape[1],
            size_n=self.w1[0].shape[2] * self.pack_factor,
            num_bits=self.num_bits,
        )

        self.w2[0] = vllm_ops.awq_marlin_moe_repack(
            self.w2[0],
            self.w2_g_idx_sort_indices,
            size_k=self.w2[0].shape[1],
            size_n=self.w2[0].shape[2] * self.pack_factor,
            num_bits=self.num_bits,
        )

        # Why does this take the intermediate size for size_k?
        self.w1[1] = marlin_moe_permute_scales(
            s=self.w1[1],
            size_k=self.split_inter_size * self.pack_factor,
            size_n=self.w1[1].shape[2],
            group_size=self.group_size,
        )

        self.w2[1] = marlin_moe_permute_scales(
            s=self.w2[1],
            size_k=self.split_inter_size * self.pack_factor,
            size_n=self.w2[1].shape[2],
            group_size=self.group_size,
        )

        self.w1[2] = moe_awq_to_marlin_zero_points(
            self.w1[2],
            size_k=self.w1[2].shape[1],
            size_n=self.w1[2].shape[2] * self.pack_factor,
            num_bits=self.num_bits,
        )

        self.w2[2] = moe_awq_to_marlin_zero_points(
            self.w2[2],
            size_k=self.w2[2].shape[1],
            size_n=self.w2[2].shape[2] * self.pack_factor,
            num_bits=self.num_bits,
        )

        self.workspace = marlin_make_workspace_new(device, 4)

    def _cuda(self, cpu_tensor):
        device_id = get_current_device_id()
        if self.quantized_weight:
            return cpu_tensor.cuda(device_id)
        return cpu_tensor.cuda(device_id)

    def verify_load(self):
        return self.w1 is not None and self.w2 is not None
