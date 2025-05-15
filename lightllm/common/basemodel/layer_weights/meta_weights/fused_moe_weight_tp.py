import os
import torch
import threading
from typing import Optional, Tuple, List, Dict, Any
from .base_weight import BaseWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_current_device_id
from lightllm.common.quantization import Quantcfg


class FusedMoeWeightTP(BaseWeight):
    def __init__(
        self,
        gate_proj_name: str,
        down_proj_name: str,
        up_proj_name: str,
        e_score_correction_bias_name: str,
        weight_prefix: str,
        n_routed_experts: int,
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
        self.n_routed_experts = n_routed_experts
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
        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        use_fp8_w8a8 = self.quant_method is not None

        from lightllm.common.fused_moe.grouped_fused_moe import fused_experts_impl

        fused_experts_impl(
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
                w1_list = []
                for i_experts in range(self.n_routed_experts):
                    expert_gate_up_proj = torch.cat(
                        [self.experts_gate_projs[i_experts], self.experts_up_projs[i_experts]], dim=0
                    )
                    expert_gate_up_proj = expert_gate_up_proj
                    w1_list.append(expert_gate_up_proj)

                inter_shape, hidden_size = w1_list[0].shape[0], w1_list[0].shape[1]
                w1 = torch._utils._flatten_dense_tensors(w1_list).view(len(w1_list), inter_shape, hidden_size)
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
                w1_scale_list = []
                for i_experts in range(self.n_routed_experts):
                    expert_gate_up_proj_scale = torch.cat(
                        [self.experts_gate_proj_scales[i_experts], self.experts_up_proj_scales[i_experts]], dim=0
                    )
                    w1_scale_list.append(expert_gate_up_proj_scale)

                inter_shape, hidden_size = w1_scale_list[0].shape[0], w1_scale_list[0].shape[1]
                w1_scale = torch._utils._flatten_dense_tensors(w1_scale_list).view(
                    len(w1_scale_list), inter_shape, hidden_size
                )
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
