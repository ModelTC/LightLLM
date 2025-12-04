import numpy as np
import torch
from .fused_moe_weight_ep import FusedMoeWeightEP
from lightllm.utils.log_utils import init_logger
from typing import Dict

logger = init_logger(__name__)


class FusedMoeWeightEPAutoRedundancy:
    def __init__(
        self,
        ep_fused_moe_weight: FusedMoeWeightEP,
    ) -> None:
        super().__init__()
        self._ep_w = ep_fused_moe_weight
        self.redundancy_expert_num = self._ep_w.redundancy_expert_num

    def clear_counter(self):
        self._ep_w.routed_expert_counter_tensor.fill_(0)
        return

    def prepare_redundancy_experts(
        self,
    ):
        expert_counter = self._ep_w.routed_expert_counter_tensor.detach().cpu().numpy()
        logger.info(
            f"layer_index {self._ep_w.layer_num} global_rank {self._ep_w.global_rank_} expert_counter: {expert_counter}"
        )
        self._ep_w.routed_expert_counter_tensor.fill_(0)

        start_expert_id = self._ep_w.ep_n_routed_experts * self._ep_w.global_rank_
        no_redundancy_expert_ids = list(range(start_expert_id, start_expert_id + self._ep_w.ep_n_routed_experts))

        # 统计 0 rank 上的全局 topk 冗余信息，帮助导出一份全局可用的静态使用的冗余专家静态配置。
        if self._ep_w.global_rank_ == 0:
            # int(e) for serialization, int64 can not be serialized by json.dump.
            topk_redundancy_expert_ids = list(int(e) for e in np.argsort(expert_counter)[-self.redundancy_expert_num :])
        else:
            topk_redundancy_expert_ids = None

        # 不要选中当前已经存在的非冗余专家作为冗余专家
        expert_counter[no_redundancy_expert_ids] = 0

        self.redundancy_expert_ids = list(np.argsort(expert_counter)[-self.redundancy_expert_num :])
        logger.info(
            f"layer_index {self._ep_w.layer_num} global_rank {self._ep_w.global_rank_}"
            f" new select redundancy_expert_ids : {self.redundancy_expert_ids}"
        )

        # 准备加载过度变量。
        self.experts_up_projs = [None] * self.redundancy_expert_num
        self.experts_gate_projs = [None] * self.redundancy_expert_num
        self.experts_up_proj_scales = [None] * self.redundancy_expert_num
        self.experts_gate_proj_scales = [None] * self.redundancy_expert_num
        self.w2_list = [None] * self.redundancy_expert_num
        self.w2_scale_list = [None] * self.redundancy_expert_num
        self.w1 = [None, None]  # weight, weight_scale
        self.w2 = [None, None]  # weight, weight_scale
        return topk_redundancy_expert_ids

    def load_hf_weights(self, weights):
        # 加载冗余专家的权重参数
        for i, redundant_expert_id in enumerate(self.redundancy_expert_ids):
            i_experts = redundant_expert_id
            w1_weight = f"{self._ep_w.weight_prefix}.{i_experts}.{self._ep_w.w1_weight_name}.weight"
            w2_weight = f"{self._ep_w.weight_prefix}.{i_experts}.{self._ep_w.w2_weight_name}.weight"
            w3_weight = f"{self._ep_w.weight_prefix}.{i_experts}.{self._ep_w.w3_weight_name}.weight"
            if w1_weight in weights:
                self.experts_gate_projs[i] = weights[w1_weight]
            if w3_weight in weights:
                self.experts_up_projs[i] = weights[w3_weight]
            if w2_weight in weights:
                self.w2_list[i] = weights[w2_weight]

        if self._ep_w.quantized_weight:
            self._load_weight_scale(weights)
        self._fuse()

    def _fuse(self):
        if self._ep_w.quantized_weight:
            self._fuse_weight_scale()
        with self._ep_w.lock:
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
                total_expert_num = self.redundancy_expert_num

                w1 = torch.empty((total_expert_num, gate_out_dim + up_out_dim, gate_in_dim), dtype=dtype, device="cpu")

                for i_experts in range(self.redundancy_expert_num):
                    w1[i_experts, 0:gate_out_dim:, :] = self.experts_gate_projs[i_experts]
                    w1[i_experts, gate_out_dim:, :] = self.experts_up_projs[i_experts]

                inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(len(self.w2_list), inter_shape, hidden_size)
                if not self._ep_w.quantized_weight and self._ep_w.quant_method is not None:
                    qw1, qw1_scale, qw1_zero_point = self._ep_w.quant_method.quantize(w1)
                    qw2, qw2_scale, qw2_zero_point = self._ep_w.quant_method.quantize(w2)
                    self.w1[0] = qw1
                    self.w1[1] = qw1_scale
                    self.w2[0] = qw2
                    self.w2[1] = qw2_scale
                else:
                    self.w1[0] = w1
                    self.w2[0] = w2
                delattr(self, "w2_list")
                delattr(self, "experts_up_projs")
                delattr(self, "experts_gate_projs")

    def _fuse_weight_scale(self):
        with self._ep_w.lock:
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
                total_expert_num = self.redundancy_expert_num
                w1_scale = torch.empty(
                    (total_expert_num, gate_out_dim + up_out_dim, gate_in_dim), dtype=dtype, device="cpu"
                )
                for i_experts in range(self.redundancy_expert_num):
                    w1_scale[i_experts, 0:gate_out_dim:, :] = self.experts_gate_proj_scales[i_experts]
                    w1_scale[i_experts, gate_out_dim:, :] = self.experts_up_proj_scales[i_experts]

                inter_shape, hidden_size = self.w2_scale_list[0].shape[0], self.w2_scale_list[0].shape[1]
                w2_scale = torch._utils._flatten_dense_tensors(self.w2_scale_list).view(
                    len(self.w2_scale_list), inter_shape, hidden_size
                )
                self.w1[1] = w1_scale
                self.w2[1] = w2_scale
                delattr(self, "w2_scale_list")
                delattr(self, "experts_up_proj_scales")
                delattr(self, "experts_gate_proj_scales")

    def _load_weight_scale(self, weights: Dict[str, torch.Tensor]) -> None:
        # 加载冗余专家的scale参数
        for i, redundant_expert_id in enumerate(self.redundancy_expert_ids):
            i_experts = redundant_expert_id
            w1_scale = (
                f"{self._ep_w.weight_prefix}.{i_experts}.{self._ep_w.w1_weight_name}.{self._ep_w.weight_scale_suffix}"
            )
            w2_scale = (
                f"{self._ep_w.weight_prefix}.{i_experts}.{self._ep_w.w2_weight_name}.{self._ep_w.weight_scale_suffix}"
            )
            w3_scale = (
                f"{self._ep_w.weight_prefix}.{i_experts}.{self._ep_w.w3_weight_name}.{self._ep_w.weight_scale_suffix}"
            )
            if w1_scale in weights:
                self.experts_gate_proj_scales[i] = weights[w1_scale]
            if w3_scale in weights:
                self.experts_up_proj_scales[i] = weights[w3_scale]
            if w2_scale in weights:
                self.w2_scale_list[i] = weights[w2_scale]

    def commit(self):
        for index, dest_tensor in enumerate(self._ep_w.w1):
            if dest_tensor is not None:
                assert isinstance(
                    dest_tensor, torch.Tensor
                ), f"dest_tensor should be a torch.Tensor, but got {type(dest_tensor)}"
                dest_tensor[-self.redundancy_expert_num :, :, :] = self.w1[index][:, :, :]

        for index, dest_tensor in enumerate(self._ep_w.w2):
            if dest_tensor is not None:
                assert isinstance(
                    dest_tensor, torch.Tensor
                ), f"dest_tensor should be a torch.Tensor, but got {type(dest_tensor)}"
                dest_tensor[-self.redundancy_expert_num :, :, :] = self.w2[index][:, :, :]

        self._ep_w.redundancy_expert_ids_tensor.copy_(
            torch.tensor(self.redundancy_expert_ids, dtype=torch.int64, device="cpu")
        )
