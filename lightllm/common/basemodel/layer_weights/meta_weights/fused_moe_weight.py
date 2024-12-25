import os
import torch
from .base_weight import BaseWeight
from lightllm.utils.dist_utils import get_world_size, get_rank
import threading
from lightllm.common.quantization import vLLMFP8w8a8QuantizationMethod

from lightllm.common.vllm_kernel import _custom_ops as ops
from lightllm.common.fused_moe import fused_experts
from lightllm.utils.device_utils import get_current_device_id


class FusedMoeWeight(BaseWeight):
    def __init__(
        self, gate_proj_name, down_proj_name, up_proj_name, weight_prefix, n_routed_experts, split_inter_size, data_type
    ):
        super().__init__()
        self.w1_weight_name = gate_proj_name
        self.w2_weight_name = down_proj_name
        self.w3_weight_name = up_proj_name
        self.weight_prefix = weight_prefix
        self.n_routed_experts = n_routed_experts
        self.split_inter_size = split_inter_size
        self.data_type_ = data_type
        self.tp_rank_ = get_rank()
        self.experts_up_projs = [None] * self.n_routed_experts
        self.experts_gate_projs = [None] * self.n_routed_experts
        self.expert_gate_up_proj_etp = None
        self.expert_down_proj_etp = None
        self.w2_list = [None] * self.n_routed_experts
        self.quant_method = None
        self.lock = threading.Lock()

    def set_quant_method(self, quant_method):
        if isinstance(quant_method, vLLMFP8w8a8QuantizationMethod):
            self.quant_method = quant_method
            if self.quant_method is not None:
                self.quant_method.is_moe = True

    def experts(self, input_tensor, router_logits, top_k, renormalize, use_grouped_topk, topk_group, num_expert_group):
        topk_weights, topk_ids = ops.select_experts(
            hidden_states=input_tensor,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
        )
        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        use_fp8_w8a8 = self.quant_method is not None
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

    def _fuse(self):
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
                self.w1 = torch._utils._flatten_dense_tensors(w1_list).view(len(w1_list), inter_shape, hidden_size)
                inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                self.w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(
                    len(self.w2_list), inter_shape, hidden_size
                )
                if self.quant_method is not None:
                    self.w1 = self.quant_method.quantize(self.w1)
                    self.w2 = self.quant_method.quantize(self.w2)
                else:
                    self.w1 = [self._cuda(self.w1), None]
                    self.w2 = [self._cuda(self.w2), None]
                delattr(self, "w2_list")
                delattr(self, "experts_up_projs")
                delattr(self, "experts_gate_projs")

    def _load_hf_weights_etp(self, weights):
        world_size_ = get_world_size()
        assert self.n_routed_experts % world_size_ == 0
        n_expert_ep = self.n_routed_experts // world_size_

        # tp to ep here
        expert_gate_up_proj_last = None
        expert_down_proj_last = None

        for i_experts_ep in range(n_expert_ep):
            expert_up_proj = None
            expert_gate_proj = None
            expert_gate_up_proj = None
            expert_down_proj = None
            i_experts = i_experts_ep + n_expert_ep * self.tp_rank_

            if f"{self.weight_prefix}.{i_experts}.up_proj.weight" in weights:
                expert_up_proj = weights[f"{self.weight_prefix}.{i_experts}.up_proj.weight"]

                # self.experts_up_proj[i_experts] = expert_up_proj

            if f"{self.weight_prefix}.{i_experts}.gate_proj.weight" in weights:
                expert_gate_proj = weights[f"{self.weight_prefix}.{i_experts}.gate_proj.weight"]
                # self.experts_gate_proj[i_experts] = expert_gate_proj

            if expert_gate_proj is not None and expert_up_proj is not None:
                expert_gate_up_proj = torch.cat([expert_gate_proj, expert_up_proj], dim=0)
                self.experts_gate_projs[i_experts_ep] = expert_gate_up_proj  # self._cuda(expert_gate_up_proj)
                expert_gate_up_proj_last = expert_gate_up_proj

            if f"{self.weight_prefix}.{i_experts}.down_proj.weight" in weights:
                expert_down_proj = weights[f"{self.weight_prefix}.{i_experts}.down_proj.weight"]
                self.experts_up_projs[i_experts_ep] = expert_down_proj  # self._cuda(expert_down_proj)
                expert_down_proj_last = expert_down_proj

            with self.lock:
                if expert_gate_up_proj_last is not None:
                    # package, if there is broken experts

                    if self.expert_gate_up_proj_etp is None:
                        self.expert_gate_up_proj_etp = torch.zeros(
                            (n_expert_ep,) + expert_gate_up_proj_last.shape, dtype=expert_gate_up_proj_last.dtype
                        ).cuda(self.tp_rank_)

                    for i_experts_ep in range(n_expert_ep):
                        if self.experts_gate_projs[i_experts_ep] is not None:
                            self.expert_gate_up_proj_etp[i_experts_ep, :] = self.experts_gate_projs[i_experts_ep]

                if expert_down_proj_last is not None:
                    # package, if there is broken experts
                    if self.expert_down_proj_etp is None:
                        self.expert_down_proj_etp = torch.zeros(
                            (n_expert_ep,) + expert_down_proj_last.shape, dtype=expert_down_proj_last.dtype
                        ).cuda(self.tp_rank_)

                    for i_experts_ep in range(n_expert_ep):
                        if self.experts_up_projs[i_experts_ep] is not None:
                            self.expert_down_proj_etp[i_experts_ep, :] = self.experts_up_projs[i_experts_ep]

    def load_hf_weights(self, weights):
        if os.environ.get("ETP_MODE_ENABLED") == "true":
            self._load_hf_weights_etp(weights)
        else:
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

            self._fuse()

    def _cuda(self, cpu_tensor):
        device_id = get_current_device_id()
        return cpu_tensor.contiguous().to(self.data_type_).cuda(device_id)

    def verify_load(self):
        if os.environ.get("ETP_MODE_ENABLED") == "true":
            return True
        else:
            return self.w1 is not None and self.w2 is not None
