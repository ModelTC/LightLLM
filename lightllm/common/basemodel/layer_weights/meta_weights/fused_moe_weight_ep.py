import torch
from typing import Optional, Tuple, List, Dict, Any
from lightllm.utils.dist_utils import get_global_world_size, get_global_rank, get_current_device_id
from .fused_moe_weight_tp import FusedMoeWeightTP


class FusedMoeWeightEP(FusedMoeWeightTP):
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
        quant_cfg=None,
    ) -> None:
        super().__init__(
            gate_proj_name,
            down_proj_name,
            up_proj_name,
            e_score_correction_bias_name,
            weight_prefix,
            n_routed_experts,
            split_inter_size,
            data_type,
            network_config,
            layer_num,
            quant_cfg,
        )
        self.expert_gate_up_proj_etp = None
        self.expert_down_proj_etp = None
        self.global_rank_ = get_global_rank()
        self.device_id_ = get_current_device_id()

    def _load_hf_weights(self, weights):
        world_size_ = get_global_world_size()
        assert self.n_routed_experts % world_size_ == 0
        n_expert_ep = self.n_routed_experts // world_size_

        # tp to ep here
        expert_gate_up_proj_last = None
        expert_down_proj_last = None
        if self.e_score_correction_bias_name in weights:
            self.e_score_correction_bias = self._cuda(weights[self.e_score_correction_bias_name])

        for i_experts_ep in range(n_expert_ep):
            expert_up_proj = None
            expert_gate_proj = None
            expert_gate_up_proj = None
            expert_down_proj = None
            i_experts = i_experts_ep + n_expert_ep * self.global_rank_

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
                        ).cuda(self.device_id_)

                    for i_experts_ep in range(n_expert_ep):
                        if self.experts_gate_projs[i_experts_ep] is not None:
                            self.expert_gate_up_proj_etp[i_experts_ep, :] = self.experts_gate_projs[i_experts_ep]

                if expert_down_proj_last is not None:
                    # package, if there is broken experts
                    if self.expert_down_proj_etp is None:
                        self.expert_down_proj_etp = torch.zeros(
                            (n_expert_ep,) + expert_down_proj_last.shape, dtype=expert_down_proj_last.dtype
                        ).cuda(self.device_id_)

                    for i_experts_ep in range(n_expert_ep):
                        if self.experts_up_projs[i_experts_ep] is not None:
                            self.expert_down_proj_etp[i_experts_ep, :] = self.experts_up_projs[i_experts_ep]

    def verify_load(self):
        return self.expert_gate_up_proj_etp is not None and self.expert_down_proj_etp is not None
