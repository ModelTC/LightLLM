import time
from typing import Dict, List

import torch
import torch.distributed as dist

import lightllm.utils.petrel_helper as utils
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_func
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import (
    FusedMoeWeight,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.prefill_eplb import (
    build_logical_to_physical_map,
    estimate_rank_load,
    plan_redundant_experts,
)
from lightllm.utils.dist_utils import (
    get_current_device_id,
    get_global_rank,
    get_global_world_size,
)
from lightllm.utils.envs_utils import (
    get_env_start_args,
    get_prefill_eplb_max_rebalances,
    get_prefill_eplb_step_interval,
)
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)
PREFILL_EPLB_MIN_AVG_TOKENS_PER_EXPERT = 100


def _find_ep_weights(model: TpPartBaseModel) -> List[FusedMoeWeight]:
    weights = []
    for layer in model.trans_layers_weight:
        for value in vars(layer).values():
            if isinstance(value, FusedMoeWeight) and value.enable_ep_moe:
                weights.append(value)
    return sorted(weights, key=lambda weight: weight.layer_num_)


def _imbalance_summary(rank_load: torch.Tensor) -> Dict[str, float]:
    layer_imbalance = rank_load.max(dim=1).values / rank_load.mean(dim=1)
    sorted_imbalance = torch.sort(layer_imbalance).values
    p95_index = max(0, (95 * layer_imbalance.numel() + 99) // 100 - 1)
    return {
        "max": float(layer_imbalance.max().item()),
        "p95": float(sorted_imbalance[p95_index].item()),
    }


class _PlannedExpertWeightLoader:
    def __init__(
        self,
        weight: FusedMoeWeight,
        expert_ids: torch.Tensor,
    ):
        experts_per_rank = weight.n_routed_experts // weight.global_world_size
        self.weight = weight
        self.expert_slots = [
            (int(expert_id), experts_per_rank + slot) for slot, expert_id in enumerate(expert_ids.tolist())
        ]

    def load_hf_weights(self, weights):
        for expert_id, local_slot in self.expert_slots:
            with self.weight.lock:
                self.weight._load_expert(expert_id, local_slot, weights)
                self.weight._load_expert_scale(expert_id, local_slot, weights)
                self.weight._load_expert_zero_point(
                    expert_id,
                    local_slot,
                    weights,
                )


class PrefillEPLBManager:
    """Online EPLB using prefill logical-expert load only."""

    def __init__(self, model: TpPartBaseModel):
        self.model = model
        self.args = get_env_start_args()
        self.weights = _find_ep_weights(model)
        assert self.weights, "prefill EPLB requires at least one EP MoE layer"

        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.device_id = get_current_device_id()
        self.step_interval = get_prefill_eplb_step_interval()
        self.max_rebalances = get_prefill_eplb_max_rebalances()
        self.prefill_steps = 0
        self.rebalance_count = 0
        self.gloo_group = dist.new_group(
            list(range(self.world_size)),
            backend="gloo",
        )

        routed_expert_nums = {weight.n_routed_experts for weight in self.weights}
        redundant_nums = {weight.redundancy_expert_num for weight in self.weights}
        assert len(routed_expert_nums) == 1
        assert len(redundant_nums) == 1
        self.num_logical_experts = routed_expert_nums.pop()
        self.redundant_experts_per_rank = redundant_nums.pop()

        self.use_safetensors = True
        files = utils.PetrelHelper.list(self.args.model_dir, extension="all")
        self.candidate_files = sorted(file for file in files if file.endswith(".safetensors"))
        if not self.candidate_files:
            self.use_safetensors = False
            self.candidate_files = sorted(file for file in files if file.endswith(".bin"))
        assert self.candidate_files, "prefill EPLB only supports safetensors and PyTorch weight files"

        for weight in self.weights:
            weight.routed_expert_counter_tensor.zero_()

        if self.global_rank == 0:
            logger.info(
                "prefill_eplb enabled "
                f"layers={len(self.weights)} "
                f"logical_experts={self.num_logical_experts} "
                f"redundant_experts_per_rank="
                f"{self.redundant_experts_per_rank} "
                f"step_interval={self.step_interval} "
                f"max_rebalances={self.max_rebalances}"
            )

    def step_after_prefill(self):
        if self.rebalance_count >= self.max_rebalances:
            return
        self.prefill_steps += 1
        if self.prefill_steps % self.step_interval != 0:
            return

        torch.cuda.synchronize(self.device_id)
        global_expert_load = torch.stack(
            [weight.routed_expert_counter_tensor.detach().cpu() for weight in self.weights]
        )
        dist.all_reduce(
            global_expert_load,
            op=dist.ReduceOp.SUM,
            group=self.gloo_group,
        )
        minimum_samples = self.num_logical_experts * PREFILL_EPLB_MIN_AVG_TOKENS_PER_EXPERT
        layer_samples = global_expert_load.sum(dim=1)
        if torch.any(layer_samples < minimum_samples):
            if self.global_rank == 0:
                logger.info(
                    "prefill_eplb skip rearrangement: "
                    f"minimum_layer_samples={int(layer_samples.min().item())} "
                    f"required={minimum_samples}"
                )
            return

        placement = plan_redundant_experts(
            global_expert_load,
            self.world_size,
            self.redundant_experts_per_rank,
        )
        experts_per_rank = self.num_logical_experts // self.world_size
        before_rank_load = (
            global_expert_load.to(torch.float64)
            .reshape(
                len(self.weights),
                self.world_size,
                experts_per_rank,
            )
            .sum(dim=-1)
        )
        after_rank_load = estimate_rank_load(global_expert_load, placement)
        before = _imbalance_summary(before_rank_load)
        after = _imbalance_summary(after_rank_load)

        if self.global_rank == 0:
            logger.info(
                "prefill_eplb rearranging "
                f"prefill_steps={self.prefill_steps} "
                f"estimated_imbalance_max_before={before['max']:.4f} "
                f"estimated_imbalance_max_after={after['max']:.4f} "
                f"estimated_imbalance_p95_before={before['p95']:.4f} "
                f"estimated_imbalance_p95_after={after['p95']:.4f}"
            )

        started_at = time.time()
        local_placement = placement[:, self.global_rank, :]
        loaders = [
            _PlannedExpertWeightLoader(weight, expert_ids) for weight, expert_ids in zip(self.weights, local_placement)
        ]
        for file in self.candidate_files:
            load_func(
                file,
                use_safetensors=self.use_safetensors,
                pre_post_layer=None,
                transformer_layer_list=loaders,
                weight_dir=self.args.model_dir,
            )

        dist.barrier(group=self.gloo_group)
        for layer_index, weight in enumerate(self.weights):
            layer_placement = placement[layer_index]
            logical_to_physical, logical_replica_count = build_logical_to_physical_map(
                layer_placement,
                self.num_logical_experts,
            )
            local_expert_ids = local_placement[layer_index]
            weight.redundancy_expert_ids = local_expert_ids.tolist()
            weight.redundancy_expert_idx_to_local_idx = {
                int(expert_id): experts_per_rank + slot for slot, expert_id in enumerate(local_expert_ids.tolist())
            }
            weight.redundancy_expert_ids_tensor.copy_(
                local_expert_ids,
                non_blocking=False,
            )
            weight.prefill_eplb_logical_to_physical_map.copy_(
                logical_to_physical,
                non_blocking=False,
            )
            weight.prefill_eplb_logical_replica_count.copy_(
                logical_replica_count,
                non_blocking=False,
            )
            weight.routed_expert_counter_tensor.zero_()

        torch.cuda.synchronize(self.device_id)
        dist.barrier(group=self.gloo_group)
        self.rebalance_count += 1
        if self.global_rank == 0:
            logger.info(
                "prefill_eplb rearranged " f"count={self.rebalance_count} " f"cost={time.time() - started_at:.2f}s"
            )
