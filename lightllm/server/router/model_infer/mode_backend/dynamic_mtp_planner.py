import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size


@dataclass
class DynamicMTPPlan:
    req_num: int
    original_batch_size: int
    dynamic_batch_size: int
    keep_indices: torch.Tensor
    per_req_rows: List[int]
    estimated_accept_mean: float
    estimated_accept_std: float


class _EMAValue:
    def __init__(self, decay: float, init_value: Optional[float] = None) -> None:
        self.decay = decay
        self.value = init_value
        self.initialized = init_value is not None

    def update(self, new_value: float) -> float:
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * new_value
        return self.value

    def get(self, fallback: float) -> float:
        return self.value if self.initialized else fallback


class DynamicMTPPlanner:
    def __init__(
        self,
        mtp_step: int,
        ema_decay: float = 0.9,
        confidence_k: float = 1.0,
    ) -> None:
        self.mtp_step = mtp_step
        self.max_rows_per_req = mtp_step + 1
        self.confidence_k = confidence_k
        self.accept_mean = _EMAValue(ema_decay, init_value=float(self.max_rows_per_req))
        self.accept_second_moment = _EMAValue(ema_decay, init_value=float(self.max_rows_per_req**2))
        self.req_accept_mean: Dict[int, _EMAValue] = {}
        self.req_prob: Dict[int, _EMAValue] = {}
        self.latency_ms_by_batch_size: Dict[int, _EMAValue] = {}
        self.accepted_token_speed = _EMAValue(ema_decay)
        self.verify_row_speed = _EMAValue(ema_decay)
        self.actual_speedup = _EMAValue(ema_decay)
        self.single_token_speed_by_req_num: Dict[int, _EMAValue] = {}
        self.last_plan: Optional[DynamicMTPPlan] = None

    def trim_before_forward(
        self,
        model_input: ModelInput,
        run_reqs: List[InferReq],
        decode_reqs: List[InferReq],
    ):
        plan = self._build_plan(model_input=model_input, decode_reqs=decode_reqs)
        self.last_plan = plan
        if plan.dynamic_batch_size == plan.original_batch_size:
            return model_input, run_reqs, plan

        pruned_indices = self._invert_indices(plan.keep_indices, plan.original_batch_size)
        if pruned_indices.numel() > 0:
            pruned_mem_indexes = model_input.mem_indexes_cpu[pruned_indices]
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(pruned_mem_indexes)
            g_infer_state_lock.release()

        trimmed_input = copy.copy(model_input)
        keep_indices = plan.keep_indices
        keep_list = keep_indices.tolist()

        trimmed_input.batch_size = plan.dynamic_batch_size
        trimmed_input.b_req_idx = model_input.b_req_idx[keep_indices].contiguous()
        trimmed_input.b_mtp_index = model_input.b_mtp_index[keep_indices].contiguous()
        trimmed_input.b_seq_len = model_input.b_seq_len[keep_indices].contiguous()
        trimmed_input.mem_indexes_cpu = model_input.mem_indexes_cpu[keep_indices].contiguous()
        trimmed_input.mem_indexes = None
        trimmed_input.total_token_num = int(trimmed_input.b_seq_len.sum().item())
        trimmed_input.max_kv_seq_len = int(trimmed_input.b_seq_len.max().item())
        trimmed_input.multimodal_params = [model_input.multimodal_params[index] for index in keep_list]
        trimmed_run_reqs = [run_reqs[index] for index in keep_list]
        trimmed_input.b_mark_shared_group = self._build_mtp_shared_group_infos(trimmed_run_reqs)
        trimmed_input.check_input()
        return trimmed_input, trimmed_run_reqs, plan

    def update_after_verify(
        self,
        plan: DynamicMTPPlan,
        decode_reqs: List[InferReq],
        mtp_accept_len_cpu: torch.Tensor,
        elapsed_ms: float,
        per_req_probs_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        if plan is None:
            return

        accept_lens = [int(value) for value in mtp_accept_len_cpu.numpy()]
        if not accept_lens:
            return

        batch_mean = sum(accept_lens) / len(accept_lens)
        batch_second = sum(value * value for value in accept_lens) / len(accept_lens)
        self.accept_mean.update(batch_mean)
        self.accept_second_moment.update(batch_second)

        for req, accept_len in zip(decode_reqs, accept_lens):
            req_ema = self.req_accept_mean.get(req.req_id)
            if req_ema is None:
                req_ema = _EMAValue(self.accept_mean.decay, init_value=batch_mean)
                self.req_accept_mean[req.req_id] = req_ema
            req_ema.update(float(accept_len))

        if per_req_probs_cpu is not None:
            for req, req_prob in zip(decode_reqs, per_req_probs_cpu.numpy()):
                req_prob_ema = self.req_prob.get(req.req_id)
                if req_prob_ema is None:
                    req_prob_ema = _EMAValue(self.accept_mean.decay)
                    self.req_prob[req.req_id] = req_prob_ema
                req_prob_ema.update(self._clip_prob(float(req_prob)))

        if elapsed_ms > 0:
            current_output_speed = sum(accept_lens) / elapsed_ms
            latency_ema = self.latency_ms_by_batch_size.get(plan.dynamic_batch_size)
            if latency_ema is None:
                latency_ema = _EMAValue(self.accept_mean.decay)
                self.latency_ms_by_batch_size[plan.dynamic_batch_size] = latency_ema
            latency_ema.update(elapsed_ms)
            self.accepted_token_speed.update(current_output_speed)
            self.verify_row_speed.update(plan.dynamic_batch_size / elapsed_ms)
            if plan.dynamic_batch_size == plan.req_num:
                single_token_speed = self.single_token_speed_by_req_num.get(plan.req_num)
                if single_token_speed is None:
                    single_token_speed = _EMAValue(self.accept_mean.decay)
                    self.single_token_speed_by_req_num[plan.req_num] = single_token_speed
                single_token_speed.update(plan.req_num / elapsed_ms)
            baseline_speed = self.single_token_speed_by_req_num.get(plan.req_num)
            if baseline_speed is not None and baseline_speed.get(0.0) > 0:
                self.actual_speedup.update(current_output_speed / baseline_speed.get(0.0))

    def get_stats_snapshot(self) -> Dict[str, float]:
        return {
            "accept_mean": self.accept_mean.get(float(self.max_rows_per_req)),
            "accept_second_moment": self.accept_second_moment.get(float(self.max_rows_per_req**2)),
            "accepted_token_speed": self.accepted_token_speed.get(0.0),
            "verify_row_speed": self.verify_row_speed.get(0.0),
            "actual_speedup": self.actual_speedup.get(0.0),
        }

    def _build_plan(self, model_input: ModelInput, decode_reqs: List[InferReq]) -> DynamicMTPPlan:
        req_num = len(decode_reqs)
        original_batch_size = model_input.batch_size
        if req_num == 0 or self.mtp_step == 0:
            keep_indices = torch.arange(original_batch_size, dtype=torch.long, device="cpu")
            return DynamicMTPPlan(req_num, original_batch_size, original_batch_size, keep_indices, [], 1.0, 0.0)

        mean = self.accept_mean.get(float(self.max_rows_per_req))
        second = self.accept_second_moment.get(float(self.max_rows_per_req**2))
        variance = max(0.0, second - mean * mean)
        std = math.sqrt(variance)
        budget = math.ceil(req_num * mean + self.confidence_k * math.sqrt(req_num) * std)
        dynamic_batch_size = max(req_num, min(original_batch_size, budget))

        per_req_rows = self._allocate_rows(decode_reqs=decode_reqs, dynamic_batch_size=dynamic_batch_size)
        keep_indices, per_req_rows = self._build_keep_indices(model_input=model_input, per_req_rows=per_req_rows)
        dynamic_batch_size = int(keep_indices.numel())

        return DynamicMTPPlan(
            req_num=req_num,
            original_batch_size=original_batch_size,
            dynamic_batch_size=dynamic_batch_size,
            keep_indices=keep_indices,
            per_req_rows=per_req_rows,
            estimated_accept_mean=mean,
            estimated_accept_std=std,
        )

    def _allocate_rows(self, decode_reqs: List[InferReq], dynamic_batch_size: int) -> List[int]:
        req_num = len(decode_reqs)
        per_req_rows = [1 for _ in range(req_num)]
        remaining = dynamic_batch_size - req_num
        if remaining <= 0:
            return per_req_rows

        req_order = sorted(
            range(req_num),
            key=lambda index: self._req_prob(decode_reqs[index]),
            reverse=True,
        )

        for req_index in req_order:
            req_prob = self._req_prob(decode_reqs[req_index])
            for _ in range(self.mtp_step):
                if remaining <= 0:
                    break
                if random.random() >= req_prob:
                    break
                per_req_rows[req_index] += 1
                remaining -= 1
            if remaining <= 0:
                break
        return per_req_rows

    def _req_prob(self, req: InferReq) -> float:
        req_prob = self.req_prob.get(req.req_id)
        if req_prob is not None:
            return self._clip_prob(req_prob.get(1.0))
        fallback = self.accept_mean.get(float(self.max_rows_per_req)) / float(self.max_rows_per_req)
        return self._clip_prob(fallback)

    def _clip_prob(self, value: float) -> float:
        return min(1.0, max(0.0, value))

    def _build_keep_indices(self, model_input: ModelInput, per_req_rows: List[int]):
        keep_indices = []
        effective_per_req_rows = [0 for _ in per_req_rows]
        req_index = -1
        cur_req_kept = 0
        cur_req_target = 0
        for index, mtp_index in enumerate(model_input.b_mtp_index.tolist()):
            if mtp_index == 0:
                req_index += 1
                cur_req_kept = 0
                cur_req_target = per_req_rows[req_index]
            if cur_req_kept < cur_req_target:
                keep_indices.append(index)
                cur_req_kept += 1
                effective_per_req_rows[req_index] += 1
        return torch.tensor(keep_indices, dtype=torch.long, device="cpu"), effective_per_req_rows

    def _invert_indices(self, keep_indices: torch.Tensor, total_size: int) -> torch.Tensor:
        keep_mask = torch.zeros((total_size,), dtype=torch.bool, device="cpu")
        keep_mask[keep_indices] = True
        return torch.nonzero(~keep_mask, as_tuple=False).view(-1)

    def _build_mtp_shared_group_infos(self, run_reqs: List[InferReq]) -> torch.Tensor:
        max_batch_shared_group_size = get_diverse_max_batch_shared_group_size()
        req_ids = [req.req_id for req in run_reqs]
        b_mark_shared_group = []
        current_group = []
        for req_id in req_ids:
            if not current_group:
                current_group.append(req_id)
            elif req_id == current_group[-1]:
                current_group.append(req_id)
            else:
                b_mark_shared_group.extend([0 for _ in range(len(current_group))])
                b_mark_shared_group[-1] = len(current_group)
                current_group.clear()
                current_group.append(req_id)

            if len(current_group) == max_batch_shared_group_size:
                b_mark_shared_group.extend([0 for _ in range(len(current_group))])
                b_mark_shared_group[-1] = len(current_group)
                current_group.clear()

        if current_group:
            b_mark_shared_group.extend([0 for _ in range(len(current_group))])
            b_mark_shared_group[-1] = len(current_group)

        return torch.tensor(b_mark_shared_group, dtype=torch.int32, device="cpu")
