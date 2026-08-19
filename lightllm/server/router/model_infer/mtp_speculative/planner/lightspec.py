from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from lightllm.server.router.model_infer.mtp_speculative.planner.base import (
    BaseMtpPlanner,
    SpecDecodePlan,
    _EMAValue,
    _InferCostMsTable,
)


class LightSpecPlanner(BaseMtpPlanner):
    """Choose the current verify budget and the next draft configuration.

    LightSpec evaluates a runtime configuration as ``(N, B, d)``: logical
    requests, physical target rows, and draft configuration. Planning follows
    the paper's budget-then-fill order. It chooses the current ``B`` within the
    proposal produced by ``pre_draft_step``. The next ``draft_step`` is selected
    from self-consistent ``(B, d)`` configurations for the following iteration,
    so a short current proposal cannot prevent the planner from drafting deeper
    again. Candidate identities belong to the GPU Fill stage.

    Each configuration minimizes estimated milliseconds per committed token:
    ``(target_cost(B) + draft_cost(N, B, d)) / expected_progress(N, B, d)``.
    Progress is learned from one normalized batch observation ``sum(accept_len)
    / B`` per iteration, then smoothed with an EMA. It is never updated once per
    request; doing so would make adaptation depend on concurrency.

    The current MTP mode determines both its valid draft configurations and
    complete draft cost. The planner owns those scheduling inputs instead of
    depending on the proposer implementation.
    """

    def __init__(
        self,
        spec_mode: str,
        backend,
    ) -> None:
        self.spec_mode = spec_mode
        self.backend = backend
        self.max_draft_step = int(backend.max_draft_step)
        self.block_size = int(backend.draft_models[0].block_size) if self.spec_mode == "dflash" else None
        self.draft_steps = self.get_draft_steps()

        self.target_infer_costs = _InferCostMsTable()
        self.draft_infer_costs = _InferCostMsTable()
        self._register_cuda_graph_costs()

        # Each observation is the normalized committed progress U / B from one
        # complete batch. The draft configuration is part of the key because a
        # deeper candidate pool may improve Fill even at the same (N, B).
        self.progress_ema_by_config: Dict[Tuple[int, int, int], _EMAValue] = {}
        # Full-width verification provides an unbiased survival probability
        # for every draft depth. It is the fallback for configurations that
        # have not produced their own progress observation yet.
        self.prefix_survival_by_depth: List[Optional[_EMAValue]] = [None] * self.max_draft_step

        # The current verify width is bounded by the proposal built last time.
        self.pre_draft_step = self.max_draft_step

    def get_draft_steps(self) -> Tuple[int, ...]:
        """Return the draft configurations supported by the current MTP mode."""

        if self.spec_mode in ("vanilla_no_att", "eagle_no_att"):
            return tuple(range(self.max_draft_step + 1))
        if self.spec_mode in ("vanilla_with_att", "eagle_with_att", "eagle3"):
            return tuple(range(1, self.max_draft_step + 1))
        if self.spec_mode == "dflash":
            return (self.max_draft_step,)
        raise ValueError(f"unsupported LightSpec mode: {self.spec_mode}")

    def get_draft_cost_ms(self, req_num: int, verify_batch_size: int, draft_step: int) -> float:
        """Return the complete draft cost for one ``(N, B, d)`` configuration."""

        if self.spec_mode in ("vanilla_no_att", "eagle_no_att"):
            return self.draft_infer_costs.estimate(req_num) * draft_step
        if self.spec_mode in ("vanilla_with_att", "eagle_with_att", "eagle3"):
            assert draft_step > 0, f"{self.spec_mode} requires draft_step to be greater than 0"
            draft_cost_ms = self.draft_infer_costs.estimate(verify_batch_size)
            if draft_step > 1:
                draft_cost_ms += self.draft_infer_costs.estimate(req_num) * (draft_step - 1)
            return draft_cost_ms
        if self.spec_mode == "dflash":
            assert self.block_size is not None
            extend_cost_ms = self.draft_infer_costs.estimate(verify_batch_size)
            block_cost_ms = self.draft_infer_costs.estimate(req_num * self.block_size)
            return extend_cost_ms + block_cost_ms
        raise ValueError(f"unsupported LightSpec mode: {self.spec_mode}")

    def _register_cuda_graph_costs(self) -> None:
        target_graph = self.backend.model.graph
        if target_graph is not None:
            for batch_size, infer_cost_ms in target_graph.infer_cost_ms_by_batch_size.items():
                self.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=infer_cost_ms)

        for draft_model in self.backend.draft_models:
            draft_graph = draft_model.graph
            if draft_graph is None:
                continue
            for batch_size, infer_cost_ms in draft_graph.infer_cost_ms_by_batch_size.items():
                self.draft_infer_costs.update(batch_size=batch_size, infer_cost_ms=infer_cost_ms)

    def plan(self, decode_reqs: List, original_batch_size: int) -> SpecDecodePlan:
        req_num = len(decode_reqs)
        pre_draft_step = self.pre_draft_step
        if req_num == 0:
            self.pre_draft_step = self.max_draft_step
            return SpecDecodePlan(
                dynamic_batch_size=0,
                draft_step=self.max_draft_step,
                pre_draft_step=pre_draft_step,
            )

        # A request entering its first decode has only the guaranteed target
        # row. Existing requests additionally expose pre_draft_step candidates.
        # This bounds Verify by the candidate pool that physically exists.
        req_num_with_proposals = sum(req.cur_output_len > 1 for req in decode_reqs)
        available_batch_size = req_num + req_num_with_proposals * pre_draft_step
        max_batch_size = min(original_batch_size, available_batch_size)
        all_reqs_have_proposals = req_num_with_proposals == req_num

        if not self.progress_ema_by_config:
            # Costs come from CUDA Graph capture, while progress requires one
            # real verification batch. Keep the available proposal intact until
            # J(N, B, d) has a progress observation.
            self.pre_draft_step = self.max_draft_step
            return SpecDecodePlan(
                dynamic_batch_size=max_batch_size,
                draft_step=self.max_draft_step,
                pre_draft_step=pre_draft_step,
                all_reqs_have_proposals=all_reqs_have_proposals,
            )

        min_batch_size = req_num
        batch_sizes = self.target_infer_costs.get_batch_size_keys_between(min_batch_size, max_batch_size)
        costs = [
            self._get_cost_ms(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                draft_step=pre_draft_step,
            )
            for dynamic_batch_size in batch_sizes
        ]
        dynamic_batch_size = batch_sizes[np.argmin(costs)]
        draft_step = self._select_draft_step(req_num=req_num)

        self.pre_draft_step = draft_step
        return SpecDecodePlan(
            dynamic_batch_size=dynamic_batch_size,
            draft_step=draft_step,
            pre_draft_step=pre_draft_step,
            all_reqs_have_proposals=all_reqs_have_proposals,
        )

    def update_feedback(
        self,
        plan: SpecDecodePlan,
        req_num: int,
        accept_lengths,
        schedule_scores=None,
    ) -> None:
        # The progress EMA records one complete-batch sample for a single
        # (N, B, d) configuration. all_reqs_have_proposals is false if any request is
        # on its first decode and therefore has no preceding proposal; its
        # structural accept_len=1 would otherwise bias that configuration's
        # progress downward. Skip the mixed batch instead of partially
        # updating the batch-level statistic with different N/B semantics.
        if not plan.all_reqs_have_proposals:
            return
        self.update_verified_batch(
            accept_lengths=accept_lengths,
            req_num=req_num,
            dynamic_batch_size=plan.dynamic_batch_size,
            verified_draft_step=plan.pre_draft_step,
        )

    def update_verified_batch(
        self,
        accept_lengths,
        req_num: int,
        dynamic_batch_size: int,
        verified_draft_step: int,
    ) -> None:
        """Record one batch-level progress sample for the verified configuration."""

        accept_lengths = np.asarray(accept_lengths)
        if accept_lengths.size == 0:
            return

        config = (int(req_num), int(dynamic_batch_size), int(verified_draft_step))
        progress = float(accept_lengths.sum()) / dynamic_batch_size
        if config not in self.progress_ema_by_config:
            self.progress_ema_by_config[config] = _EMAValue(
                decay=0.9,
                init_value=progress,
            )
        self.progress_ema_by_config[config].update(progress)

        # 只有完整 verify 布局 B = N * (d + 1) 才能统计各 draft 深度的前缀存活率。
        # 如果 B 被动态压缩，较深位置的候选可能根本没有参与验证；此时把这些位置
        # 当作未接受会系统性低估深层候选的效果。
        #
        # accept_lengths 包含每个请求必然提交的 1 个 target token，因此
        # accept_lengths > depth 表示该请求至少接受了前 depth 个 draft token。
        # survival 是第 depth 个 draft token 的前缀存活概率，并通过 EMA 平滑；
        # planner 使用它估算尚未实际运行过的 (N, B, d) 配置能够提交多少 token。
        if dynamic_batch_size == req_num * (verified_draft_step + 1):
            for depth in range(1, verified_draft_step + 1):
                survival = float(np.mean(accept_lengths > depth))
                survival_ema = self.prefix_survival_by_depth[depth - 1]
                if survival_ema is None:
                    survival_ema = _EMAValue(decay=0.9, init_value=survival)
                    self.prefix_survival_by_depth[depth - 1] = survival_ema
                survival_ema.update(survival)

    def _select_draft_step(self, req_num: int) -> int:
        """Choose the best self-consistent next proposal depth.

        The current target batch is bounded by ``pre_draft_step``, but the
        proposal generated now is consumed by the next iteration. Evaluate
        each candidate depth with the verify widths that depth can create so a
        short current proposal cannot permanently prevent deeper drafting.
        """

        best_cost_ms = float("inf")
        best_draft_step = self.draft_steps[0]
        for draft_step in self.draft_steps:
            max_batch_size = req_num * (draft_step + 1)
            batch_sizes = self.target_infer_costs.get_batch_size_keys_between(req_num, max_batch_size)
            for dynamic_batch_size in batch_sizes:
                cost_ms = self._get_cost_ms(
                    req_num=req_num,
                    dynamic_batch_size=dynamic_batch_size,
                    draft_step=draft_step,
                )
                if cost_ms < best_cost_ms:
                    best_cost_ms = cost_ms
                    best_draft_step = draft_step
        return best_draft_step

    def _get_cost_ms(self, req_num: int, dynamic_batch_size: int, draft_step: int) -> float:
        """Estimate milliseconds per committed token for one configuration."""

        accept_ratio = self._estimate_progress(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            draft_step=draft_step,
        )
        total_time = self.target_infer_costs.estimate(dynamic_batch_size) + self.get_draft_cost_ms(
            req_num=req_num,
            verify_batch_size=dynamic_batch_size,
            draft_step=draft_step,
        )
        token_num = min((dynamic_batch_size * accept_ratio), req_num * (draft_step + 1))
        token_num = max(token_num, req_num)
        cost_ms = total_time / token_num
        return cost_ms

    def _estimate_progress(self, req_num: int, dynamic_batch_size: int, draft_step: int) -> float:
        config = (int(req_num), int(dynamic_batch_size), int(draft_step))
        ema = self.progress_ema_by_config.get(config)
        return ema.get() if ema is not None else self._estimate_prefix_progress(*config)

    def _estimate_prefix_progress(self, req_num: int, dynamic_batch_size: int, draft_step: int) -> float:
        if not any(self.prefix_survival_by_depth):
            return 1.0

        remaining_draft_rows = dynamic_batch_size - req_num
        expected_tokens = float(req_num)
        for depth in range(min(draft_step, self.max_draft_step)):
            selected_rows = min(req_num, remaining_draft_rows)
            if selected_rows <= 0:
                break
            survival_ema = self.prefix_survival_by_depth[depth]
            survival = 1.0 if survival_ema is None else survival_ema.get()
            expected_tokens += selected_rows * survival
            remaining_draft_rows -= selected_rows
        return expected_tokens / dynamic_batch_size
