from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.server.router.model_infer.speculative.planner import (
    DSparkDynamicSpecPlanner,
    DynamicSpecPlanner,
    Eagle3DynamicSpecPlanner,
    FixedSpecPlanner,
    SpecDecodePlan,
)
from lightllm.server.router.model_infer.speculative.proposers import build_spec_proposer
from lightllm.server.router.model_infer.speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.speculative.runner import (
    SpecDecodeForwardState,
    SpecDecodePostState,
    SpecDecodeRunner,
)
from lightllm.server.router.model_infer.speculative.verifier import SpecVerifier, SpecVerifyResult


class SpecEngine:
    """Owns one speculative decoding pipeline for a model backend.

    BaseModel only returns ``ModelOutput.spec_hidden``.  The engine owns
    planning, target verification, draft extension/decoding, and the request
    bookkeeping around that pipeline.  Algorithm-specific state stays in the
    proposer.

    The target-to-draft data path is explicit:
    1. target forward returns ``ModelOutput.spec_hidden``
    2. proposer performs one draft extend from that hidden state
    3. recurrent proposers perform zero or more unit-stride draft decodes
    4. verifier accepts target rows and scatters the next candidate block
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        self.spec_mode = backend.args.mtp_mode
        self.enable_dynamic_spec = backend.args.mtp_dynamic_verify
        self.verifier = SpecVerifier(backend=backend)
        self.proposer = build_spec_proposer(engine=self)
        self.decode_runner = SpecDecodeRunner(engine=self)
        self.planner = self._build_decode_planner()
        self._register_cuda_graph_costs()
        self._dynamic_accept_stats_calls = 0

    def alloc_extra_mem_indexes(self, token_count: int) -> torch.Tensor:
        """Allocate speculative draft-owned temporary KV slots."""

        token_count = int(token_count)
        assert token_count >= 0
        if token_count == 0:
            return torch.empty((0,), dtype=torch.int32, device="cpu")

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(token_count)
        return g_infer_context.req_manager.mem_manager.alloc(token_count)

    def prepare_draft_prefill_input(
        self,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: torch.Tensor,
    ) -> ModelInput:
        """Build draft prefill input from target prefill input.

        `next_token_ids`: [run_req_num]
        `mtp_draft_input_hiddens`: captured target feature.  The first
        dimension matches the target prefill token layout after padding/unpad
        handling; the second dimension is either hidden_size or
        hidden_size * len(target_layer_ids).
        """

        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        return prepare_mtp_prefill_inputs(
            model_input=model_input,
            b_next_token_ids=next_token_ids,
            mtp_draft_input_hiddens=mtp_draft_input_hiddens,
        )

    def prepare_draft_decode_input(
        self,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: torch.Tensor,
    ) -> ModelInput:
        """Mutate a decode ModelInput for one draft forward.

        `next_token_ids`: [verify_batch]
        `mtp_draft_input_hiddens`: [verify_batch, hidden_dim_for_draft]
        """

        model_input.input_ids = next_token_ids
        model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
        if self.spec_mode in ("eagle_with_att", "eagle_no_att", "eagle3"):
            model_input.draft_step = 0
        return model_input

    def build_initial_draft_state(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        self.proposer.build_initial_draft_state(
            model_input=model_input,
            model_output=model_output,
            next_token_ids=next_token_ids,
        )

    def build_initial_draft_state_overlap(
        self,
        model_input0: ModelInput,
        model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        model_input1: ModelInput,
        model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        self.proposer.build_initial_draft_state_overlap(
            model_input0=model_input0,
            model_output0=model_output0,
            next_token_ids0=next_token_ids0,
            model_input1=model_input1,
            model_output1=model_output1,
            next_token_ids1=next_token_ids1,
        )

    def plan_decode(self, model_input: ModelInput, req_num: int) -> SpecDecodePlan:
        """Return the fixed or dynamic speculative plan for one decode iteration."""

        return self.planner.plan(req_num=req_num, original_batch_size=model_input.batch_size)

    def run_decode_speculative_forward(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        run_reqs: List,
        req_num: int,
        plan: SpecDecodePlan,
        selected_row_mask_cpu: Optional[torch.Tensor],
        next_token_ids: torch.Tensor,
        next_token_logprobs: torch.Tensor,
        next_token_ranks: torch.Tensor,
        copy_next_token_infos: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> SpecDecodeForwardState:
        return self.decode_runner.run_speculative_forward(
            model_input=model_input,
            model_output=model_output,
            run_reqs=run_reqs,
            req_num=req_num,
            plan=plan,
            selected_row_mask_cpu=selected_row_mask_cpu,
            next_token_ids=next_token_ids,
            next_token_logprobs=next_token_logprobs,
            next_token_ranks=next_token_ranks,
            copy_next_token_infos=copy_next_token_infos,
        )

    def resolve_decode_pre_post_reqs(self, state: SpecDecodeForwardState, decode_reqs: List):
        return self.decode_runner.resolve_pre_post_reqs(state=state, decode_reqs=decode_reqs)

    def finish_decode_post(
        self,
        state: SpecDecodeForwardState,
        req_num: int,
        run_reqs: List,
    ) -> SpecDecodePostState:
        return self.decode_runner.finish_post(state=state, req_num=req_num, run_reqs=run_reqs)

    def prepare_decode_model_input(
        self,
        model_input: ModelInput,
        req_num: int,
        plan: SpecDecodePlan,
    ):
        """Apply target verify-row compaction when the dynamic planner selects it."""

        if not plan.is_dynamic:
            return model_input, None

        if plan.dynamic_batch_size == model_input.batch_size:
            return model_input, None

        self._clear_stale_dynamic_token_probs(pre_draft_step=plan.pre_draft_step)

        from lightllm.common.basemodel.triton_kernel.mtp_utils import prepare_dynamic_spec_model_input

        model_input, selected_row_mask = prepare_dynamic_spec_model_input(
            model_input=model_input,
            req_num=req_num,
            dynamic_batch_size=plan.dynamic_batch_size,
            req_to_next_token_probs=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_probs,
            pre_draft_step=plan.pre_draft_step,
        )
        return model_input, selected_row_mask

    def async_copy_selected_row_mask(self, selected_row_mask: Optional[torch.Tensor]):
        if selected_row_mask is None:
            return None
        return g_pin_mem_manager.async_copy_from_gpu_tensor(
            key="selected_row_mask",
            gpu_tensor=selected_row_mask,
        )

    def build_decode_req_lists(
        self,
        original_run_reqs,
        selected_row_mask_cpu: Optional[torch.Tensor],
        accepted_index_cpu: torch.Tensor,
    ):
        """Build post-handle request lists after optional verify-row compaction."""

        if self.enable_dynamic_spec and selected_row_mask_cpu is not None:
            selected_row_mask_numpy = selected_row_mask_cpu.numpy()
            run_reqs = [original_run_reqs[i] for i in range(len(original_run_reqs)) if selected_row_mask_numpy[i] == 1]
        else:
            run_reqs = original_run_reqs

        accepted_index_cpu_numpy = accepted_index_cpu.numpy()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu_numpy[i] == 1]
        return run_reqs, verify_ok_reqs

    def build_decode_free_mem_indexes_cpu(
        self,
        model_input: ModelInput,
        selected_row_mask_cpu: Optional[torch.Tensor],
        accepted_index_cpu: torch.Tensor,
    ) -> torch.Tensor:
        mem_indexes_cpu = model_input.mem_indexes_cpu
        if not self.enable_dynamic_spec or selected_row_mask_cpu is None:
            return mem_indexes_cpu[accepted_index_cpu == 0]

        selected_mask = selected_row_mask_cpu.to(dtype=torch.bool)
        accepted_mask = accepted_index_cpu.to(dtype=torch.bool)
        selected_mem_indexes_cpu = mem_indexes_cpu[selected_mask]
        assert selected_mem_indexes_cpu.shape[0] == accepted_mask.shape[0]

        unselected_mem_indexes_cpu = mem_indexes_cpu[~selected_mask]
        rejected_selected_mem_indexes_cpu = selected_mem_indexes_cpu[~accepted_mask]
        if len(unselected_mem_indexes_cpu) == 0:
            return rejected_selected_mem_indexes_cpu
        if len(rejected_selected_mem_indexes_cpu) == 0:
            return unselected_mem_indexes_cpu
        return torch.cat([unselected_mem_indexes_cpu, rejected_selected_mem_indexes_cpu], dim=0)

    def update_dynamic_accept_stats(
        self,
        req_num: int,
        run_reqs,
        accepted_index_cpu: torch.Tensor,
        spec_accept_len_cpu: torch.Tensor,
        dynamic_batch_size: Optional[int],
        pre_draft_step: Optional[int] = None,
    ) -> None:
        if not self.enable_dynamic_spec:
            return

        assert dynamic_batch_size is not None
        assert len(run_reqs) == accepted_index_cpu.shape[0]
        assert spec_accept_len_cpu.shape[0] == req_num
        accept_lengths = spec_accept_len_cpu.numpy()
        accept_count = int(accept_lengths.sum())
        total_count = int(dynamic_batch_size)
        is_full_verify = dynamic_batch_size == req_num * (self.backend.max_draft_step + 1)

        # The first target decode has no preceding draft proposal, so every
        # request structurally accepts only its base token.  Treating that
        # cold-start iteration as a K-wide acceptance sample makes a highly
        # predictable workload look maximally hard and can collapse the
        # controller before the first real proposal is verified.
        self._dynamic_accept_stats_calls += 1
        if self._dynamic_accept_stats_calls == 1:
            return

        if self.spec_mode == "eagle3":
            self.planner.update_req_num_to_dynamic_batch_size_to_accept_ratio(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                accept_ratio=accept_count / total_count,
                pre_draft_step=pre_draft_step,
            )
            if is_full_verify:
                self.planner.update_full_verify_tokens_per_req(
                    accept_count / req_num,
                    req_num=req_num,
                )
            self.planner.update_observed_iteration_stats(
                tokens_per_req=accept_count / req_num,
                verify_rows_per_req=dynamic_batch_size / req_num,
                is_full_verify=is_full_verify,
                req_num=req_num,
            )
            # Confidence-selected rows bias the survival curve upward, so only
            # full-width verification provides valid Eagle3 prefix samples.
            if is_full_verify:
                self.planner.update_verified_batch_prefix_stats(
                    verify_and_accept_lengths=[
                        (self.backend.max_draft_step + 1, int(accept_len)) for accept_len in accept_lengths
                    ],
                )
        else:
            self.planner.update_req_num_to_dynamic_batch_size_to_accept_ratio(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                accept_ratio=accept_count / total_count,
            )
            verify_rows_per_req = max(1, int(round(dynamic_batch_size / req_num)))
            for accept_len in accept_lengths:
                self.planner.update_verified_prefix_stats(
                    verify_len=verify_rows_per_req,
                    accept_len=int(accept_len),
                )

    def needs_schedule_probs_cpu(self) -> bool:
        """Whether this planner consumes proposal confidence on the CPU."""

        return self.enable_dynamic_spec and self.spec_mode == "dspark"

    def update_dynamic_schedule_stats(
        self,
        req_num: int,
        schedule_probs_cpu: Optional[torch.Tensor],
    ) -> None:
        if not self.needs_schedule_probs_cpu() or schedule_probs_cpu is None:
            return

        self.planner.update_predicted_schedule_probs(
            schedule_probs=schedule_probs_cpu,
            req_num=req_num,
        )

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: Optional[ModelOutput],
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: Optional[torch.Tensor] = None,
    ) -> SpecProposal:
        return self.proposer.propose_next(
            main_model_input=main_model_input,
            main_model_output=main_model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=draft_step,
            accept_len=accept_len,
        )

    def propose_next_overlap(
        self,
        main_model_input0: ModelInput,
        main_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        real_verify_rows0: int,
        accept_len0: torch.Tensor,
        main_model_input1: ModelInput,
        main_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
        real_verify_rows1: int,
        accept_len1: torch.Tensor,
        draft_step: int,
    ) -> SpecProposal:
        return self.proposer.propose_next_overlap(
            main_model_input0=main_model_input0,
            main_model_output0=main_model_output0,
            next_token_ids0=next_token_ids0,
            real_verify_rows0=real_verify_rows0,
            accept_len0=accept_len0,
            main_model_input1=main_model_input1,
            main_model_output1=main_model_output1,
            next_token_ids1=next_token_ids1,
            real_verify_rows1=real_verify_rows1,
            accept_len1=accept_len1,
            draft_step=draft_step,
        )

    def verify_target_tokens(
        self,
        new_next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
    ) -> SpecVerifyResult:
        return self.verifier.verify_target_tokens(
            new_next_token_ids=new_next_token_ids,
            b_req_idx=b_req_idx,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
        )

    def build_all_next_token_probs(
        self,
        proposal: SpecProposal,
        draft_step: int,
    ) -> Optional[torch.Tensor]:
        """Build selected-token probabilities for dynamic speculative scatter.

        Output shape is [verify_batch, max_draft_step + 1].  Column 0 is the target
        token probability, fixed to 1 because the target sample is always the
        base accepted position.  Draft columns store selected-token
        probabilities from each proposer step.
        """

        if not self.enable_dynamic_spec:
            return None

        schedule_probs = proposal.schedule_probs if proposal.schedule_probs is not None else proposal.draft_probs
        assert schedule_probs is not None

        verify_row_count = proposal.token_ids.shape[0]
        all_next_token_probs = torch.zeros(
            size=(verify_row_count, self.backend.max_draft_step + 1),
            dtype=torch.float32,
            device=proposal.token_ids.device,
        )
        all_next_token_probs[:, 0] = 1.0

        if isinstance(schedule_probs, torch.Tensor):
            assert schedule_probs.shape == (verify_row_count, draft_step)
            if draft_step > 0:
                all_next_token_probs[:, 1 : draft_step + 1] = schedule_probs
            return all_next_token_probs

        assert len(schedule_probs) == draft_step
        for step_idx, step_probs in enumerate(schedule_probs):
            all_next_token_probs[:, step_idx + 1] = step_probs
        return all_next_token_probs

    def pad_all_next_token_ids(self, token_ids: torch.Tensor, draft_step: int) -> torch.Tensor:
        """Pad a shorter dynamic proposal to the configured speculative width."""

        if not self.enable_dynamic_spec or draft_step >= self.backend.max_draft_step:
            return token_ids

        append_next_token_ids = torch.ones(
            size=(token_ids.shape[0], self.backend.max_draft_step - draft_step),
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        return torch.cat([token_ids, append_next_token_ids], dim=-1)

    def scatter_next_tokens(
        self,
        b_req_mtp_start_loc: torch.Tensor,
        all_next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        spec_accept_len: torch.Tensor,
        all_next_token_probs: Optional[torch.Tensor] = None,
    ) -> None:
        self.verifier.scatter_next_tokens(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            spec_accept_len=spec_accept_len,
            all_next_token_probs=all_next_token_probs,
        )

    def scatter_token_id_steps(
        self,
        token_id_steps: List[torch.Tensor],
        b_req_mtp_start_loc: torch.Tensor,
        b_req_idx: torch.Tensor,
        spec_accept_len: torch.Tensor,
        row_count: Optional[int] = None,
    ) -> torch.Tensor:
        """Stack proposal token columns and scatter them for the next verify."""

        all_next_token_ids = torch.stack(token_id_steps, dim=1)
        if row_count is not None:
            all_next_token_ids = all_next_token_ids[: int(row_count), :]
        self.scatter_next_tokens(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            spec_accept_len=spec_accept_len,
        )
        return all_next_token_ids

    def _build_decode_planner(self):
        if not self.enable_dynamic_spec:
            return FixedSpecPlanner(max_draft_step=self.backend.max_draft_step)
        if self.spec_mode == "dspark":
            return DSparkDynamicSpecPlanner(max_draft_step=self.backend.max_draft_step)
        if self.spec_mode == "eagle3":
            return Eagle3DynamicSpecPlanner(max_draft_step=self.backend.max_draft_step)
        return DynamicSpecPlanner(max_draft_step=self.backend.max_draft_step)

    def _register_cuda_graph_costs(self) -> None:
        if not self.enable_dynamic_spec:
            return

        target_graph = self.backend.model.graph
        if target_graph is not None:
            for batch_size, infer_cost_ms in target_graph.infer_cost_ms_by_batch_size.items():
                self.planner.update_infer_cost(
                    batch_size=batch_size,
                    infer_cost_ms=infer_cost_ms,
                    is_draft_model=False,
                )

        if self.spec_mode == "dspark":
            return
        for draft_model in self.backend.draft_models:
            draft_graph = draft_model.graph
            if draft_graph is None:
                continue
            for batch_size, infer_cost_ms in draft_graph.infer_cost_ms_by_batch_size.items():
                self.planner.update_infer_cost(
                    batch_size=batch_size,
                    infer_cost_ms=infer_cost_ms,
                    is_draft_model=True,
                )

    def _clear_stale_dynamic_token_probs(self, pre_draft_step: int) -> None:
        # Columns after the previous draft length are stale and must not be
        # sampled by dynamic row compaction in the current target forward.
        self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_probs[
            :, (pre_draft_step + 1) :
        ].fill_(0.0)
