from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.pin_mem_manager import AsyncPinnedCpuTensor, g_pin_mem_manager
from lightllm.server.router.model_infer.mtp_speculative.planner import (
    BaseMtpPlanner,
    DSparkPlanner,
    FixedSpecPlanner,
    LightSpecPlanner,
    SpecDecodePlan,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers import build_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer, SpecProposal

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


class SpecEngine:
    """Owns MTP planning, proposal generation, and target verification."""

    def __init__(
        self,
        backend: ModeBackend,
        spec_mode: str,
        enable_dynmaic_mtp: bool,
    ) -> None:
        self.backend = backend
        self.spec_mode = spec_mode
        self.proposer: BaseSpecProposer = build_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )
        self.planner: BaseMtpPlanner = self._build_mtp_planner(
            spec_mode=spec_mode,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )

    # Prefill draft-state initialization.

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        self.proposer.fill_draft_model_kv_state(
            target_model_input=target_model_input,
            target_model_output=target_model_output,
            target_next_token_ids=target_next_token_ids,
        )

    # Decode planning.

    def plan_decode(self, model_input: ModelInput, decode_reqs: List) -> SpecDecodePlan:
        """Return the fixed or dynamic speculative plan for one decode iteration."""

        return self.planner.plan(
            decode_reqs=decode_reqs,
            origin_batch_size=model_input.batch_size,
        )

    def prepare_decode_model_input(
        self,
        model_input: ModelInput,
        req_num: int,
        plan: SpecDecodePlan,
    ) -> Tuple[ModelInput, Optional[AsyncPinnedCpuTensor]]:
        """Apply target verify-row compaction when the planned batch is smaller."""

        assert model_input.batch_size == plan.origin_batch_size
        if plan.dynamic_batch_size == plan.origin_batch_size:
            return model_input, None

        from lightllm.common.basemodel.triton_kernel.dynamic_mtp_utils import prepare_dynamic_mtp_model_input
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        # mem_indexes 是本轮 decode 新申请、尚未绑定请求和 token 位置的 KV slot。
        # 动态 verify 只需要保留 dynamic_batch_size 个任意 slot，因此 CPU 和已存在的
        # GPU 索引都可以直接截取前缀，无需等待 selected_row_mask_cpu。多申请的 CPU
        # 尾部索引在这里立即归还；后续 forward 会根据压缩后的 b_req_idx/b_seq_len
        # 建立保留 slot 与实际请求位置之间的映射。该操作需要放在下方动态输入构建
        # 之前，避免其内部 to_cuda 将原始完整 batch 的 mem indexes 全量复制到 GPU。
        unused_mem_indexes_cpu = model_input.mem_indexes_cpu[plan.dynamic_batch_size :]
        model_input.mem_indexes_cpu = model_input.mem_indexes_cpu[: plan.dynamic_batch_size]
        if model_input.mem_indexes is not None:
            model_input.mem_indexes = model_input.mem_indexes[: plan.dynamic_batch_size]
        if unused_mem_indexes_cpu.numel() > 0:
            g_infer_context.req_manager.mem_manager.free(unused_mem_indexes_cpu)

        model_input, selected_row_mask = prepare_dynamic_mtp_model_input(
            model_input=model_input,
            req_num=req_num,
            dynamic_batch_size=plan.dynamic_batch_size,
            req_to_next_token_scores=(
                self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_scores
            ),
            pre_draft_step=plan.pre_draft_step,
        )
        selected_row_mask_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor_with_event(
            key="selected_row_mask",
            gpu_tensor=selected_row_mask,
        )
        return model_input, selected_row_mask_cpu

    # Draft proposal generation.

    def propose_next(
        self,
        target_model_input: ModelInput,  # batch_size = verify_batch_size
        target_model_output: ModelOutput,  # logits: [verify_batch_size, vocab_size]
        target_next_token_ids: torch.Tensor,  # [verify_batch_size]
        b_req_mtp_start_loc: torch.Tensor,  # [req_num]
        draft_step: int,
        accept_len: Optional[torch.Tensor] = None,  # [req_num]
    ) -> SpecProposal:
        return self.proposer.propose_next(
            target_model_input=target_model_input,
            target_model_output=target_model_output,
            target_next_token_ids=target_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=draft_step,
            accept_len=accept_len,
        )

    # Target sampling and verification.

    def sample_and_verify(
        self,
        logits: torch.Tensor,
        run_reqs: List,
        b_req_idx: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        b_mtp_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
        from lightllm.server.router.model_infer.mtp_speculative.utils import (
            update_mtp_state_after_verify,
            verify_mtp_tokens,
        )

        if self.spec_mode == "dflash2":
            assert logits.shape[0] == b_req_mtp_start_loc.shape[0] * (
                self.backend.max_draft_step + 1
            ), "DFlash2 requires fixed-width verification"

        # Greedy DFlash2 uses token equality and the regular sampler's logprobs,
        # without reading proposal distributions or consuming rejection RNG.
        if self.spec_mode == "dflash2" and any(req.sampling_param.shm_param.top_k != 1 for req in run_reqs):
            from lightllm.server.router.model_infer.mtp_speculative.dflash2 import sample_and_verify_dflash2_tokens

            next_token_ids, next_token_logprobs, accept_lengths, accepted_index = sample_and_verify_dflash2_tokens(
                backend=self.backend,
                logits=logits,
                run_reqs=run_reqs,
                b_req_idx=b_req_idx,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
            )
            update_mtp_state_after_verify(
                backend=self.backend,
                b_req_idx=b_req_idx,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                b_mtp_index=b_mtp_index,
                accepted_index=accepted_index,
            )
        else:
            next_token_ids, next_token_logprobs = sample(logits, run_reqs, self.backend.eos_id)
            accept_lengths, accepted_index = verify_mtp_tokens(
                backend=self.backend,
                next_token_ids=next_token_ids,
                b_req_idx=b_req_idx,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                b_mtp_index=b_mtp_index,
            )
        return next_token_ids, next_token_logprobs, accept_lengths, accepted_index

    def prepare_next_verification_state(
        self,
        proposal: SpecProposal,
        target_next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        b_req_idx: torch.Tensor,
        mtp_accept_len: torch.Tensor,
    ) -> None:
        from lightllm.server.router.model_infer.mtp_speculative.utils import scatter_mtp_next_tokens

        scatter_mtp_next_tokens(
            backend=self.backend,
            proposal=proposal,
            target_next_token_ids=target_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        if self.spec_mode == "dflash2":
            from lightllm.server.router.model_infer.mtp_speculative.dflash2 import save_dflash2_proposal_state

            save_dflash2_proposal_state(
                backend=self.backend,
                proposal=proposal,
                b_req_idx=b_req_idx,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
            )

    # Planner runtime statistics.

    def update_planner_statics(
        self,
        plan: SpecDecodePlan,
        proposal: SpecProposal,
        req_num: int,
        accept_lengths_cpu: torch.Tensor,
    ) -> None:
        """Update the current planner with iteration-level runtime statistics."""

        self.planner.update_statics(
            plan=plan,
            proposal=proposal,
            req_num=req_num,
            accept_lengths=accept_lengths_cpu,
        )

    def _build_mtp_planner(self, spec_mode: str, enable_dynmaic_mtp: bool) -> BaseMtpPlanner:
        if not enable_dynmaic_mtp:
            return FixedSpecPlanner(max_draft_step=self.backend.max_draft_step)
        if spec_mode == "dspark":
            return DSparkPlanner(backend=self.backend)
        return LightSpecPlanner(spec_mode=spec_mode, backend=self.backend)
