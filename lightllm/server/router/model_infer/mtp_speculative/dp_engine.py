from typing import List, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    linear_att_mtp_state_index_update,
    mtp_scatter_next_token_ids,
    mtp_verify,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_planner import BaseDpPlanner, build_dp_planner
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers import build_dp_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


class DPSpecEngine:
    """普通 DP prefill/decode 使用的 MTP engine。"""

    def __init__(self, backend, spec_mode: str, enable_dynmaic_mtp: bool) -> None:
        self.backend = backend
        self.proposer: BaseDpProposer = build_dp_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )
        self.planner: BaseDpPlanner = build_dp_planner(backend=backend)

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        self.proposer.build_draft_state_from_prefill(
            target_model_input=target_model_input,
            target_model_output=target_model_output,
            next_token_ids=next_token_ids,
        )

    def verify_tokens(
        self,
        next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        b_mtp_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        accept_lengths, accepted_index = mtp_verify(
            req_to_next_token_ids=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=next_token_ids,
            b_req_idx=b_req_idx,
        )
        if self.backend.is_linear_att_mixed_model:
            assert b_mtp_index is not None
            linear_att_mtp_state_index_update(
                req_to_mtp_state_index=self.backend.model.req_manager.req_to_mtp_state_index,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                b_req_idx=b_req_idx,
                b_mtp_index=b_mtp_index,
                accepted_index=accepted_index,
                verify_width=self.backend.max_draft_step + 1,
            )
        return accept_lengths, accepted_index

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        accept_len: Optional[torch.Tensor] = None,
    ) -> SpecProposal:
        return self.proposer.propose_next(
            main_model_input=main_model_input,
            main_model_output=main_model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=self.planner.get_draft_step(),
            accept_len=accept_len,
        )

    def scatter_next_tokens(
        self,
        b_req_mtp_start_loc: torch.Tensor,
        all_next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        schedule_scores: Optional[torch.Tensor] = None,
    ) -> None:
        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            mtp_accept_len=mtp_accept_len,
            req_to_next_token_scores=(
                self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_scores
                if schedule_scores is not None
                else None
            ),
            schedule_scores=schedule_scores,
        )

    def record_request_spec_metrics(
        self,
        decode_reqs: List,
        accept_lengths_cpu: torch.Tensor,
    ) -> None:
        """累计 DP 固定 verify layout 下每个请求的 MTP 指标。"""

        if not self.backend.is_master_in_dp:
            return

        accept_lengths = accept_lengths_cpu.tolist()
        assert len(accept_lengths) == len(decode_reqs)
        for req, accept_len in zip(decode_reqs, accept_lengths):
            req.update_mtp_accepted_token_num(accept_token_num=accept_len - 1)
            verify_token_num = req.mtp_step + 1
            if verify_token_num > 0:
                req.update_mtp_verify_token_num(verify_token_num=verify_token_num)
                req.update_mtp_verify_step_num(verify_step_num=1)


__all__ = ["DPSpecEngine"]
