import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import (
    BaseDpOverlapProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_utils import (
    propose_next_dp_no_att_overlap,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import (
    VanillaSpecProposal,
)


class DpOverlapVanillaNoAttProposer(BaseDpOverlapProposer):
    """DP ``vanilla_no_att`` proposer。"""

    def fill_draft_model_kv_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
    ) -> None:
        pass

    def propose_next_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
        real_request_num0: int,
        real_request_num1: int,
        accept_len: torch.Tensor,
        draft_step: int,
    ) -> VanillaSpecProposal:
        proposal = propose_next_dp_no_att_overlap(
            proposer=self,
            target_model_input0=target_model_input0,
            target_model_output0=target_model_output0,
            target_next_token_ids0=target_next_token_ids0,
            target_model_input1=target_model_input1,
            target_model_output1=target_model_output1,
            target_next_token_ids1=target_next_token_ids1,
            real_request_num0=real_request_num0,
            real_request_num1=real_request_num1,
            accept_len=accept_len,
            draft_step=draft_step,
            get_draft_model=lambda step: self.backend.draft_models[step],
            map_draft_token_ids=lambda token_ids: token_ids,
        )
        return VanillaSpecProposal(
            token_ids=proposal.token_ids,
            extra_mem_indexes_cpu=proposal.extra_mem_indexes_cpu,
            schedule_scores=proposal.schedule_scores,
        )
