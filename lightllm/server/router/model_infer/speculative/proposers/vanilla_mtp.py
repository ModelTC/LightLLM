from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer, SpecProposal


class VanillaMTPProposer(BaseSpecProposer):
    """Chained MTP proposer.

    This path uses `mtp_step` independent draft modules.  Step i consumes the
    hidden feature produced by step i - 1 and predicts one candidate token.

    Target -> draft transfer:
    - target prefill/decode captures final hidden states with shape
      [token_num, hidden_size]
    - runtime injects them into ModelInput.mtp_draft_input_hiddens before each
      draft forward
    """

    def build_initial_draft_state(
        self,
        *,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
    ) -> None:
        draft_model_input = model_input
        draft_next_token_ids = next_token_ids
        draft_hidden = self.runtime.get_hidden()
        for draft_model in self.backend.draft_models:
            draft_model_input = self.prepare_draft_prefill_input(
                model_input=draft_model_input,
                next_token_ids=draft_next_token_ids,
                mtp_draft_input_hiddens=draft_hidden,
            )
            draft_model_output = draft_model.forward(draft_model_input)
            draft_next_token_ids = self.backend._gen_argmax_token_ids(draft_model_output)
            draft_hidden = self.runtime.get_hidden()
        return None

    def build_initial_draft_state_overlap(
        self,
        *,
        model_input0: ModelInput,
        next_token_ids0: torch.Tensor,
        model_input1: ModelInput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        draft_model_input0 = model_input0
        draft_model_input1 = model_input1
        draft_next_token_ids0 = next_token_ids0
        draft_next_token_ids1 = next_token_ids1
        draft_hidden0 = self.runtime.get_hidden(0)
        draft_hidden1 = self.runtime.get_hidden(1)

        for draft_model in self.backend.draft_models:
            draft_model_input0 = self.prepare_draft_prefill_input(
                model_input=draft_model_input0,
                next_token_ids=draft_next_token_ids0,
                mtp_draft_input_hiddens=draft_hidden0,
                microbatch_index=0,
            )
            draft_model_input1 = self.prepare_draft_prefill_input(
                model_input=draft_model_input1,
                next_token_ids=draft_next_token_ids1,
                mtp_draft_input_hiddens=draft_hidden1,
                microbatch_index=1,
            )
            draft_model_output0, draft_model_output1 = draft_model.microbatch_overlap_prefill(
                draft_model_input0,
                draft_model_input1,
            )
            draft_next_token_ids0 = self.backend._gen_argmax_token_ids(draft_model_output0)
            draft_next_token_ids1 = self.backend._gen_argmax_token_ids(draft_model_output1)
            draft_hidden0 = self.runtime.get_hidden(0)
            draft_hidden1 = self.runtime.get_hidden(1)
        return None

    def propose_next(
        self,
        *,
        main_model_input: ModelInput,
        main_model_output: ModelOutput = None,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        verify_result=None,
    ) -> SpecProposal:
        del main_model_output
        del verify_result
        assert 0 <= draft_step <= self.backend.mtp_step
        draft_model_input = main_model_input
        draft_next_token_ids = next_token_ids
        draft_hidden = self.runtime.get_hidden() if draft_step > 0 else None
        all_next_token_ids = [next_token_ids]
        draft_probs = [] if self.enable_dynamic_mtp else None

        for step in range(draft_step):
            draft_model = self.backend.draft_models[step]
            draft_model_input = self.prepare_draft_decode_input(
                model_input=draft_model_input,
                next_token_ids=draft_next_token_ids,
                mtp_draft_input_hiddens=draft_hidden,
            )
            draft_model_output = draft_model.forward(draft_model_input)
            draft_hidden = self.runtime.get_hidden()
            if self.enable_dynamic_mtp:
                draft_next_token_ids, draft_prob = self.backend._gen_argmax_token_ids_and_prob(draft_model_output)
                draft_probs.append(draft_prob)
            else:
                draft_next_token_ids = self.backend._gen_argmax_token_ids(draft_model_output)
            all_next_token_ids.append(draft_next_token_ids)

        return SpecProposal(
            token_ids=torch.stack(all_next_token_ids, dim=1),
            extra_mem_indexes_cpu=None,
            draft_probs=draft_probs,
        )
