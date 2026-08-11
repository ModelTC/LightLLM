from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer, SpecProposal


class VanillaMTPProposer(BaseSpecProposer):
    """Chained MTP proposer.

    This path uses `max_draft_step` independent draft modules. Step i consumes the
    hidden feature produced by step i - 1 and predicts one candidate token.

    Target -> draft transfer:
    - target prefill/decode captures final hidden states with shape
      [token_num, hidden_size]
    - SpecEngine injects them into ModelInput.mtp_draft_input_hiddens before each
      draft forward
    """

    def build_initial_draft_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        draft_model_input = target_model_input
        source_model_output = target_model_output
        draft_next_token_ids = next_token_ids
        for draft_model in self.backend.draft_models:
            assert source_model_output.spec_hidden is not None
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids,
                mtp_draft_input_hiddens=source_model_output.spec_hidden,
            )
            source_model_output = draft_model.forward(draft_model_input)
            draft_next_token_ids = self.backend._gen_argmax_token_ids(source_model_output)

    def build_initial_draft_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        draft_model_input0 = target_model_input0
        draft_model_input1 = target_model_input1
        source_model_output0 = target_model_output0
        source_model_output1 = target_model_output1
        draft_next_token_ids0 = next_token_ids0
        draft_next_token_ids1 = next_token_ids1

        for draft_model in self.backend.draft_models:
            assert source_model_output0.spec_hidden is not None
            assert source_model_output1.spec_hidden is not None
            draft_model_input0 = prepare_mtp_prefill_inputs(
                model_input=draft_model_input0,
                b_next_token_ids=draft_next_token_ids0,
                mtp_draft_input_hiddens=source_model_output0.spec_hidden,
            )
            draft_model_input1 = prepare_mtp_prefill_inputs(
                model_input=draft_model_input1,
                b_next_token_ids=draft_next_token_ids1,
                mtp_draft_input_hiddens=source_model_output1.spec_hidden,
            )
            source_model_output0, source_model_output1 = draft_model.microbatch_overlap_prefill(
                draft_model_input0,
                draft_model_input1,
            )
            draft_next_token_ids0 = self.backend._gen_argmax_token_ids(source_model_output0)
            draft_next_token_ids1 = self.backend._gen_argmax_token_ids(source_model_output1)

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> SpecProposal:
        assert 0 <= draft_step <= self.backend.max_draft_step
        draft_model_input = main_model_input
        draft_next_token_ids = next_token_ids
        draft_hidden = main_model_output.spec_hidden if draft_step > 0 else None
        all_next_token_ids = [next_token_ids]
        draft_probs = [] if self.enable_dynamic_spec else None

        for step in range(draft_step):
            draft_model = self.backend.draft_models[step]
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mtp_draft_input_hiddens = draft_hidden
            draft_model_output = draft_model.forward(draft_model_input)
            draft_hidden = draft_model_output.spec_hidden
            assert draft_hidden is not None
            if self.enable_dynamic_spec:
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
