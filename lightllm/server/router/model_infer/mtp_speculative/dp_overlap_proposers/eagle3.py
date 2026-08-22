import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_with_att import (
    DpOverlapEagleWithAttProposer,
)


class DpOverlapEagle3Proposer(DpOverlapEagleWithAttProposer):
    """复用 DP EAGLE With-Att KV 流程并执行 draft-to-target 词表映射。"""

    def _map_draft_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return self.backend.draft_models[0].map_draft_vocab_to_main_vocab(draft_token_ids)

    def _gen_argmax_token_ids(self, model_output: ModelOutput) -> torch.Tensor:
        draft_token_ids = super()._gen_argmax_token_ids(model_output)
        return self._map_draft_token_ids(draft_token_ids)

    def _gen_argmax_token_ids_and_prob(self, model_output: ModelOutput):
        draft_token_ids, draft_token_probs = super()._gen_argmax_token_ids_and_prob(model_output)
        return self._map_draft_token_ids(draft_token_ids), draft_token_probs
