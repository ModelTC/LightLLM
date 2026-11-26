from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo


@dataclass
class Glm5NsaStep:
    """Per query row of one step: its visible kv length, request slot and the start of its request's
    keys in the step's ragged key layout (prefill only). The NSA indexer and FP8 prefill share it."""

    q_kv_seq_len: torch.Tensor
    q_req_idx: torch.Tensor
    q_kv_start_loc: Optional[torch.Tensor] = None


def build_glm5_nsa_step(infer_state: InferStateInfo) -> Glm5NsaStep:
    if not infer_state.is_prefill:
        # Decode rows each see their request's whole cache.
        return Glm5NsaStep(q_kv_seq_len=infer_state.b_seq_len, q_req_idx=infer_state.b_req_idx)
    from lightllm.models.glm5_next.triton_kernel.kpool import prefill_nsa_params

    q_kv_seq_len, q_req_idx, q_kv_start_loc = prefill_nsa_params(
        infer_state.position_ids, infer_state.b_req_idx, infer_state.b1_cu_q_seq_len, infer_state.b1_cu_kv_seq_len
    )
    return Glm5NsaStep(q_kv_seq_len=q_kv_seq_len, q_req_idx=q_req_idx, q_kv_start_loc=q_kv_start_loc)


class Glm5NextInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.nsa_step: Optional[Glm5NsaStep] = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if model.attention_backend_manager.has_attention(role=self.att_state_role):
            self.nsa_step = build_glm5_nsa_step(self)
