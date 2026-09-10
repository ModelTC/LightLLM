import dataclasses
import torch
from typing import Tuple

from ..base_att import AttControl
from .fp import TritonAttBackend, TritonPrefillAttState, TritonDecodeAttState


class NeoTritonAttBackend(TritonAttBackend):
    def create_att_prefill_state(self, infer_state) -> "NeoTritonPrefillAttState":
        return NeoTritonPrefillAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class NeoTritonPrefillAttState(TritonPrefillAttState):
    def init_state(self):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        from ...triton_kernel.att.prefill_att.context_attention_fwd_neo import context_attention_fwd_neo
        from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo

        self.infer_state: NeoChatInferStateInfo

        out = alloc_func(q.shape, q.dtype)
        context_attention_fwd_neo(
            q,
            k,
            v,
            out,
            self.infer_state.b_req_idx,
            self.infer_state.b_q_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.max_q_seq_len,
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_image_token_end,
        )
        return out
