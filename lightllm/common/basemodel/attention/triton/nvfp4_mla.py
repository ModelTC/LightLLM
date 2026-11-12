"""Dense MLA on an NVFP4 latent, retaining RoPE in the computation dtype."""

import dataclasses
import weakref

import torch

from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from ...triton_kernel.mla_att.nvfp4 import nvfp4_mla_attention, prepare_prefill_queries


class NVFP4MlaTritonAttBackend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state):
        return NVFP4MlaPrefillState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state):
        return NVFP4MlaDecodeState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class NVFP4MlaPrefillState(BasePrefillAttState):
    query_req: torch.Tensor = None
    lengths: torch.Tensor = None
    current_row_table: torch.Tensor = None

    def init_state(self):
        state = self.infer_state
        self.query_req, self.lengths = prepare_prefill_queries(
            state.b_req_idx,
            state.b1_cu_q_seq_len,
            state.b_seq_len,
            state.input_ids.numel(),
            state.max_q_seq_len,
            self.backend.uses_causal_attention(),
        )

    def prefill_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert att_control.mla_prefill and v is None
        row_table = self.current_row_table
        if row_table is None:
            row_table = torch.full((k.shape[0],), -1, dtype=torch.int32, device=q[0].device)
            row_table.scatter_(
                0,
                self.infer_state.mem_index.long(),
                torch.arange(self.infer_state.mem_index.numel(), device=q[0].device, dtype=torch.int32),
            )
            # Shape probing captures a temporary graph without replaying it.
            # Only cache a mapping whose initialization actually executed.
            if not torch.cuda.is_current_stream_capturing():
                self.current_row_table = row_table
        table = self.infer_state.req_manager.req_to_token_indexs
        return nvfp4_mla_attention(
            q[0],
            k,
            self.infer_state.req_manager.req_to_token_indexs,
            att_control.mla_prefill_dict["softmax_scale"],
            q_rope=q[1],
            current_kv=att_control.mla_prefill_dict["prefill_cache_kv"],
            row_table=row_table,
            query_req=self.query_req,
            lengths=self.lengths,
            max_seq_len=min(self.infer_state.max_kv_seq_len, table.shape[1]),
        )


@dataclasses.dataclass
class NVFP4MlaDecodeState(BaseDecodeAttState):
    def init_state(self):
        pass

    def decode_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert att_control.mla_decode and v is None
        table = self.infer_state.req_manager.req_to_token_indexs
        return nvfp4_mla_attention(
            q[0],
            k,
            self.infer_state.req_manager.req_to_token_indexs,
            att_control.mla_decode_dict["softmax_scale"],
            q_rope=q[1],
            query_req=self.infer_state.b_req_idx,
            lengths=self.infer_state.b_seq_len,
            max_seq_len=min(self.infer_state.max_kv_seq_len, table.shape[1]),
        )
