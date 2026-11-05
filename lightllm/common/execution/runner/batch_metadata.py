from dataclasses import dataclass

import torch
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_prefill_params
from lightllm.common.basemodel.triton_kernel.gen_decode_params import gen_decode_params


def prepare_sequence_metadata(state):
    if state.is_prefill:
        (
            state.b_q_seq_len,
            state.b1_cu_q_seq_len,
            state.b_kv_seq_len,
            state.b1_cu_kv_seq_len,
            state.position_ids,
        ) = gen_prefill_params(
            input_token_num=state.input_ids.shape[0],
            b_ready_cache_len=state.b_ready_cache_len,
            b_seq_len=state.b_seq_len,
        )
        state.b_q_start_loc = state.b1_cu_q_seq_len[0:-1]
    else:
        (
            state.b_q_seq_len,
            state.b1_cu_q_seq_len,
            state.b_kv_seq_len,
            state.b1_cu_kv_seq_len,
            state.position_ids,
        ) = gen_decode_params(state.b_seq_len)
        state.b_kv_start_loc = state.b1_cu_kv_seq_len[0:-1]


@dataclass(frozen=True)
class RotaryTables:
    cos: torch.Tensor
    sin: torch.Tensor

    def gather(self, positions):
        if positions.ndim == 1:
            return self.cos.index_select(0, positions), self.sin.index_select(0, positions)
        return self.cos[positions], self.sin[positions]


@dataclass(frozen=True)
class RotaryMetadata:
    tables: tuple[tuple[str, RotaryTables], ...]

    def __call__(self, state):
        prepare_sequence_metadata(state)
        if state.is_prefill:
            state.max_seq_len = state.max_kv_seq_len
        for suffix, tables in self.tables:
            cos, sin = tables.gather(state.position_ids)
            setattr(state, "position_cos" + suffix, cos.view(state.position_ids.shape[0], -1))
            setattr(state, "position_sin" + suffix, sin.view(state.position_ids.shape[0], -1))
