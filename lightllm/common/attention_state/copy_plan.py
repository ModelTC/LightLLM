from dataclasses import dataclass

import torch

from .token_copy import TokenPageCopyPlan, TransferBatch
from .bounded_att.providers import bind_runtime_page_copy


@dataclass(frozen=True)
class CopyPlan:
    page_buffer: torch.Tensor
    token_plans: tuple[TokenPageCopyPlan, ...]
    state_copies: tuple

    @classmethod
    def bind(cls, full_states, bounded_states, page_buffer, token_layouts, state_layouts, write):
        tokens = tuple(
            TokenPageCopyPlan.bind(
                state.get_transfer_fields(), token_layouts[state.state_name], page_buffer, write, pad_tail=True
            )
            for state in full_states
        )
        states = tuple(
            bind_runtime_page_copy(state, page_buffer, state_layouts[state.state_name], write)
            for state in bounded_states
        )
        return cls(page_buffer, tokens, states)

    @classmethod
    def for_pd(cls, managers, page_buffer, write):
        if not managers:
            raise ValueError("a copy plan requires a local TP group")
        tokens, states = [], []
        for manager in managers:
            operator = manager.get_pd_operator()
            if operator.get_page_bytes() != page_buffer.shape[1]:
                raise ValueError("PD page layout does not match this group")
            plan = cls.bind(
                manager.get_full_attention_states(),
                manager.get_bounded_attention_states(),
                page_buffer,
                operator.get_pd_full_att_page_layouts(),
                operator.get_pd_bounded_att_page_layouts(),
                write,
            )
            tokens.extend(plan.token_plans)
            states.extend(plan.state_copies)
        return cls(page_buffer, tuple(tokens), tuple(states))

    def run(self, *, pd_page_index, task_kind="kv", token_indices=None, req_idx=None, state_token_len=None):
        if not 0 <= pd_page_index < len(self.page_buffer):
            raise ValueError("invalid PD page index")
        if task_kind == "kv":
            if token_indices is None:
                raise ValueError("KV copy requires token indices")
            if self.token_plans:
                batch = TransferBatch.prepare(token_indices, [pd_page_index])
                for plan in self.token_plans:
                    plan.validate_batch(batch)
                for plan in self.token_plans:
                    plan.run(batch)
        elif task_kind == "att_state":
            if req_idx is None or state_token_len is None or state_token_len < 0 or not self.state_copies:
                raise ValueError("state copy requires a complete logical recovery point")
            for copy in self.state_copies:
                copy(pd_page_index=pd_page_index, req_idx=req_idx, state_token_len=state_token_len)
        else:
            raise ValueError(f"unknown PD task kind {task_kind}")
