from dataclasses import dataclass
from functools import partial

import torch

from .token_copy import TokenPageCopyPlan, TransferBatch


@dataclass(frozen=True)
class CopyPlan:

    page_buffer: torch.Tensor
    token_copies: tuple
    state_copies: tuple
    token_plans: tuple[TokenPageCopyPlan, ...] = ()

    @classmethod
    def for_pd(cls, managers, page_buffer, write):
        token_copies, state_copies, token_plans = [], [], []
        if not managers:
            raise ValueError("a copy plan requires a local TP group")
        for manager in managers:
            operator = manager.get_pd_operator()
            if operator.get_page_bytes() != page_buffer.shape[1]:
                raise ValueError("PD page layout does not match this group")
            groups = (
                (manager.get_full_attention_states(), operator.get_pd_full_att_page_layouts(), token_copies),
                (
                    manager.get_bounded_attention_states(),
                    operator.get_pd_bounded_att_page_layouts(),
                    state_copies,
                ),
            )
            for states, layouts, copies in groups:
                for state in states:
                    fields = state.get_transfer_fields() if copies is token_copies else None
                    if fields is not None:
                        token_plans.append(
                            TokenPageCopyPlan.bind(fields, layouts[state.state_name], page_buffer, write, pad_tail=True)
                        )
                        continue
                    if copies is token_copies:
                        run = state.write_to_pd_page if write else state.read_from_pd_page
                        copies.append(partial(run, pd_page_buffers=page_buffer, layout=layouts[state.state_name]))
                    else:
                        copies.append(state.bind_pd_copy(page_buffer, layouts[state.state_name], write))
        return cls(page_buffer, tuple(token_copies), tuple(state_copies), tuple(token_plans))

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
            for copy in self.token_copies:
                copy(pd_page_index=pd_page_index, token_indices=token_indices)
        elif task_kind == "att_state":
            if req_idx is None or state_token_len is None or state_token_len < 0 or not self.state_copies:
                raise ValueError("state copy requires a complete logical recovery point")
            for copy in self.state_copies:
                copy(pd_page_index=pd_page_index, req_idx=req_idx, state_token_len=state_token_len)
        else:
            raise ValueError(f"unknown PD task kind {task_kind}")
