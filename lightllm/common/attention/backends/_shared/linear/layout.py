from dataclasses import dataclass, field

import torch

from lightllm.common.attention.backends.base import AttentionExecutionContext

from .mtp_state_params import build_dynamic_mtp_layout


@dataclass(slots=True)
class LinearDecodeContext(AttentionExecutionContext):
    b_conv_buffer_idx: torch.Tensor | None = None
    b_ssm_buffer_idx: torch.Tensor | None = None
    b1_mtp_cu_q_seq_len: torch.Tensor | None = None
    b_cu_seqlens: torch.Tensor | None = None
    accepted_by_state: dict[int, torch.Tensor] = field(default_factory=dict)

    def accepted_tokens(self, state_index: torch.Tensor) -> torch.Tensor:
        """Gather once per State per update; Graph captures the gather for every replay.

        Layer bindings retain the source tensors, so their identities remain valid for
        this context. HOLD sequences must select the initial convolution state.
        """
        key = id(state_index)
        if key not in self.accepted_by_state:
            self.accepted_by_state[key] = torch.where(
                self.b_conv_buffer_idx == self.batch.hold_req_id,
                1,
                state_index[self.b_conv_buffer_idx] + 1,
            )
        return self.accepted_by_state[key]


def update_linear_decode(context: LinearDecodeContext, draft_step: int, dynamic: bool, *, replay: bool) -> None:
    batch = context.batch
    context.accepted_by_state.clear()
    if draft_step == 0:
        context.b_conv_buffer_idx = batch.b_req_idx
        context.b_ssm_buffer_idx = batch.b_req_idx
        context.b1_mtp_cu_q_seq_len = None
        if replay:
            context.b_cu_seqlens = torch.arange(batch.batch_size + 1, dtype=torch.int32, device=batch.device)
        return
    width = draft_step + 1
    if dynamic:
        context.b1_mtp_cu_q_seq_len, context.b_conv_buffer_idx = build_dynamic_mtp_layout(
            b_req_idx=batch.b_req_idx,
            b_mtp_index=batch.b_mtp_index,
            hold_req_id=batch.hold_req_id,
        )
    else:
        if batch.batch_size % width:
            raise ValueError("linear decode batch size must be divisible by draft_step + 1")
        context.b1_mtp_cu_q_seq_len = torch.arange(
            0,
            batch.batch_size + 1,
            width,
            dtype=torch.int32,
            device=batch.device,
        )
        context.b_conv_buffer_idx = batch.b_req_idx.view(-1, width)[:, 0].contiguous()
    if replay:
        context.b_ssm_buffer_idx = context.b_conv_buffer_idx
        context.b_cu_seqlens = context.b1_mtp_cu_q_seq_len
    else:
        offsets = torch.arange(width, dtype=batch.b_req_idx.dtype, device=batch.device)
        context.b_ssm_buffer_idx = context.b_conv_buffer_idx[:, None] * width + offsets
