from dataclasses import dataclass
from functools import partial, singledispatch

import torch

from .conv_ssm_state import ConvSSMState
from .sliding_window.fp8_e8m0_swa_state import FP8E8M0SlidingWindowState
from .sliding_window.fp8_e8m0_swa_transfer import copy_swa_window
from .window_compressor.window_compressor_state import WindowCompressorState
from .window_compressor.window_compressor_transfer import copy_compressor_window
from .copy_kernel import bind_bounded_copy
from .copy_plan import BatchedStateView


@dataclass
class RecurrentStateOperations:
    runtime: BatchedStateView
    snapshots: dict
    conv: torch.Tensor
    ssm: torch.Tensor
    mtp_step: int
    copy: object

    def reset(self, req_indices):
        for req in req_indices:
            self.conv[:, req].zero_()
            start = req * (self.mtp_step + 1)
            self.ssm[:, start : start + self.mtp_step + 1].zero_()
        self.runtime.finish_restore(req_indices)

    def _snapshot(self, req_indices, state_token_lens, page_kind, page_slot_ids, write):
        if len(req_indices) != len(state_token_lens) or len(req_indices) != len(page_slot_ids):
            raise ValueError("request, length and page counts differ")
        pool = self.snapshots[page_kind]
        assert pool.fields, f"{page_kind} pages are disabled"
        fields = dict(pool.fields)
        offsets = self.runtime.offsets(req_indices, self.conv.device) if write else {}
        for name, field in self.runtime.fields:
            external = fields[name].tensor
            copies = [(field.tensor[req], external[slot]) for req, slot in zip(req_indices, page_slot_ids)]
            if not write:
                copies = [(dst, src) for src, dst in copies]
            self.copy(copies, source_byte_offsets=offsets.get(name))
        if not write:
            self.runtime.finish_restore(req_indices)

    def save(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        self._snapshot(req_indices, state_token_lens, page_kind, page_slot_ids, True)

    def restore(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        self._snapshot(req_indices, state_token_lens, page_kind, page_slot_ids, False)

    def commit(self, req_indices, accepted_steps):
        if req_indices.device != self.conv.device or accepted_steps.device != self.conv.device:
            raise ValueError("commit inputs must share the recurrent state device")
        if self.runtime.cursor is not None:
            self.runtime.cursor[req_indices] = accepted_steps - 1


@dataclass
class WindowStateOperations:
    buffer: torch.Tensor
    snapshots: dict
    kernel: object
    checkpoint_stride: int = 1
    checkpoint_required: bool = True
    gate_start: int | None = None

    def reset(self, req_indices):
        for req in req_indices:
            self.buffer[:, req].zero_()
            if self.gate_start is not None:
                self.buffer[:, req, :, self.gate_start :].fill_(float("-inf"))

    def _snapshot(self, req_indices, state_token_lens, page_kind, page_slot_ids, write):
        if len(req_indices) != len(state_token_lens) or len(req_indices) != len(page_slot_ids):
            raise ValueError("request, length and page counts differ")
        assert all(
            length % self.checkpoint_stride == 0 for length in state_token_lens
        ), "prefix must close a compressor stride"
        if not self.checkpoint_required:
            return
        pool = self.snapshots[page_kind]
        assert pool.fields, f"{page_kind} pages are disabled"
        external = pool.fields[0][1].tensor
        for req, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self.kernel(local=self.buffer[:, req], external=external[slot], state_token_len=length, write=write)

    def save(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        self._snapshot(req_indices, state_token_lens, page_kind, page_slot_ids, True)

    def restore(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        self._snapshot(req_indices, state_token_lens, page_kind, page_slot_ids, False)

    def commit(self, req_indices, accepted_steps):
        pass


@singledispatch
def bind_state_operations(state):
    raise NotImplementedError(f"no state lifecycle provider for {type(state).__name__}")


def _snapshots(state):
    return {kind: state.get_snapshot_transfer_view(kind) for kind in ("big", "small")}


@bind_state_operations.register(ConvSSMState)
def _bind_recurrent(state):
    return RecurrentStateOperations(
        state.get_runtime_transfer_view(),
        _snapshots(state),
        state.conv_state,
        state.ssm_state,
        state.mtp_step,
        bind_bounded_copy(state.conv_state.device),
    )


@bind_state_operations.register(FP8E8M0SlidingWindowState)
def _bind_sliding_window(state):
    return WindowStateOperations(state.swa_buffer, _snapshots(state), partial(copy_swa_window, config=state.config))


@bind_state_operations.register(WindowCompressorState)
def _bind_compressor(state):
    return WindowStateOperations(
        state.buffer,
        _snapshots(state),
        copy_compressor_window,
        state.config.compress_stride,
        bool(state.config.overlap_tokens),
        state.config.candidate_dim,
    )
