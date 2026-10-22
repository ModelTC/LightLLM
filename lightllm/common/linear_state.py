"""Worker-local recurrent state shared by attention, checkpoints, and PD I/O.

The runner orders operations on its current CUDA stream. This component does
not allocate request IDs, choose prefix boundaries, or synchronize streams.
Attention backends retain responsibility for recurrent computation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence, Tuple, Union

import torch


if TYPE_CHECKING:
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMBackend
    from lightllm.common.basemodel.triton_kernel.linear_att.compact_ssm import CompactSSMBackend


@dataclass(frozen=True, slots=True)
class SSMRunConfig:
    BV: int
    num_warps: int
    num_stages: int = 3

    def __post_init__(self):
        for value in (self.BV, self.num_warps, self.num_stages):
            if type(value) is not int:
                raise TypeError("SSM kernel configuration must be an integer")


@dataclass(frozen=True, slots=True)
class LinearStateCommit:
    _run_config: SSMRunConfig


class LinearStateTensors(NamedTuple):
    conv: torch.Tensor
    ssm: torch.Tensor


class LinearStateManager:
    """Shared Conv/SSM lifecycle for GDN and KDA attention resources."""

    def __init__(
        self,
        config,
        request_capacity,
        mtp_step=0,
        device="cuda",
        *,
        ssm_state_mode="native",
        ssm_replay_config=None,
        kda=False,
        gate_lower_bound=-5.0,
    ):
        self.config = config
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self.ssm_slots_per_req = mtp_step + 1 if ssm_state_mode == "native" else 1
        self.mtp_state_index = torch.zeros(request_capacity, dtype=torch.int32, device=device) if mtp_step else None
        self._logical_to_storage_layer = {layer: index for index, layer in enumerate(config.logical_layer_ids)}
        layers = len(config.logical_layer_ids)
        self.conv_state = torch.zeros(
            (layers, request_capacity, config.conv_channels, config.conv_width + mtp_step),
            dtype=config.conv_dtype,
            device=device,
        )
        self.ssm_state = torch.zeros(
            (layers, request_capacity * self.ssm_slots_per_req, config.num_v_heads, config.key_dim, config.value_dim),
            dtype=config.ssm_dtype,
            device=device,
        )
        ssm_replay_config = ssm_replay_config or {}
        self.ssm_backend: Optional[Union["ReplaySSMBackend", "CompactSSMBackend"]] = None
        if ssm_state_mode == "replay":
            from lightllm.common.basemodel.triton_kernel.linear_att import replayssm

            self.ssm_backend = replayssm.ReplaySSMBackend(
                self.ssm_state,
                ssm_replay_config.get("cache_len", 16),
                self.mtp_step + 1,
                config.conv_dtype,
                num_key_heads=config.num_k_heads,
                kda=kda,
                gate_lower_bound=gate_lower_bound,
                compute_mode=ssm_replay_config.get("compute_mode", "inline"),
                fold_method=ssm_replay_config.get("fold_method", "raw"),
            )
        elif ssm_state_mode == "compact":
            from lightllm.common.basemodel.triton_kernel.linear_att import compact_ssm

            self.ssm_backend = compact_ssm.CompactSSMBackend(
                self.ssm_state,
                self.mtp_step + 1,
                config.conv_dtype,
                num_key_heads=config.num_k_heads,
                kda=kda,
                gate_lower_bound=gate_lower_bound,
            )

    def _reset_tracking(self, req_idx):
        if self.ssm_backend is not None:
            self.ssm_backend.clear_history(req_idx)
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_idx].zero_()

    def reset(self, req_idx):
        """Initialize a newly allocated/reused request, including speculative slots."""
        self._reset_tracking(req_idx)
        self.conv_state[:, req_idx].zero_()
        start = req_idx * self.ssm_slots_per_req
        self.ssm_state[:, start : start + self.ssm_slots_per_req].zero_()

    def prepare_prefill(self, reqs):
        """Merge accepted history and return convolution and SSM buffer indexes."""
        if self.ssm_backend is not None:
            self.ssm_backend.merge_accepted_updates(reqs)
        return reqs, reqs * self.ssm_slots_per_req

    def prepare_decode(self, reqs, cu_seqlens=None, *, single_token=False):
        if self.ssm_backend is not None:
            self.ssm_backend.prepare_decode(reqs, cu_seqlens)
            if single_token:
                # Preserve the forward cursor snapshot before accepting the
                # guaranteed token, including when forward is graph-replayed.
                self.ssm_backend.accept_updates(reqs)

    def prepare_mtp_decode(self, reqs, cu_seqlens):
        """Return Conv/SSM indexes and prior accepted token counts; prepare replay history."""
        accepted_token_counts = self.mtp_state_index[reqs] + 1
        accepted_token_counts = torch.where(reqs == self.mtp_state_index.numel() - 1, 1, accepted_token_counts)
        if self.ssm_backend is not None:
            ssm_indexes = reqs
        else:
            offsets = torch.arange(self.ssm_slots_per_req, device=reqs.device, dtype=reqs.dtype)
            ssm_indexes = reqs[:, None] * self.ssm_slots_per_req + offsets[None, :]
        self.prepare_decode(reqs, cu_seqlens)
        return reqs, ssm_indexes, accepted_token_counts

    def create_commit(self, run_config: Optional[SSMRunConfig]) -> Optional[LinearStateCommit]:
        from lightllm.common.basemodel.triton_kernel.linear_att.compact_ssm import CompactSSMBackend

        if run_config is not None and isinstance(self.ssm_backend, CompactSSMBackend):
            return LinearStateCommit(run_config)
        return None

    def commit_accepted_state(
        self,
        b_req_mtp_start_loc,
        b_req_idx,
        b_mtp_index,
        accepted_index,
        verify_width,
        linear_state_batches: Optional[Sequence[Tuple[int, Optional[LinearStateCommit]]]] = None,
    ):
        """Commit accepted prefixes without requiring full-state materialization.

        ``linear_state_batches`` partitions logical requests in the same order as the MTP
        start locations. Each commit belongs to the forward that wrote that
        partition's scratch; padding rows do not participate in acceptance.
        """
        from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update
        from lightllm.common.basemodel.triton_kernel.linear_att import replayssm

        replay_cursors = None
        if isinstance(self.ssm_backend, replayssm.ReplaySSMBackend) and self.ssm_backend.verify_width > 1:
            replay_cursors = self.ssm_backend.cursors
        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=self.mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
            replay_cursors=replay_cursors,
        )
        if self.ssm_backend is None or replay_cursors is not None:
            return
        reqs = b_req_idx[b_req_mtp_start_loc.long()]
        if linear_state_batches is None or isinstance(self.ssm_backend, replayssm.ReplaySSMBackend):
            self.ssm_backend.accept_updates(reqs, self.mtp_state_index)
            return
        assert sum(size for size, _ in linear_state_batches) == reqs.numel()
        start = 0
        for size, commit in linear_state_batches:
            if size:
                self.ssm_backend.accept_updates(
                    reqs[start : start + size],
                    self.mtp_state_index,
                    run_config=commit._run_config if commit is not None else None,
                )
            start += size

    def get_accepted_state(self, req_idx):
        """Return the current accepted prefix without mutating active state.

        The caller associates this prefix with its cache/transport token boundary.
        These tensors may alias the active pool: enqueue consuming copies on the
        current stream before advancing/releasing the request. Cross-stream or
        cross-process callers must use their existing completion events before
        recycling sources or destinations. No CPU transfer or sync is implicit.
        """
        width = self.config.conv_width
        conv = self.conv_state[:, req_idx, ..., :width]
        if self.mtp_state_index is not None:
            offsets = torch.arange(width, device=conv.device) + self.mtp_state_index[req_idx]
            conv = self.conv_state[:, req_idx].index_select(-1, offsets)
        if self.ssm_backend is not None:
            ssm = self.ssm_backend.get_accepted_state(req_idx)
        elif self.ssm_slots_per_req > 1:
            index = req_idx * self.ssm_slots_per_req + self.mtp_state_index[req_idx]
            ssm = self.ssm_state.index_select(1, index.long().view(1))[:, 0]
        else:
            ssm = self.ssm_state[:, req_idx]
        return LinearStateTensors(conv, ssm)

    def prepare_restore(self, req_idx):
        """Invalidate old history and return canonical destinations for restore.

        The caller must fill both tensors before the next forward. This lets PD
        transport unpack TP shards directly into the pool without a staging copy.
        The same stream/completion rules as ``get_accepted_state`` apply.
        """
        self._reset_tracking(req_idx)
        width = self.config.conv_width
        return LinearStateTensors(
            self.conv_state[:, req_idx, ..., :width],
            self.ssm_state[:, req_idx * self.ssm_slots_per_req],
        )

    def restore(self, req_idx, conv, ssm):
        target = self.prepare_restore(req_idx)
        target.conv.copy_(conv, non_blocking=True)
        target.ssm.copy_(ssm, non_blocking=True)

    def get_storage_layer_index(self, logical_layer_id):
        return self._logical_to_storage_layer[logical_layer_id]


def get_linear_state(infer_state, state_class):
    (state,) = (
        state
        for state in infer_state.att_state_manager.get_bounded_attention_states(infer_state.att_state_role)
        if isinstance(state, state_class)
    )
    return state
