"""Worker-local recurrent state shared by attention, checkpoints, and PD I/O.

The runner orders operations on its current CUDA stream. This component does
not allocate request IDs, choose prefix boundaries, or synchronize streams.
Attention backends retain responsibility for recurrent computation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from lightllm.common.state_cache_manager import LayerCache, LinearAttCacheConfig

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
    """Own request state and its lifecycle, without dispatching attention.

    PD accesses this object through the shared request manager's interfaces.
    Keep its fields limited to tensors, cache operators, and pickleable
    static configuration. Graphs, streams, callbacks, batch contexts, and pinned
    CPU checkpoint pools belong to their existing worker/runner owners. Adding
    such resources requires an explicit transport view rather than serializing
    them implicitly with this object.
    """

    def __init__(self, max_request_num, linear_config: LinearAttCacheConfig, args, *, kda=False, device="cuda"):
        self.linear_config = linear_config
        self.mtp_step = args.mtp_step
        self.ssm_slots_per_req = self.mtp_step + 1 if args.ssm_state_mode == "native" else 1
        self.accepted_state_index = (
            torch.zeros(max_request_num + 1, dtype=torch.int32, device=device) if self.mtp_step > 0 else None
        )
        self.conv_state = LayerCache(
            size=max_request_num + 1,
            dtype=linear_config.conv_state_dtype,
            shape=linear_config.get_mtp_conv_state_shape(mtp_step=self.mtp_step),
            layer_num=linear_config.linear_layer_num,
            device=device,
        )
        self.ssm_state = LayerCache(
            size=(max_request_num + 1) * self.ssm_slots_per_req,
            dtype=linear_config.ssm_state_dtype,
            shape=linear_config.get_ssm_state_shape(),
            layer_num=linear_config.linear_layer_num,
            device=device,
        )
        self.ssm_backend: Optional[Union["ReplaySSMBackend", "CompactSSMBackend"]] = None
        if args.ssm_state_mode == "replay":
            from lightllm.common.basemodel.triton_kernel.linear_att import replayssm

            self.ssm_backend = replayssm.ReplaySSMBackend(
                self.ssm_state.buffer,
                args.ssm_replay_config.get("cache_len", 16),
                self.mtp_step + 1,
                linear_config.conv_state_dtype,
                num_key_heads=linear_config.num_linear_k_heads,
                kda=kda,
                gate_lower_bound=linear_config.gate_lower_bound,
                compute_mode=args.ssm_replay_config.get("compute_mode", "inline"),
                fold_method=args.ssm_replay_config.get("fold_method", "raw"),
            )
        elif args.ssm_state_mode == "compact":
            from lightllm.common.basemodel.triton_kernel.linear_att import compact_ssm

            self.ssm_backend = compact_ssm.CompactSSMBackend(
                self.ssm_state.buffer,
                self.mtp_step + 1,
                linear_config.conv_state_dtype,
                num_key_heads=linear_config.num_linear_k_heads,
                kda=kda,
                gate_lower_bound=linear_config.gate_lower_bound,
            )

    def _reset_tracking(self, req_idx):
        if self.ssm_backend is not None:
            self.ssm_backend.clear_history(req_idx)
        if self.accepted_state_index is not None:
            self.accepted_state_index[req_idx].zero_()

    def reset(self, req_idx):
        """Initialize a newly allocated/reused request, including speculative slots."""
        self._reset_tracking(req_idx)
        self.conv_state.buffer[:, req_idx].zero_()
        start = req_idx * self.ssm_slots_per_req
        self.ssm_state.buffer[:, start : start + self.ssm_slots_per_req].zero_()

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
        accepted_token_counts = self.accepted_state_index[reqs] + 1
        accepted_token_counts = torch.where(reqs == self.accepted_state_index.numel() - 1, 1, accepted_token_counts)
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
            req_to_mtp_state_index=self.accepted_state_index,
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
            self.ssm_backend.accept_updates(reqs, self.accepted_state_index)
            return
        assert sum(size for size, _ in linear_state_batches) == reqs.numel()
        start = 0
        for size, commit in linear_state_batches:
            if size:
                self.ssm_backend.accept_updates(
                    reqs[start : start + size],
                    self.accepted_state_index,
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
        width = self.linear_config.get_conv_state_shape()[-1]
        conv = self.conv_state.buffer[:, req_idx, ..., :width]
        if self.accepted_state_index is not None:
            offsets = torch.arange(width, device=conv.device) + self.accepted_state_index[req_idx]
            conv = self.conv_state.buffer[:, req_idx].index_select(-1, offsets)
        if self.ssm_backend is not None:
            ssm = self.ssm_backend.get_accepted_state(req_idx)
        elif self.ssm_slots_per_req > 1:
            index = req_idx * self.ssm_slots_per_req + self.accepted_state_index[req_idx]
            ssm = self.ssm_state.buffer.index_select(1, index.long().view(1))[:, 0]
        else:
            ssm = self.ssm_state.buffer[:, req_idx]
        return LinearStateTensors(conv, ssm)

    def prepare_restore(self, req_idx):
        """Invalidate old history and return canonical destinations for restore.

        The caller must fill both tensors before the next forward. This lets PD
        transport unpack TP shards directly into the pool without a staging copy.
        The same stream/completion rules as ``get_accepted_state`` apply.
        """
        self._reset_tracking(req_idx)
        width = self.linear_config.get_conv_state_shape()[-1]
        return LinearStateTensors(
            self.conv_state.buffer[:, req_idx, ..., :width],
            self.ssm_state.buffer[:, req_idx * self.ssm_slots_per_req],
        )

    def restore(self, req_idx, conv, ssm):
        target = self.prepare_restore(req_idx)
        target.conv.copy_(conv, non_blocking=True)
        target.ssm.copy_(ssm, non_blocking=True)

    def save_checkpoints(self, b_req_idx, req_indexes, buffer_indexes, destination):
        """Copy current accepted prefixes into caller-allocated checkpoint slots."""
        if self.ssm_backend is None:
            from lightllm.common.basemodel.triton_kernel.linear_att_copy import copy_linear_att_state_to_kv_buffer

            indexes = torch.tensor(buffer_indexes, dtype=torch.int32, device="cpu").to(
                b_req_idx.device, non_blocking=True
            )
            copy_linear_att_state_to_kv_buffer(
                b_req_idx=b_req_idx,
                big_page_buffer_ids=indexes,
                gpu_conv_state=self.conv_state.buffer,
                gpu_ssm_state=self.ssm_state.buffer,
                cpu_kv_conv_state=destination.conv_state_cache.buffer,
                cpu_kv_ssm_state=destination.ssm_state_cache.buffer,
                mtp_step=self.ssm_slots_per_req - 1,
                accepted_state_indices=self.accepted_state_index,
            )
        else:
            for req_idx, buffer_idx in zip(req_indexes, buffer_indexes):
                if buffer_idx != -1:
                    self.copy_accepted_state_to(req_idx, *destination.get_state_cache(buffer_idx))

    def copy_accepted_state_to(self, req_idx, conv, ssm):
        accepted_state = self.get_accepted_state(req_idx)
        conv.copy_(accepted_state.conv, non_blocking=True)
        ssm.copy_(accepted_state.ssm, non_blocking=True)

    def get_layer_buffers(self, layer_idx_in_all):
        linear_layer_idx = self.linear_config.get_linear_layer_index(layer_idx_in_all)
        return self.conv_state.buffer[linear_layer_idx], self.ssm_state.buffer[linear_layer_idx]
