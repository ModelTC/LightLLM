"""Worker-local recurrent state shared by attention, checkpoints, and PD I/O.

The runner orders operations on its current CUDA stream. This component does
not allocate request IDs, choose prefix boundaries, or synchronize streams.
Attention backends retain responsibility for recurrent computation.
"""

from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from lightllm.common.state_cache_manager import LayerCache, LinearAttCacheConfig
from lightllm.common.recurrent_state_types import RecurrentStateContext, SSMUpdateBackend

if TYPE_CHECKING:
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache as _ReplaySSMCache
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache as _CompactSSMCache


class RecurrentStateSnapshot(NamedTuple):
    conv: torch.Tensor
    ssm: torch.Tensor


class RecurrentState:
    """Own request state and its lifecycle, without dispatching attention.

    PD currently shares this object through CUDA IPC as part of the memory
    manager. Keep its fields limited to tensors, cache operators, and pickleable
    static configuration. Graphs, streams, callbacks, batch contexts, and pinned
    CPU checkpoint pools belong to their existing worker/runner owners. Adding
    such resources requires an explicit transport view rather than serializing
    them implicitly with this object.
    """

    def __init__(self, max_request_num, linear_config: LinearAttCacheConfig, args, *, kda=False, device="cuda"):
        self.linear_config = linear_config
        self.mtp_step = args.mtp_step
        self.slots_per_req = self.mtp_step + 1 if args.ssm_state_mode == "native" else 1
        self.accepted_index = (
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
            size=(max_request_num + 1) * self.slots_per_req,
            dtype=linear_config.ssm_state_dtype,
            shape=linear_config.get_ssm_state_shape(),
            layer_num=linear_config.linear_layer_num,
            device=device,
        )
        self.updates: Optional[Union["_ReplaySSMCache", "_CompactSSMCache"]] = None
        if args.ssm_state_mode == "replay":
            from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache

            self.updates = ReplaySSMCache(
                self.ssm_state.buffer,
                args.ssm_replay_config.get("cache_len", 16),
                self.mtp_step + 1,
                linear_config.conv_state_dtype,
                num_key_heads=linear_config.num_linear_k_heads,
                kda=kda,
                lower_bound=linear_config.gate_lower_bound,
                projection_mode=args.ssm_replay_config.get("compute_mode", "inline"),
                fold_method=args.ssm_replay_config.get("fold_method", "raw"),
            )
        elif args.ssm_state_mode == "compact":
            from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache

            self.updates = CompactSSMCache(
                self.ssm_state.buffer,
                self.mtp_step + 1,
                linear_config.conv_state_dtype,
                num_key_heads=linear_config.num_linear_k_heads,
                kda=kda,
                lower_bound=linear_config.gate_lower_bound,
            )

    @property
    def update_backend(self) -> Optional[SSMUpdateBackend]:
        """Optional compact/replay execution backend for attention to call."""
        return self.updates

    def _clear_positions(self, req_idx):
        if self.updates is not None:
            self.updates.clear_history(req_idx)
        if self.accepted_index is not None:
            self.accepted_index[req_idx].zero_()

    def reset(self, req_idx):
        """Initialize a newly allocated/reused request, including speculative slots."""
        self._clear_positions(req_idx)
        self.conv_state.buffer[:, req_idx].zero_()
        start = req_idx * self.slots_per_req
        self.ssm_state.buffer[:, start : start + self.slots_per_req].zero_()

    def prepare_prefill(self, reqs):
        if self.updates is not None:
            self.updates.merge_accepted_updates(reqs)
        return reqs * self.slots_per_req

    def prepare_decode(self, reqs, cu_seqlens=None, *, single_token=False):
        if self.updates is not None:
            self.updates.prepare_decode(reqs, cu_seqlens)
            if single_token:
                # Preserve the forward cursor snapshot before accepting the
                # guaranteed token, including when forward is graph-replayed.
                self.updates.accept_updates(reqs)

    def commit(
        self,
        b_req_mtp_start_loc,
        b_req_idx,
        b_mtp_index,
        accepted_index,
        verify_width,
        batches: Optional[Sequence[Tuple[int, Optional[RecurrentStateContext]]]] = None,
    ):
        """Commit accepted prefixes without requiring full-state materialization.

        ``batches`` partitions logical requests in the same order as the MTP
        start locations. Each context belongs to the forward that wrote that
        partition's scratch; padding rows do not participate in acceptance.
        """
        from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update
        from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache

        replay_cursors = None
        if isinstance(self.updates, ReplaySSMCache) and self.updates.verify_width > 1:
            replay_cursors = self.updates.cursors
        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=self.accepted_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
            replay_cursors=replay_cursors,
        )
        if self.updates is None or replay_cursors is not None:
            return
        reqs = b_req_idx[b_req_mtp_start_loc.long()]
        if batches is None or isinstance(self.updates, ReplaySSMCache):
            self.updates.accept_updates(reqs, self.accepted_index)
            return
        assert sum(size for size, _ in batches) == reqs.numel()
        start = 0
        for size, context in batches:
            if size:
                self.updates.accept_updates(
                    reqs[start : start + size],
                    self.accepted_index,
                    run_config=context.config if context is not None else None,
                )
            start += size

    def snapshot(self, req_idx):
        """Export the current accepted prefix without mutating active state.

        The caller associates this prefix with its cache/transport token boundary.
        These tensors may alias the active pool: enqueue consuming copies on the
        current stream before advancing/releasing the request. Cross-stream or
        cross-process callers must use their existing completion events before
        recycling sources or destinations. No CPU transfer or sync is implicit.
        """
        width = self.linear_config.get_conv_state_shape()[-1]
        conv = self.conv_state.buffer[:, req_idx, ..., :width]
        if self.accepted_index is not None:
            offsets = torch.arange(width, device=conv.device) + self.accepted_index[req_idx]
            conv = self.conv_state.buffer[:, req_idx].index_select(-1, offsets)
        if self.updates is not None:
            ssm = self.updates.snapshot_accepted_state(req_idx)
        elif self.slots_per_req > 1:
            index = req_idx * self.slots_per_req + self.accepted_index[req_idx]
            ssm = self.ssm_state.buffer.index_select(1, index.long().view(1))[:, 0]
        else:
            ssm = self.ssm_state.buffer[:, req_idx]
        return RecurrentStateSnapshot(conv, ssm)

    def restore_target(self, req_idx):
        """Invalidate old history and return canonical destinations for restore.

        The caller must fill both tensors before the next forward. This lets PD
        transport unpack TP shards directly into the pool without a staging copy.
        The same stream/completion rules as ``snapshot`` apply.
        """
        self._clear_positions(req_idx)
        width = self.linear_config.get_conv_state_shape()[-1]
        return RecurrentStateSnapshot(
            self.conv_state.buffer[:, req_idx, ..., :width],
            self.ssm_state.buffer[:, req_idx * self.slots_per_req],
        )

    def restore(self, req_idx, conv, ssm):
        target = self.restore_target(req_idx)
        target.conv.copy_(conv, non_blocking=True)
        target.ssm.copy_(ssm, non_blocking=True)

    def save_checkpoints(self, b_req_idx, req_indexes, buffer_indexes, destination):
        """Copy current accepted prefixes into caller-allocated checkpoint slots."""
        if self.updates is None:
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
                mtp_step=self.slots_per_req - 1,
                conv_offsets=self.accepted_index,
            )
        else:
            for req_idx, buffer_idx in zip(req_indexes, buffer_indexes):
                if buffer_idx != -1:
                    self.export(req_idx, *destination.get_state_cache(buffer_idx))

    def export(self, req_idx, conv, ssm):
        snapshot = self.snapshot(req_idx)
        conv.copy_(snapshot.conv, non_blocking=True)
        ssm.copy_(snapshot.ssm, non_blocking=True)

    def layer_buffers(self, layer_idx_in_all):
        layer = self.linear_config.get_linear_layer_index(layer_idx_in_all)
        return self.conv_state.buffer[layer], self.ssm_state.buffer[layer]
