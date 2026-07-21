from __future__ import annotations

from dataclasses import dataclass

import torch

from .objs import ActionTaskIdentity, PrefixContextIdentity


@dataclass(frozen=True)
class RegisteredPrefixMapping:
    prefix_mem_indexes: torch.Tensor
    scratch_mem_indexes: torch.Tensor
    prefix_seq_lens: torch.Tensor


class ActionPrefixContextCache:
    """Worker-local metadata cache for the high-frequency action path.

    It stores indexes only; ownership of the physical KV remains in the target
    runtime.  A full, versioned context identity prevents a delayed REPLACE or
    CLOSE from aliasing a newer mapping.
    """

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self._device = torch.device(device)
        self._mappings: dict[PrefixContextIdentity, RegisteredPrefixMapping] = {}

    def _indexes(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device=self._device, dtype=torch.int32).contiguous().clone()

    @staticmethod
    def _lengths(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu", dtype=torch.int32).reshape(-1).clone()

    def register(
        self,
        identity: PrefixContextIdentity,
        *,
        prefix_mem_indexes: torch.Tensor,
        scratch_mem_indexes: torch.Tensor,
        prefix_seq_lens: torch.Tensor,
    ) -> RegisteredPrefixMapping:
        candidate = RegisteredPrefixMapping(
            prefix_mem_indexes=self._indexes(prefix_mem_indexes),
            scratch_mem_indexes=self._indexes(scratch_mem_indexes).reshape(-1),
            prefix_seq_lens=self._lengths(prefix_seq_lens),
        )
        current = self._mappings.get(identity)
        if current is not None:
            # Tensor dataclass equality is not scalar-valued, so compare every
            # field explicitly before accepting an idempotent retry.
            if not (
                torch.equal(
                    current.prefix_mem_indexes,
                    candidate.prefix_mem_indexes,
                )
                and torch.equal(
                    current.scratch_mem_indexes,
                    candidate.scratch_mem_indexes,
                )
                and torch.equal(
                    current.prefix_seq_lens,
                    candidate.prefix_seq_lens,
                )
            ):
                raise RuntimeError("prefix context identity was reused with different KV mappings")
            return current
        self._mappings[identity] = candidate
        return candidate

    def resolve(
        self,
        identity: PrefixContextIdentity,
        *,
        prefix_seq_lens: torch.Tensor,
        suffix_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            mapping = self._mappings[identity]
        except KeyError as exc:
            raise RuntimeError(f"prefix context {identity} is not registered in the action worker") from exc
        requested_lens = self._lengths(prefix_seq_lens)
        if not torch.equal(mapping.prefix_seq_lens, requested_lens):
            raise RuntimeError("prefix context sequence length changed after registration")
        if suffix_length <= 0 or suffix_length > mapping.scratch_mem_indexes.numel():
            raise ValueError("action suffix exceeds registered context scratch")
        return (
            mapping.prefix_mem_indexes,
            mapping.scratch_mem_indexes[:suffix_length],
        )

    def release(self, identity: PrefixContextIdentity) -> bool:
        return self._mappings.pop(identity, None) is not None

    def __contains__(self, identity: PrefixContextIdentity) -> bool:
        return identity in self._mappings


class _ScopedKVWriteOperator:
    def __init__(self, view: "ScopedKVMemoryView"):
        self.view = view

    def copy_kv_to_mem_manager(self, *, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor) -> None:
        lease = self.view._scratch_lease
        if lease is None:
            raise RuntimeError("actionserver attempted KV write without a lease")
        indexes = mem_index.reshape(-1)
        # Value comparisons on CUDA tensors force a device-to-host sync, so
        # layers validate the task-scoped lease by tensor identity.
        if (
            indexes.device != lease.device
            or indexes.dtype != lease.dtype
            or indexes.shape != lease.shape
            or indexes.data_ptr() != lease.data_ptr()
        ):
            raise RuntimeError("actionserver attempted to write outside scratch KV")
        self.view._shared_mem_manager.operator.copy_kv_to_mem_manager(
            layer_index=layer_index, mem_index=mem_index, kv=kv
        )


class ScopedKVMemoryView:
    """Read-only target KV plus an action-owned logical request table.

    The physical KV buffer is shared with the target model, but action suffix
    indexes are installed only in ``req_to_token_indexs`` owned by this view.
    This keeps text decode free to extend the target model's logical row.
    """

    def __init__(self, shared_mem_manager):
        self._shared_mem_manager = shared_mem_manager
        self.kv_buffer = shared_mem_manager.kv_buffer
        self.req_to_token_indexs = torch.full_like(shared_mem_manager.req_to_token_indexs, -1)
        self.head_num = shared_mem_manager.head_num
        self.head_dim = shared_mem_manager.head_dim
        self.layer_num = shared_mem_manager.layer_num
        self.dtype = shared_mem_manager.dtype
        self._scratch_lease: torch.Tensor | None = None
        self._active_identity: ActionTaskIdentity | None = None
        self._active_rows: torch.Tensor | None = None
        self.operator = _ScopedKVWriteOperator(self)

    def get_att_input_params(self, layer_index: int):
        return self._shared_mem_manager.get_att_input_params(layer_index)

    def begin_task_mapping(
        self,
        *,
        identity: ActionTaskIdentity,
        target_req_indexes: torch.Tensor,
        prefix_seq_lens: torch.Tensor,
        scratch_mem_indexes: torch.Tensor,
        prefix_mem_indexes: torch.Tensor,
        action_req_indexes: torch.Tensor,
    ) -> torch.Tensor:
        """Install one task in the action-owned logical mapping."""

        if self._active_identity is not None or self._scratch_lease is not None:
            raise RuntimeError("an action KV task is already active")

        device = self.req_to_token_indexs.device
        target_rows = target_req_indexes.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        prefix_lens = prefix_seq_lens.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        action_rows = action_req_indexes.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        batch_size = target_rows.numel()
        if batch_size == 0:
            raise ValueError("an action KV task must contain at least one row")
        if prefix_lens.numel() != batch_size or action_rows.numel() != batch_size:
            raise ValueError("action KV task metadata has inconsistent batch sizes")
        if torch.unique(action_rows).numel() != batch_size:
            raise ValueError("action logical request rows must be unique")
        table_rows, table_width = self.req_to_token_indexs.shape
        if bool(torch.any(target_rows < 0)) or bool(torch.any(target_rows >= table_rows)):
            raise IndexError("target request row is out of range")
        if bool(torch.any(action_rows < 0)) or bool(torch.any(action_rows >= table_rows)):
            raise IndexError("action request row is out of range")
        if bool(torch.any(prefix_lens <= 0)):
            raise ValueError("action prefix lengths must be positive")

        scratch = scratch_mem_indexes.to(device=device, dtype=torch.int32, non_blocking=True).reshape(-1)
        if scratch.numel() % batch_size != 0:
            raise ValueError("scratch indexes cannot be divided across action rows")
        suffix_len = scratch.numel() // batch_size
        if suffix_len == 0:
            raise ValueError("action scratch lease must not be empty")
        if bool(torch.any(prefix_lens + suffix_len > table_width)):
            raise ValueError("action prefix and suffix exceed the logical table")

        prefix_rows = self._prefix_rows(
            prefix_lens=prefix_lens,
            prefix_mem_indexes=prefix_mem_indexes,
        )
        try:
            # Clear the whole selected row so a shorter reuse cannot observe
            # logical indexes left by an earlier generation.
            self.req_to_token_indexs[action_rows] = -1
            for batch_index, (row, prefix_len) in enumerate(
                zip(action_rows.tolist(), prefix_lens.tolist(), strict=True)
            ):
                self.req_to_token_indexs[row, :prefix_len].copy_(prefix_rows[batch_index])
                scratch_start = batch_index * suffix_len
                self.req_to_token_indexs[row, prefix_len : prefix_len + suffix_len].copy_(
                    scratch[scratch_start : scratch_start + suffix_len]
                )
            self._scratch_lease = scratch
            self._active_identity = identity
            self._active_rows = action_rows
        except Exception:
            self.req_to_token_indexs[action_rows] = -1
            self._scratch_lease = None
            self._active_identity = None
            self._active_rows = None
            raise
        return action_rows.to(dtype=torch.int32)

    def _prefix_rows(
        self,
        *,
        prefix_lens: torch.Tensor,
        prefix_mem_indexes: torch.Tensor,
    ) -> list[torch.Tensor]:
        prefix = prefix_mem_indexes.to(
            device=self.req_to_token_indexs.device,
            dtype=torch.int32,
            non_blocking=True,
        )
        lengths = prefix_lens.tolist()
        if prefix.ndim == 2:
            if prefix.shape[0] != len(lengths):
                raise ValueError("prefix mapping row count does not match the task")
            if any(length > prefix.shape[1] for length in lengths):
                raise ValueError("prefix mapping is shorter than prefix_seq_lens")
            return [prefix[index, :length].clone() for index, length in enumerate(lengths)]
        if prefix.ndim != 1 or prefix.numel() != sum(lengths):
            raise ValueError("flattened prefix mapping must contain sum(prefix_seq_lens) indexes")
        rows = []
        offset = 0
        for length in lengths:
            rows.append(prefix[offset : offset + length].clone())
            offset += length
        return rows

    def get_scratch_write_indexes(self, expected_numel: int) -> torch.Tensor:
        lease = self._scratch_lease
        if lease is None:
            raise RuntimeError("action scratch lease is not active")
        if lease.numel() != expected_numel:
            raise ValueError("action scratch lease has an unexpected size")
        return lease

    def end_task_mapping(self, identity: ActionTaskIdentity):
        if self._active_identity != identity:
            raise RuntimeError("action KV task identity does not own the active mapping")
        if self._active_rows is not None:
            self.req_to_token_indexs[self._active_rows] = -1
        self._active_identity = None
        self._active_rows = None
        self._scratch_lease = None

    def close(self):
        if self._active_rows is not None:
            self.req_to_token_indexs[self._active_rows] = -1
        self._active_identity = None
        self._active_rows = None
        self._scratch_lease = None
        self._shared_mem_manager = None
        self.kv_buffer = None
        self.req_to_token_indexs = None
        self.operator = None
