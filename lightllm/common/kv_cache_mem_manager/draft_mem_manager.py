"""Draft-model views over target-owned KV cache storage."""

from __future__ import annotations

import torch


class DraftMemoryManagerOperator:
    """Write draft KV in the target manager's canonical cache geometry."""

    def __init__(self, mem_manager: "DraftMemoryManagerView") -> None:
        self.mem_manager = mem_manager

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        view = self.mem_manager
        if kv.ndim != 3 or kv.shape[1:] != (2 * view.head_num, view.head_dim):
            raise ValueError(
                "draft KV shape must be " f"[tokens, {2 * view.head_num}, {view.head_dim}], got {list(kv.shape)}"
            )
        canonical_kv = kv.view(kv.shape[0], 2, view.head_num, view.head_dim).view(
            kv.shape[0], 2 * view.owner.head_num, view.owner.head_dim
        )
        return view.owner.operator.copy_kv_to_mem_manager(layer_index, mem_index, canonical_kv)

    def __getattr__(self, name):
        return getattr(self.mem_manager.owner.operator, name)


class DraftMemoryManagerView:
    """Expose a compatible draft KV geometry without allocating another cache.

    A draft may split each target KV head into an integer number of narrower
    heads.  The underlying bytes remain owned by ``owner`` so calibration,
    quantization, allocation and layer mapping remain canonical to the target.
    """

    def __init__(self, owner, *, head_num: int, head_dim: int) -> None:
        if head_num <= 0 or head_dim <= 0:
            raise ValueError(f"draft KV geometry must be positive, got heads={head_num}, head_dim={head_dim}")
        owner_width = owner.head_num * owner.head_dim
        draft_width = head_num * head_dim
        if draft_width != owner_width:
            raise ValueError(
                "draft KV width must equal target-owned cache width: " f"draft={draft_width}, owner={owner_width}"
            )
        if head_num % owner.head_num:
            raise ValueError(
                "draft KV head count must be an integer multiple of target-owned heads: "
                f"draft={head_num}, owner={owner.head_num}"
            )
        self.owner = owner
        self.head_num = head_num
        self.head_dim = head_dim
        self._head_repeat = head_num // owner.head_num
        self._scales = self._expand_scales("scales")
        self._q_scales = self._expand_scales("q_scales")
        self.operator = DraftMemoryManagerOperator(self)

    def _expand_scales(self, name: str):
        scales = getattr(self.owner, name, None)
        if scales is None:
            return None
        return torch.repeat_interleave(scales, self._head_repeat, dim=-1)

    @property
    def kv_buffer(self):
        return self.owner.kv_buffer.view(
            self.owner.kv_buffer.shape[0], self.owner.kv_buffer.shape[1], 2 * self.head_num, self.head_dim
        )

    @property
    def scales(self):
        return self._scales

    @property
    def q_scales(self):
        return self._q_scales

    def get_att_input_params(self, layer_index: int):
        k, v = self.owner.get_att_input_params(layer_index)
        return (
            k.view(k.shape[0], self.head_num, self.head_dim),
            v.view(v.shape[0], self.head_num, self.head_dim),
        )

    def __getattr__(self, name):
        return getattr(self.owner, name)
