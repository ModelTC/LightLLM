from __future__ import annotations

from collections.abc import Sequence

import torch
from safetensors import safe_open


class Pi0SafeTensorLoader:
    """Small component loader that never materializes the full 14 GB checkpoint."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self._file = None

    def __enter__(self) -> "Pi0SafeTensorLoader":
        self._file = safe_open(self.checkpoint_path, framework="pt", device="cpu")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file = None

    def keys(self) -> Sequence[str]:
        return self._file.keys()

    def has(self, name: str) -> bool:
        return name in self._file.keys()

    def tensor(
        self,
        name: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        row_slice: slice | None = None,
        column_slice: slice | None = None,
    ) -> torch.Tensor:
        if not self.has(name):
            raise KeyError(f"missing pi0 checkpoint tensor: {name}")
        value = self._file.get_slice(name)
        if row_slice is not None and column_slice is not None:
            value = value[row_slice, column_slice]
        elif row_slice is not None:
            value = value[row_slice]
        elif column_slice is not None:
            value = value[:, column_slice]
        else:
            value = value[:]
        return value.to(device=device, dtype=dtype or value.dtype).contiguous()
