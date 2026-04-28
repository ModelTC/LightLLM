import os
import random
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    import flashinfer.comm as flashinfer_comm
    from flashinfer.comm.mnnvl import TorchDistBackend

    _FI_OK = hasattr(flashinfer_comm, "allreduce_fusion") and hasattr(
        flashinfer_comm, "create_allreduce_fusion_workspace"
    )
except ImportError:
    flashinfer_comm = None
    TorchDistBackend = None
    _FI_OK = False

_MiB = 1024 * 1024
# Default upper bound for the FlashInfer fast path (oneshot lamport regime).
# Used when (compute_cap, world_size) is not in the table below. Above the
# resolved bound, dispatch falls through to SymmMem multimem / NCCL.
FI_ALLREDUCE_DEFAULT_MAX_BYTES = 256 * 1024

_FI_ALLREDUCE_MAX_BYTES = {
    "9.0": {2: 512 * 1024, 4: 256 * 1024, 6: 256 * 1024, 8: 128 * 1024},
    "10.0": {2: 1024 * 1024, 4: 512 * 1024, 6: 256 * 1024, 8: 256 * 1024},
    "10.3": {2: 1024 * 1024, 4: 512 * 1024, 6: 512 * 1024, 8: 256 * 1024},
}

_FI_WORKSPACE_MAX_SIZE_MB = {
    "9.0": {2: 2.0, 4: 1.0, 6: 1.0, 8: 0.5},
    "10.0": {2: 2.0, 4: 2.0, 6: 1.0, 8: 1.0},
    "10.3": {2: 2.0, 4: 2.0, 6: 2.0, 8: 1.0},
}


class FlashInferAllReduce:
    """Small-message all-reduce via flashinfer trtllm oneshot lamport.

    Out-of-place: callers assign back via ``t.data = fi.all_reduce(t)``.
    """

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]) -> None:
        self.disabled = True
        self._workspace = None
        self._ws_hidden_dim = None
        self._ws_dtype = None
        self._ws_max_token_num = 0

        if not _FI_OK or not torch.cuda.is_available():
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(group=group)
        self.rank = dist.get_rank(group=group)
        if self.world_size == 1:
            return

        cap = torch.cuda.get_device_capability(device)
        cap_str = f"{cap[0]}.{cap[1]}"
        ws_table = _FI_WORKSPACE_MAX_SIZE_MB.get(cap_str)
        if ws_table is None or self.world_size not in ws_table:
            return
        self.max_workspace_size = int(ws_table[self.world_size] * _MiB)
        self.max_bytes = _FI_ALLREDUCE_MAX_BYTES.get(cap_str, {}).get(self.world_size, FI_ALLREDUCE_DEFAULT_MAX_BYTES)
        assert self.max_bytes <= self.max_workspace_size, (
            "FlashInferAllReduce config mismatch: "
            f"max_bytes={self.max_bytes} exceeds max_workspace_size={self.max_workspace_size}"
        )
        self.disabled = False

    def _ensure_workspace(self, hidden_dim: int, dtype: torch.dtype) -> bool:
        if self._workspace is not None and self._ws_hidden_dim == hidden_dim and self._ws_dtype == dtype:
            return True
        element_size = torch.tensor([], dtype=dtype).element_size()
        max_token_num = max(1, self.max_workspace_size // (hidden_dim * element_size))
        if self._workspace is not None:
            try:
                self._workspace.destroy()
            except Exception:
                pass
            self._workspace = None
        rng_state = random.getstate()
        try:
            random.seed(int.from_bytes(os.urandom(16), byteorder="big"))
            self._workspace = flashinfer_comm.create_allreduce_fusion_workspace(
                backend="trtllm",
                world_size=self.world_size,
                rank=self.rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                comm_backend=TorchDistBackend(group=self.group),
            )
        except Exception as e:
            logger.warning("FlashInferAllReduce workspace init failed: %s. Disabling.", e)
            self.disabled = True
            self._workspace = None
            return False
        finally:
            random.setstate(rng_state)
        self._ws_hidden_dim = hidden_dim
        self._ws_dtype = dtype
        self._ws_max_token_num = max_token_num
        return True

    def should_use(self, inp: torch.Tensor) -> bool:
        if self.disabled or not inp.is_cuda or not inp.is_contiguous():
            return False
        if inp.dtype not in (torch.bfloat16, torch.float16) or inp.dim() != 2:
            return False
        if inp.numel() * inp.element_size() >= self.max_bytes:
            return False
        _, hidden_dim = inp.shape
        if not self._ensure_workspace(hidden_dim, inp.dtype):
            return False
        return True

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return flashinfer_comm.allreduce_fusion(
            input=inp,
            workspace=self._workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
        )
