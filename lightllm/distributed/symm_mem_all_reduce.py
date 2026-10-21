import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    _SYMM_MEM_OK = True
except ImportError:
    torch_symm_mem = None
    _SYMM_MEM_OK = False

_MiB = 1024 * 1024

# Adopted from vLLM's benchmark-tuned SymmMem max-size table:
# vllm/distributed/device_communicators/all_reduce_utils.py
SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    "9.0": {2: 64 * _MiB, 4: 32 * _MiB, 6: 64 * _MiB, 8: 64 * _MiB},
    "10.0": {2: 8 * _MiB, 4: 32 * _MiB, 6: 128 * _MiB, 8: 128 * _MiB},
    "10.3": {2: 4 * _MiB, 4: 32 * _MiB, 6: 32 * _MiB, 8: 64 * _MiB},
}

# Adopted from vLLM's multimem-vs-two_shot world-size split.
# World sizes for which multimem (NVLS hardware reduce) beats two_shot.
_WORLD_SIZES_MULTIMEM = {"9.0": [4, 6, 8], "10.0": [6, 8], "10.3": [6, 8]}


class SymmMemAllreduce:
    """In-place all-reduce via torch symmetric memory (NVLink SHARP / NVLS)."""

    def __init__(self, group: ProcessGroup, device, dtype: torch.dtype = torch.bfloat16) -> None:
        self.disabled = True
        if not _SYMM_MEM_OK:
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.group = group
        self.dtype = dtype
        self.world_size = dist.get_world_size(group=group)

        cap = torch.cuda.get_device_capability(device)
        cap_str = f"{cap[0]}.{cap[1]}"
        if cap_str not in SYMM_MEM_ALL_REDUCE_MAX_SIZES:
            return
        if self.world_size not in SYMM_MEM_ALL_REDUCE_MAX_SIZES[cap_str]:
            return
        self.max_size = SYMM_MEM_ALL_REDUCE_MAX_SIZES[cap_str][self.world_size]
        self.use_multimem = self.world_size in _WORLD_SIZES_MULTIMEM.get(cap_str, [])

        try:
            self.buffer = torch_symm_mem.empty(self.max_size // dtype.itemsize, device=device, dtype=dtype)
            handle = torch_symm_mem.rendezvous(self.buffer, group.group_name)
        except RuntimeError as e:
            logger.warning("SymmMemAllreduce: rendezvous failed (%s). Disabling.", e)
            return
        # multimem and two_shot both require a multicast pointer.
        if getattr(handle, "multicast_ptr", 0) == 0:
            logger.warning("SymmMemAllreduce: multicast pointer unavailable; disabling.")
            return
        self.disabled = False
        logger.info(
            "SymmMemAllreduce enabled: world_size=%d, max_size=%d, multimem=%s",
            self.world_size,
            self.max_size,
            self.use_multimem,
        )

    def should_use(self, inp: torch.Tensor) -> bool:
        if self.disabled or inp.dtype != self.dtype or not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes % 4 != 0:
            return False
        # Lower bound is implicitly handled by the dispatch order in
        # CustomProcessGroup.all_reduce: FlashInfer claims small messages first.
        return nbytes < self.max_size

    def all_reduce(self, inp: torch.Tensor) -> None:
        n = inp.numel()
        self.buffer[:n].copy_(inp.view(-1))
        if self.use_multimem:
            torch.ops.symm_mem.multimem_all_reduce_(self.buffer[:n], "sum", self.group.group_name)
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(self.buffer[:n], "sum", self.group.group_name)
        inp.view(-1).copy_(self.buffer[:n])
