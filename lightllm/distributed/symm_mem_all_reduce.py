import torch
import torch.distributed as dist
import triton
import triton.language as tl
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


@triton.jit
def _ordered_peer_sum(Inputs, Output, SIZE: tl.constexpr, START: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.full((BLOCK,), 0, tl.float64)
    for rank in tl.static_range(len(Inputs)):
        value = tl.load(Inputs[rank] + START + index, index < SIZE, other=0)
        accumulator += value.to(tl.float64)
    tl.store(Output + index, accumulator, index < SIZE)


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
        self.handle = handle
        self.peer_buffers = None
        self.use_peer_reduce = cap_str == "9.0" and self.world_size == 8 and dtype == torch.bfloat16
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

    def ordered_all_gather(self, inp: torch.Tensor):
        if not self.should_use(inp) or inp.device != self.device:
            return None
        count = inp.numel() * self.world_size
        if not count or count * inp.element_size() > self.max_size:
            return None
        if not hasattr(torch.ops.symm_mem, "multimem_all_gather_out"):
            return None
        if not hasattr(self, "ordered_gather_buffer"):
            if torch.cuda.is_current_stream_capturing():
                return None
            self.ordered_gather_buffer = torch_symm_mem.empty(
                self.max_size // self.dtype.itemsize, device=self.device, dtype=self.dtype
            )
            torch_symm_mem.rendezvous(self.ordered_gather_buffer, self.group.group_name)
        gathered = self.ordered_gather_buffer[:count]
        torch.ops.symm_mem.multimem_all_gather_out(inp.view(-1), self.group.group_name, gathered)
        return gathered.view(self.world_size, *inp.shape)

    def ordered_reduce_scatter(self, inp: torch.Tensor, output: torch.Tensor, rank: int) -> bool:
        if (
            not self.should_use(inp)
            or not self.use_peer_reduce
            or inp.device != self.device
            or inp.numel() < 128 * 2560
        ):
            return False
        if rank != self.handle.rank:
            raise ValueError("Sequence-parallel scatter rank must match the process group rank")
        if output.numel() * self.world_size != inp.numel() or output.dtype != inp.dtype:
            raise ValueError("Ordered reduce-scatter output must hold one rank's input shard")
        if output.device != inp.device or not output.is_contiguous():
            raise ValueError("Ordered reduce-scatter output must be contiguous and on the input device")
        if self.peer_buffers is None:
            self.peer_buffers = tuple(
                self.handle.get_buffer(rank, self.buffer.shape, self.dtype) for rank in range(self.world_size)
            )
        self.buffer[: inp.numel()].copy_(inp.view(-1))
        self.handle.barrier(channel=0)
        _ordered_peer_sum[(triton.cdiv(output.numel(), 256),)](
            self.peer_buffers, output, output.numel(), rank * output.numel(), 256
        )
        self.handle.barrier(channel=1)
        return True
