"""Hybrid distributed safetensors loading for expert-parallel models.

Expert tensors remain lazy CPU mmap views, so every EP rank faults and copies
only its local experts.  Replicated tensors are packed once by a rotating rank
and broadcast as one CUDA buffer.  This avoids both bad extremes: every rank
reading dense tensors from remote storage, and broadcasting the much larger
expert section that is already naturally sharded by the model loader.
"""

import time
from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist
from safetensors import safe_open

from lightllm.utils.dist_utils import create_new_group_for_current_node
from lightllm.utils.envs_utils import get_env_start_args


@dataclass
class DistributedWeightShard:
    expert_weights: Dict[str, torch.Tensor]
    replicated_weights: Dict[str, torch.Tensor]
    broadcast_work: list

    def wait_for_replicated_weights(self) -> None:
        for work in self.broadcast_work:
            work.wait()


class DistributedSafetensorLoader:
    """Broadcast packed non-expert tensors over a node or the global group."""

    _BROADCAST_CHUNK_BYTES = 1 << 30
    _BUFFER_ALIGNMENT = 16

    def __init__(self, mode: str):
        if mode not in ("node", "global"):
            raise ValueError(f"distributed safetensor mode must be 'node' or 'global', got {mode!r}")
        if not dist.is_initialized():
            raise RuntimeError("distributed safetensor loading requires an initialized torch process group")
        if dist.get_backend() != "nccl":
            raise RuntimeError("distributed safetensor loading requires an NCCL process group")

        self.mode = mode
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.device = torch.device("cuda", torch.cuda.current_device())

        if mode == "global":
            self.group = None
            self.group_rank_start = 0
            self.group_world_size = self.global_world_size
        else:
            args = get_env_start_args()
            if args.nnodes <= 0 or self.global_world_size % args.nnodes != 0:
                raise ValueError(
                    "global world size must be divisible by nnodes for node-local weight loading: "
                    f"world_size={self.global_world_size}, nnodes={args.nnodes}"
                )
            self.group_world_size = self.global_world_size // args.nnodes
            self.group_rank_start = args.node_rank * self.group_world_size
            self.group = create_new_group_for_current_node("nccl")

        self.total_bytes = 0
        self.started_at = time.perf_counter()

    @property
    def is_progress_rank(self) -> bool:
        return self.global_rank == 0

    def load_file(self, path: str, shard_index: int) -> DistributedWeightShard:
        checkpoint = safe_open(path, framework="pt", device="cpu")
        weights = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
        del checkpoint

        expert_weights = {name: tensor for name, tensor in weights.items() if ".experts." in name}
        dense_layout = []
        buffer_bytes = 0
        for name, tensor in weights.items():
            if ".experts." in name:
                continue
            buffer_bytes = self._align(buffer_bytes)
            tensor_bytes = tensor.numel() * tensor.element_size()
            dense_layout.append((name, tensor.dtype, tensor.shape, buffer_bytes, tensor_bytes))
            buffer_bytes += tensor_bytes

        if buffer_bytes == 0:
            return DistributedWeightShard(
                expert_weights=expert_weights,
                replicated_weights={},
                broadcast_work=[],
            )

        source_rank = self.group_rank_start + shard_index % self.group_world_size
        if self.global_rank == source_rank:
            dense_buffer = torch.empty(buffer_bytes, dtype=torch.uint8, device=self.device)
            for name, dtype, shape, offset, tensor_bytes in dense_layout:
                self._tensor_view(dense_buffer, dtype, shape, offset, tensor_bytes).copy_(weights[name])
        else:
            dense_buffer = torch.empty(buffer_bytes, dtype=torch.uint8, device=self.device)

        broadcast_work = self._broadcast_buffer(dense_buffer, source_rank)
        replicated_weights = {}
        for name, dtype, shape, offset, tensor_bytes in dense_layout:
            replicated_weights[name] = self._tensor_view(dense_buffer, dtype, shape, offset, tensor_bytes)
        self.total_bytes += buffer_bytes
        return DistributedWeightShard(
            expert_weights=expert_weights,
            replicated_weights=replicated_weights,
            broadcast_work=broadcast_work,
        )

    def summary(self) -> str:
        elapsed = time.perf_counter() - self.started_at
        gib = self.total_bytes / (1 << 30)
        rate = gib / elapsed if elapsed > 0 else 0.0
        return (
            f"mode={self.mode}, replicated_payload={gib:.2f} GiB, "
            f"elapsed={elapsed:.2f}s, payload_per_total_second={rate:.2f} GiB/s"
        )

    def _broadcast_buffer(self, shard_buffer: torch.Tensor, source_rank: int) -> list:
        broadcast_work = []
        for offset in range(0, shard_buffer.numel(), self._BROADCAST_CHUNK_BYTES):
            chunk_size = min(self._BROADCAST_CHUNK_BYTES, shard_buffer.numel() - offset)
            broadcast_work.append(
                dist.broadcast(
                    shard_buffer.narrow(0, offset, chunk_size),
                    src=source_rank,
                    group=self.group,
                    async_op=True,
                )
            )
        return broadcast_work

    @classmethod
    def _align(cls, offset: int) -> int:
        return (offset + cls._BUFFER_ALIGNMENT - 1) // cls._BUFFER_ALIGNMENT * cls._BUFFER_ALIGNMENT

    @staticmethod
    def _tensor_view(
        buffer: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.Size,
        offset: int,
        tensor_bytes: int,
    ) -> torch.Tensor:
        return buffer.narrow(0, offset, tensor_bytes).view(dtype).view(shape)
