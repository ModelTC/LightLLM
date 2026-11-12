import torch
import triton

from .row_sharded_embedding_weight import RowShardedEmbeddingWeight
from lightllm.common.basemodel.triton_kernel.embedding import embedding_kernel
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PinnedRowShardedEmbeddingWeight(RowShardedEmbeddingWeight):
    def __init__(
        self, dim, vocab_size, weight_name, data_type, shard_count, *, cpu_allocator, tp_rank=None, tp_world_size=None
    ):
        if not callable(cpu_allocator):
            raise ValueError("Pinned embeddings require a CPU allocation policy")
        self.cpu_allocator = cpu_allocator
        super().__init__(dim, vocab_size, weight_name, data_type, shard_count, tp_rank, tp_world_size)

    def _create_weight(self):
        shape = (self.tp_vocab_end_id - self.tp_vocab_start_id, self.dim)
        with torch.cuda.device(self.device_id_):
            self.weight = self.cpu_allocator(shape, dtype=self.data_type_, device="cpu", pin_memory=True)
        if (
            self.weight.device.type != "cpu"
            or self.weight.dtype != self.data_type_
            or tuple(self.weight.shape) != shape
            or not self.weight.is_contiguous()
            or (self.weight.numel() and not self.weight.is_pinned())
        ):
            raise ValueError(
                "Embedding allocator must return contiguous pinned CPU storage with the requested shape/dtype"
            )
        self.weight.load_ok = False
        logger.info(
            f"Pinned embedding table: rank={self.tp_rank_} rows={shape[0]} dim={self.dim} "
            f"bytes={self.weight.numel() * self.weight.element_size()} dtype={self.weight.dtype} residency=cpu"
        )

    def load_hf_weights(self, weights):
        names = {self.weight_name} | {f"{self._shard_prefix}.shard_{index}.weight" for index in self._required_shards}
        host_weights = {name: value.cpu() if name in names else value for name, value in weights.items()}
        super().load_hf_weights(host_weights)

    def _cuda_forward(self, input_ids, out=None, alloc_func=torch.empty):
        if (
            not input_ids.is_cuda
            or input_ids.ndim != 1
            or not input_ids.is_contiguous()
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Pinned embedding lookup requires a CUDA vector of integer IDs")
        shape = (input_ids.shape[0], self.dim)
        if out is None:
            out = alloc_func(shape, dtype=self.weight.dtype, device=input_ids.device)
        if (
            tuple(out.shape) != shape
            or out.dtype != self.weight.dtype
            or out.device != input_ids.device
            or out.stride(1) != 1
        ):
            raise ValueError("Pinned embedding output must match CUDA IDs, table dtype and lookup shape")
        if not input_ids.numel():
            return out
        embedding_kernel[(input_ids.numel(),)](
            self.weight,
            input_ids,
            out,
            self.tp_vocab_start_id,
            self.tp_vocab_end_id,
            self.weight.stride(0),
            out.stride(0),
            input_ids.numel(),
            self.dim,
            triton.next_power_of_2(self.dim),
            1,
            1,
            num_warps=1,
            num_stages=1,
        )
        return out

    def _native_forward(self, *args, **kwargs):
        raise ValueError("Pinned host embedding lookup requires the CUDA Triton implementation")
