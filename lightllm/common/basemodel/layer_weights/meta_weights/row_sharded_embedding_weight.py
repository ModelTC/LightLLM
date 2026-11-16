from typing import Dict, Optional

import torch

from .embedding_weight import EmbeddingWeight


class RowShardedEmbeddingWeight(EmbeddingWeight):
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        weight_name: str,
        data_type: torch.dtype,
        shard_count: int,
        tp_rank: Optional[int] = None,
        tp_world_size: Optional[int] = None,
    ):
        if shard_count < 1 or vocab_size < 1 or vocab_size % shard_count:
            raise ValueError("Row-sharded embeddings require a positive vocabulary divisible by shard count")
        if not weight_name.endswith(".weight"):
            raise ValueError("Embedding weight name must end with .weight")
        self.shard_count = shard_count
        self.shard_rows = vocab_size // shard_count
        self._shard_prefix = weight_name[: -len(".weight")]
        self._loaded_shards = set()
        self._full_loaded = False
        super().__init__(dim, vocab_size, weight_name, data_type, tp_rank, tp_world_size)
        self._required_shards = {
            index
            for index in range(shard_count)
            if max(index * self.shard_rows, self.tp_vocab_start_id)
            < min((index + 1) * self.shard_rows, self.tp_vocab_end_id)
        }
        self.weight.load_ok = not self._required_shards

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        shard_keys = [(index, f"{self._shard_prefix}.shard_{index}.weight") for index in range(self.shard_count)]
        present = [(index, key) for index, key in shard_keys if key in weights]
        if self.weight_name in weights:
            if present or self._loaded_shards:
                raise ValueError("Cannot mix full and row-sharded embedding checkpoint tensors")
            if weights[self.weight_name].shape != (self.vocab_size, self.dim):
                raise ValueError("Full embedding checkpoint dimensions do not match")
            super().load_hf_weights(weights)
            self._full_loaded = True
            return
        if self._full_loaded:
            if present:
                raise ValueError("Cannot mix full and row-sharded embedding checkpoint tensors")
            return
        for index, key in present:
            source = weights[key]
            if source.shape != (self.shard_rows, self.dim):
                raise ValueError(f"Unexpected dimensions for embedding shard {index}: {tuple(source.shape)}")
            shard_start = index * self.shard_rows
            begin = max(shard_start, self.tp_vocab_start_id)
            end = min(shard_start + self.shard_rows, self.tp_vocab_end_id)
            if begin < end:
                self.weight[begin - self.tp_vocab_start_id : end - self.tp_vocab_start_id].copy_(
                    source[begin - shard_start : end - shard_start].to(self.data_type_)
                )
                self._loaded_shards.add(index)
        self.weight.load_ok = self._loaded_shards >= self._required_shards
