from dataclasses import dataclass
from typing import Optional

import torch
import triton

from lightllm.common.qsa.attention import indexed_attention
from lightllm.distributed import all_reduce, all_gather_into_tensor


@dataclass
class QSABatch:
    slot_table: torch.Tensor
    active_rows: torch.Tensor
    key_lengths: torch.Tensor
    first_blocks: torch.Tensor
    query_rows: torch.Tensor
    visible_lengths: torch.Tensor
    mem_indices: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    max_new_blocks: int
    starts: Optional[torch.Tensor] = None

    @classmethod
    def from_metadata(cls, metadata, slot_table, mem_indices, cos, sin, compress_ratio: int):
        if compress_ratio < 1:
            raise ValueError("QSA compression ratio must be positive")
        return cls(
            slot_table,
            metadata.active_rows,
            metadata.key_lengths,
            metadata.first_blocks,
            metadata.query_rows,
            metadata.visible_lengths,
            mem_indices.to(torch.int64),
            cos,
            sin,
            (mem_indices.numel() + compress_ratio - 1) // compress_ratio,
            metadata.starts,
        )


@dataclass
class QSAExecutionState:
    raw_keys: torch.Tensor

    pooled_keys: torch.Tensor
    key_cos: torch.Tensor
    key_sin: torch.Tensor
    compressor: Optional[torch.Tensor] = None

    def select(self, indexer, index_query, batch, decode, max_visible_length, tp_world_size, tp_rank, dist_group):
        tokens = index_query.shape[0]
        sharded = tp_world_size > 1 and tokens > 0 and tokens % (4 * tp_world_size) == 0
        rows = tokens // tp_world_size if sharded else tokens
        first = tp_rank * rows if sharded else 0
        selected = indexer.select(
            index_query[first : first + rows],
            self.pooled_keys,
            batch.query_rows[first : first + rows],
            batch.visible_lengths[first : first + rows],
            batch.slot_table,
            query_chunk_size=max(64, rows) if decode else 1024,
            max_visible_length=max_visible_length,
            physical_stride=indexer.compress_ratio if self.compressor is not None else 1,
        )
        if sharded:
            output = torch.empty((tokens, selected.shape[1]), dtype=selected.dtype, device=selected.device)
            all_gather_into_tensor(output, selected, group=dist_group)
            return output
        return selected

    def attend(
        self,
        indexer,
        hidden,
        query,
        key,
        value,
        batch: QSABatch,
        *,
        decode: bool = False,
        attention_splits: Optional[int] = None,
        max_visible_length: Optional[int] = None,
        tp_world_size: int = 1,
        tp_rank: int = 0,
        dist_group=None,
        index_projection=None,
    ):
        if index_projection is None:
            index_query, raw = indexer.project(hidden, batch.cos, batch.sin)
        else:
            index_query, raw = indexer.project(hidden, batch.cos, batch.sin, projected=index_projection)
        if index_query.dtype != self.pooled_keys.dtype:
            index_query = index_query.to(self.pooled_keys.dtype)
        if self.compressor is not None:
            from .qsa_compressor import pool_with_ring

            pool_with_ring(indexer, raw, self.compressor, batch, self.pooled_keys)
        else:
            self.raw_keys.index_copy_(0, batch.mem_indices, raw)
            indexer.pool(
                self.raw_keys,
                batch.slot_table,
                batch.key_lengths,
                self.key_cos,
                self.key_sin,
                self.pooled_keys,
                batch.first_blocks,
                batch.max_new_blocks,
                request_rows=batch.active_rows,
            )
        selected = self.select(
            indexer, index_query, batch, decode, max_visible_length, tp_world_size, tp_rank, dist_group
        )

        splits = attention_splits
        if splits is None and decode:
            splits = min(32, triton.cdiv(selected.shape[1], 64))
        wide_block = query.dtype in (torch.float16, torch.bfloat16)
        block_size, num_warps = (128 if wide_block else 64), None
        if key.dtype == torch.float8_e4m3fn and wide_block:
            if not decode and query.shape[0] >= 1024:
                block_size, num_warps = 32, 1
            elif decode and query.shape[0] <= 32:
                splits = min(16, triton.cdiv(selected.shape[1], block_size))
        return indexed_attention(
            query.contiguous(),
            key,
            value,
            selected,
            splits=splits,
            selection_lengths=batch.visible_lengths,
            short_rows_direct=decode or attention_splits is not None,
            skip_empty_splits=True,
            block_size=block_size,
            num_stages=2 if wide_block else None,
            num_warps=num_warps,
        )


@dataclass
class PLEBatch:
    starts: torch.Tensor
    ngram_history: torch.Tensor
    conv_history: torch.Tensor
    mask: Optional[torch.Tensor] = None
    conv_inputs: Optional[torch.Tensor] = None
    input_tokens: Optional[torch.Tensor] = None

    def apply(self, residual, tokens, layer_weight, tp_world_size: int, dist_group):
        hasher = layer_weight.ngram_hasher.component
        if self.mask is not None:
            tokens = tokens.masked_fill(~self.mask, hasher.eos_token_id)
        indices, _ = hasher.hash(tokens, self.starts, self.ngram_history)
        embeddings = layer_weight.ngram_embedding(indices.flatten()).view(
            tokens.numel(), hasher.heads * layer_weight.ngram_embedding.dim
        )
        if tp_world_size > 1:
            all_reduce(embeddings, group=dist_group, async_op=False)
        update, conv_inputs = layer_weight.ple.component.forward(
            residual, embeddings, self.starts, self.conv_history, self.mask
        )
        self.input_tokens, self.conv_inputs = tokens, conv_inputs
        return update

    def histories_after(self, layer_weight, accepted_lengths=None):
        if self.input_tokens is None or self.conv_inputs is None:
            raise RuntimeError("PLE candidates must be computed before requesting histories")
        ngram = layer_weight.ngram_hasher.component.history_after(
            self.input_tokens, self.starts, self.ngram_history, accepted_lengths
        )
        conv = layer_weight.ple.component.history_after(
            self.conv_inputs, self.starts, self.conv_history, accepted_lengths
        )
        return ngram, conv
