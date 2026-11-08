from dataclasses import dataclass
from typing import Optional

import torch
import triton

from lightllm.common.qsa.attention import indexed_attention
from lightllm.distributed import all_reduce


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
        )


@dataclass
class QSAExecutionState:
    raw_keys: torch.Tensor

    pooled_keys: torch.Tensor
    key_cos: torch.Tensor
    key_sin: torch.Tensor

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
    ):
        index_query, raw = indexer.project(hidden, batch.cos, batch.sin)
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
        selected = indexer.select(
            index_query,
            self.pooled_keys,
            batch.query_rows,
            batch.visible_lengths,
            batch.slot_table,
            query_chunk_size=64 if decode else 1024,
        )

        splits = attention_splits
        if splits is None and decode:
            splits = min(32, triton.cdiv(selected.shape[1], 64))
        wide_block = query.dtype in (torch.float16, torch.bfloat16)
        return indexed_attention(
            query.contiguous(),
            key,
            value,
            selected,
            splits=splits,
            selection_lengths=batch.visible_lengths,
            short_rows_direct=decode or attention_splits is not None,
            skip_empty_splits=True,
            block_size=128 if wide_block else 64,
            num_stages=2 if wide_block else None,
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
