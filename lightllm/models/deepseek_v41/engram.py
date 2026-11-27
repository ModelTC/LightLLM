"""V4.1 conditional memory, following DeepSeek's MIT-licensed inference/engram.py."""

import math
import mmap
from contextlib import nullcontext
from filelock import FileLock
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.cuda_host_memory import register_cpu_tensor_storage
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_dp_size
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


def _build_compressed_token_id_map(tokenizer):
    from tokenizers import Regex, normalizers

    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), "\ue000"),
            normalizers.Strip(),
            normalizers.Replace("\ue000", " "),
        ]
    )
    backend_tokenizer = tokenizer.backend_tokenizer
    normalized_text_to_id = {}
    token_id_to_compressed_id = []
    for token_id in range(len(tokenizer)):
        decoded_text = backend_tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in decoded_text:
            # Incomplete byte sequences can decode to the same replacement character.
            normalized_text = backend_tokenizer.id_to_token(token_id)
        else:
            normalized_text = normalizer.normalize_str(decoded_text) or decoded_text
        compressed_id = normalized_text_to_id.setdefault(normalized_text, len(normalized_text_to_id))
        token_id_to_compressed_id.append(compressed_id)
    return token_id_to_compressed_id, len(normalized_text_to_id)


class EngramHasher:
    def __init__(self, config, tokenizer, num_kv_token_slots):
        token_id_to_compressed_id, compressed_vocab_size = _build_compressed_token_id_map(tokenizer)
        expected_compressed_vocab_size = config["engram_compressed_vocab_size"]
        assert compressed_vocab_size == expected_compressed_vocab_size, (
            compressed_vocab_size,
            expected_compressed_vocab_size,
        )
        self.token_id_to_compressed_id = torch.tensor(token_id_to_compressed_id, dtype=torch.int64, device="cuda")
        self.has_vision = config.get("vision_n_layers", 0) > 0
        self.compressed_pad_token_id = token_id_to_compressed_id[config["engram_pad_token_id"]]
        # One compressed token ID per physical KV slot on this rank, including the HOLD page.
        self.compressed_token_ids_by_slot = torch.zeros(num_kv_token_slots, dtype=torch.int64, device="cuda")
        self.max_ngram_size = config["engram_max_ngram_size"]
        self.num_hash_heads = config["engram_n_heads"]
        bucket_counts_by_layer = []
        row_offsets_by_layer = []
        hash_multipliers_by_layer = []
        used_bucket_counts = set()
        min_head_bucket_count = config["engram_vocab_size"]
        multiplier_sample_limit = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
        for layer_id in config["engram_layer_ids"]:
            # Reproduce the layer-specific odd multipliers used to address the checkpoint table.
            layer_rng = np.random.default_rng(10007 * layer_id)
            multiplier_samples = layer_rng.integers(
                0, multiplier_sample_limit, size=self.max_ngram_size, dtype=np.int64
            )
            hash_multipliers_by_layer.append(multiplier_samples * 2 + 1)
            layer_bucket_counts = []
            # Every hash head uses a distinct prime as its bucket count.
            for _ in range(self.max_ngram_size - 1):
                candidate_bucket_count = min_head_bucket_count - 1
                for _ in range(self.num_hash_heads):
                    candidate_bucket_count += 1
                    while (
                        candidate_bucket_count in used_bucket_counts
                        or candidate_bucket_count < 2
                        or any(
                            candidate_bucket_count % divisor == 0
                            for divisor in range(2, math.isqrt(candidate_bucket_count) + 1)
                        )
                    ):
                        candidate_bucket_count += 1
                    used_bucket_counts.add(candidate_bucket_count)
                    layer_bucket_counts.append(candidate_bucket_count)
            bucket_counts_by_layer.append(layer_bucket_counts)
            # Each hash head owns a consecutive range of rows in the layer's embedding table.
            row_offsets_by_layer.append(np.cumsum([0, *layer_bucket_counts[:-1]]))
        self.head_bucket_counts = torch.tensor(bucket_counts_by_layer, device="cuda")
        self.head_row_offsets = torch.tensor(np.asarray(row_offsets_by_layer), device="cuda")
        self.hash_multipliers = torch.tensor(np.asarray(hash_multipliers_by_layer), device="cuda")

    def restore_token_history(self, history_rows: list[tuple[int, int, int]], request_to_kv_slots: torch.Tensor):
        history_entries = torch.tensor(history_rows, dtype=torch.int64, device="cpu", pin_memory=True).to(
            self.compressed_token_ids_by_slot.device, non_blocking=True
        )
        num_entries = len(history_rows)
        _restore_cached_token_history_kernel[(triton.cdiv(num_entries, 128),)](
            history_entries,
            request_to_kv_slots,
            self.token_id_to_compressed_id,
            self.compressed_token_ids_by_slot,
            num_entries,
            request_to_kv_slots.stride(0),
            self.token_id_to_compressed_id.numel(),
            self.has_vision,
            128,
        )

    def compute_embedding_row_ids(self, infer_state):
        """Record current tokens and return [tokens, Engram layers, n-gram orders * heads] row IDs."""
        num_tokens = infer_state.input_ids.numel()
        num_layers, num_hash_columns = self.head_bucket_counts.shape
        # Finish the history writes before reading predecessors. A prefill chunk
        # or MTP verification can contain several tokens from one request.
        _store_compressed_token_history_kernel[(triton.cdiv(num_tokens, 256),)](
            infer_state.input_ids,
            infer_state.mem_index,
            self.token_id_to_compressed_id,
            self.compressed_token_ids_by_slot,
            num_tokens,
            self.token_id_to_compressed_id.numel(),
            self.has_vision,
            256,
        )
        embedding_row_ids = torch.empty(
            (num_tokens, num_layers, num_hash_columns), dtype=torch.int64, device=infer_state.input_ids.device
        )
        request_to_kv_slots = infer_state.req_manager.req_to_token_indexs
        _compute_engram_embedding_row_ids_kernel[(triton.cdiv(num_tokens, 16), num_layers)](
            infer_state.position_ids,
            infer_state.dsv4_sparse_req_idx,
            request_to_kv_slots,
            self.compressed_token_ids_by_slot,
            self.hash_multipliers,
            self.head_bucket_counts,
            self.head_row_offsets,
            embedding_row_ids,
            num_tokens,
            request_to_kv_slots.stride(0),
            self.compressed_pad_token_id,
            self.max_ngram_size,
            self.num_hash_heads,
            num_layers,
            16,
            triton.next_power_of_2(self.num_hash_heads),
        )
        return embedding_row_ids


@triton.jit
def _load_compressed_token_ids(
    token_id_to_compressed_id, token_ids, valid_rows, VOCAB_SIZE: tl.constexpr, HAS_VISION: tl.constexpr
):
    # Image cache IDs are outside the text vocabulary and interrupt the n-gram.
    valid_text = valid_rows & (token_ids < VOCAB_SIZE) if HAS_VISION else valid_rows
    return tl.load(token_id_to_compressed_id + token_ids, valid_text, -1)


@triton.jit
def _restore_cached_token_history_kernel(
    history_entries,
    request_to_kv_slots,
    token_id_to_compressed_id,
    compressed_token_ids_by_slot,
    num_entries,
    REQUEST_SLOT_ROW_STRIDE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    HAS_VISION: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0) * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
    valid_row = row < num_entries
    entry = history_entries + row * 3
    request_slot = tl.load(entry, valid_row, 0)
    position = tl.load(entry + 1, valid_row, 0)
    token_id = tl.load(entry + 2, valid_row, 0)
    kv_slot = tl.load(request_to_kv_slots + request_slot * REQUEST_SLOT_ROW_STRIDE + position, valid_row, 0)
    compressed_id = _load_compressed_token_ids(token_id_to_compressed_id, token_id, valid_row, VOCAB_SIZE, HAS_VISION)
    # Requests sharing a cached prefix may restore the same slot and token ID.
    tl.atomic_xchg(compressed_token_ids_by_slot + kv_slot, compressed_id, sem="relaxed", mask=valid_row)


@triton.jit
def _store_compressed_token_history_kernel(
    input_token_ids,
    kv_slots,
    token_id_to_compressed_id,
    compressed_token_ids_by_slot,
    num_tokens,
    VOCAB_SIZE: tl.constexpr,
    HAS_VISION: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    token_row = tl.program_id(0) * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
    valid_row = token_row < num_tokens
    token_id = tl.load(input_token_ids + token_row, valid_row, 0)
    kv_slot = tl.load(kv_slots + token_row, valid_row, 0).to(tl.int64)
    compressed_id = _load_compressed_token_ids(token_id_to_compressed_id, token_id, valid_row, VOCAB_SIZE, HAS_VISION)
    tl.store(compressed_token_ids_by_slot + kv_slot, compressed_id, valid_row)


@triton.jit
def _compute_engram_embedding_row_ids_kernel(
    position_ids,
    request_ids,
    request_to_kv_slots,
    compressed_token_ids_by_slot,
    hash_multipliers,
    head_bucket_counts,
    head_row_offsets,
    embedding_row_ids,
    num_tokens,
    REQUEST_SLOT_ROW_STRIDE: tl.constexpr,
    COMPRESSED_PAD_TOKEN_ID: tl.constexpr,
    MAX_NGRAM_SIZE: tl.constexpr,
    NUM_HASH_HEADS: tl.constexpr,
    NUM_HASH_LAYERS: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr,
):
    token_row = tl.program_id(0) * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
    layer_index = tl.program_id(1)
    head_index = tl.arange(0, HEAD_BLOCK_SIZE)
    valid_row = token_row < num_tokens
    valid_head = head_index < NUM_HASH_HEADS
    token_position = tl.load(position_ids + token_row, valid_row, 0)
    request_id = tl.load(request_ids + token_row, valid_row, 0).to(tl.int64)
    pad_remaining_history = tl.full((TOKEN_BLOCK_SIZE,), False, tl.int1)
    rolling_hash = tl.full((TOKEN_BLOCK_SIZE,), 0, tl.int64)
    NUM_HASH_COLUMNS: tl.constexpr = (MAX_NGRAM_SIZE - 1) * NUM_HASH_HEADS
    for lookback_distance in tl.static_range(MAX_NGRAM_SIZE):
        history_position = tl.maximum(token_position - lookback_distance, 0)
        history_kv_slot = tl.load(
            request_to_kv_slots + request_id * REQUEST_SLOT_ROW_STRIDE + history_position, valid_row, 0
        ).to(tl.int64)
        history_token_id = tl.load(compressed_token_ids_by_slot + history_kv_slot, valid_row, 0)
        pad_remaining_history |= (token_position < lookback_distance) | (history_token_id == -1)
        compressed_id = tl.where(pad_remaining_history, COMPRESSED_PAD_TOKEN_ID, history_token_id)
        multiplier = tl.load(hash_multipliers + layer_index * MAX_NGRAM_SIZE + lookback_distance)
        rolling_hash = rolling_hash ^ (compressed_id * multiplier)
        if lookback_distance > 0:
            hash_column = (lookback_distance - 1) * NUM_HASH_HEADS + head_index
            head_offset = layer_index * NUM_HASH_COLUMNS + hash_column
            bucket_count = tl.load(head_bucket_counts + head_offset, valid_head, 1)
            row_offset = tl.load(head_row_offsets + head_offset, valid_head, 0)
            row_ids = rolling_hash[:, None] % bucket_count[None, :] + row_offset[None, :]
            tl.store(
                embedding_row_ids + token_row[:, None] * NUM_HASH_LAYERS * NUM_HASH_COLUMNS + head_offset[None, :],
                row_ids,
                valid_row[:, None] & valid_head[None, :],
            )


@triton.jit
def _lookup_fp8_embedding_shard_kernel(
    embedding_row_ids,
    fp8_weights,
    e8m0_scale_exponents,
    output_embeddings,
    SHARD_ROW_START: tl.constexpr,
    SHARD_ROW_END: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    BLOCK_EMBEDDING: tl.constexpr,
):
    output_row = tl.program_id(0)
    embedding_row_id = tl.load(embedding_row_ids + output_row)
    row_in_shard = (embedding_row_id >= SHARD_ROW_START) & (embedding_row_id < SHARD_ROW_END)
    shard_row = embedding_row_id - SHARD_ROW_START
    embedding_column = tl.arange(0, BLOCK_EMBEDDING)
    valid_column = embedding_column < EMBEDDING_DIM
    fp8_value = tl.load(
        fp8_weights + shard_row * EMBEDDING_DIM + embedding_column, row_in_shard & valid_column, 0.0
    ).to(tl.float32)
    # Each E8M0 scale covers 32 consecutive embedding values.
    scale_exponent = tl.load(
        e8m0_scale_exponents + shard_row * (EMBEDDING_DIM // 32) + embedding_column // 32,
        row_in_shard & valid_column,
        0,
    ).to(tl.uint32)
    scale = (scale_exponent << 23).to(tl.float32, bitcast=True)
    tl.store(output_embeddings + output_row * EMBEDDING_DIM + embedding_column, fp8_value * scale, valid_column)


class EngramEmbeddingWeight(BaseWeightTpl):
    """One row-sharded FP8 table in pinned host memory; the GPU reads it through UVA."""

    def __init__(self, weight_prefix, num_embeddings, embedding_dim):
        super().__init__()
        self.weight_prefix = weight_prefix
        self.embedding_dim = embedding_dim
        self._shard_row_capacity = (num_embeddings + self.tp_world_size_ - 1) // self.tp_world_size_
        self._shard_row_start = self.tp_rank_ * self._shard_row_capacity
        self._shard_row_end = min(num_embeddings, self._shard_row_start + self._shard_row_capacity)
        self._create_weight()

    def _create_weight(self):
        weight_bytes = self._shard_row_capacity * self.embedding_dim
        scale_bytes = self._shard_row_capacity * (self.embedding_dim // 32)
        logger.info(
            f"Allocating {self.weight_prefix}: TP rank {self.tp_rank_}, "
            f"pinned table {(weight_bytes + scale_bytes) / 2**30:.2f} GiB"
        )
        table_bytes = weight_bytes + scale_bytes
        self._host_shm = None
        if get_dp_size() > 1:
            # DP ranks on this node read the same TP shard; pin one physical table per node.
            self._host_shm = create_or_link_shm(f"{self.weight_prefix}_tp{self.tp_rank_}", table_bytes + 2)
            self._host_mapping = self._host_shm._mmap
            self._loaded_components = np.ndarray((2,), dtype=np.uint8, buffer=self._host_mapping, offset=table_bytes)
            self._load_lock = FileLock(f"/tmp/{self._host_shm.name}.load.lock")
        else:
            self._host_mapping = mmap.mmap(-1, table_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            self._loaded_components = [False, False]
            self._load_lock = nullcontext()
        self._host_mapping.madvise(mmap.MADV_HUGEPAGE)
        host_storage = torch.frombuffer(self._host_mapping, dtype=torch.uint8, count=table_bytes)
        if self._host_shm is not None:
            # SharedMemory closes the mapping when its owner is released.
            host_storage.untyped_storage()._shared_memory_owner = self._host_shm
        self._host_fp8_weights = (
            host_storage[:weight_bytes].view(torch.float8_e4m3fn).view(self._shard_row_capacity, self.embedding_dim)
        )
        self._host_e8m0_scales = (
            host_storage[weight_bytes:]
            .view(torch.float8_e8m0fnu)
            .view(self._shard_row_capacity, self.embedding_dim // 32)
        )

    def load_hf_weights(self, weights):
        shard_components = (("weight", self._host_fp8_weights), ("scale", self._host_e8m0_scales))
        for component_index, (suffix, host_tensor) in enumerate(shard_components):
            checkpoint_name = f"{self.weight_prefix}.{suffix}"
            if checkpoint_name in weights:
                with self._load_lock:
                    if not self._loaded_components[component_index]:
                        host_tensor[: self._shard_row_end - self._shard_row_start].copy_(
                            weights[checkpoint_name][self._shard_row_start : self._shard_row_end]
                        )
                        self._loaded_components[component_index] = True

    def verify_load(self):
        if not all(self._loaded_components):
            return False
        # Weight and scale are views of one contiguous host storage.
        register_cpu_tensor_storage(self._host_fp8_weights)
        return True

    def lookup_rows(self, embedding_row_ids, tp_group):
        """Look up [tokens, heads across n-gram orders] row IDs and concatenate each token's embeddings."""
        embedding_row_ids = embedding_row_ids.contiguous()
        embeddings = torch.empty(
            (*embedding_row_ids.shape, self.embedding_dim), dtype=torch.bfloat16, device=embedding_row_ids.device
        )
        _lookup_fp8_embedding_shard_kernel[(embedding_row_ids.numel(),)](
            embedding_row_ids,
            self._host_fp8_weights,
            self._host_e8m0_scales.view(torch.uint8),
            embeddings,
            self._shard_row_start,
            self._shard_row_end,
            self.embedding_dim,
            triton.next_power_of_2(self.embedding_dim),
        )
        if self.tp_world_size_ > 1:
            all_reduce(embeddings, group=tp_group)
        return embeddings.flatten(1)


@triton.jit
def _add_gated_engram_value_kernel(
    residual_streams,
    projected_key_value,
    query_norm_weights,
    key_norm_weights,
    text_token_mask,
    output_streams,
    RESIDUAL_TOKEN_STRIDE: tl.constexpr,
    KEY_VALUE_TOKEN_STRIDE: tl.constexpr,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    NUM_STREAMS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    RMS_NORM_EPS: tl.constexpr,
    HAS_TEXT_TOKEN_MASK: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    token_row = tl.program_id(0).to(tl.int64)
    stream_id = tl.program_id(1)
    hidden_column = tl.arange(0, BLOCK_HIDDEN)
    valid_column = hidden_column < HIDDEN_SIZE
    stream_column = stream_id * HIDDEN_SIZE + hidden_column
    residual = tl.load(residual_streams + token_row * RESIDUAL_TOKEN_STRIDE + stream_column, valid_column, 0).to(
        tl.float32
    )
    memory_key = tl.load(projected_key_value + token_row * KEY_VALUE_TOKEN_STRIDE + stream_column, valid_column, 0).to(
        tl.float32
    )
    query_norm_weight = tl.load(query_norm_weights + stream_column, valid_column, 0).to(tl.float32)
    key_norm_weight = tl.load(key_norm_weights + stream_column, valid_column, 0).to(tl.float32)
    weighted_dot = tl.sum(residual * memory_key * query_norm_weight * key_norm_weight, 0)
    query_inv_rms = tl.rsqrt(tl.sum(residual * residual, 0) / HIDDEN_SIZE + RMS_NORM_EPS)
    key_inv_rms = tl.rsqrt(tl.sum(memory_key * memory_key, 0) / HIDDEN_SIZE + RMS_NORM_EPS)
    weighted_dot *= query_inv_rms * key_inv_rms * HIDDEN_SIZE ** -0.5
    gate_weight = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(weighted_dot), 1e-6)), weighted_dot))
    if HAS_TEXT_TOKEN_MASK:
        gate_weight = tl.where(tl.load(text_token_mask + token_row), gate_weight, 0.0)
    # The value is shared by all streams and follows their keys in the projection.
    shared_memory_value = tl.load(
        projected_key_value + token_row * KEY_VALUE_TOKEN_STRIDE + NUM_STREAMS * HIDDEN_SIZE + hidden_column,
        valid_column,
        0,
    ).to(tl.float32)
    tl.store(
        output_streams + token_row * OUTPUT_TOKEN_STRIDE + stream_column,
        residual + gate_weight * shared_memory_value,
        valid_column,
    )


def apply_engram_to_streams(
    residual_streams, embedding_row_ids, layer_weight, rms_norm_eps, tp_group, text_token_mask=None
):
    """Look up n-gram embeddings and add a gated memory value to each residual stream."""
    embeddings = layer_weight.engram_embed_.lookup_rows(embedding_row_ids, tp_group)
    projected_key_value = layer_weight.engram_wkv_.mm(embeddings)
    num_streams, hidden_size = residual_streams.shape[-2:]
    output_streams = torch.empty(residual_streams.shape, dtype=residual_streams.dtype, device=residual_streams.device)
    _add_gated_engram_value_kernel[(residual_streams.shape[0], num_streams)](
        residual_streams,
        projected_key_value,
        layer_weight.engram_q_.weight,
        layer_weight.engram_k_.weight,
        text_token_mask,
        output_streams,
        residual_streams.stride(0),
        projected_key_value.stride(0),
        output_streams.stride(0),
        num_streams,
        hidden_size,
        rms_norm_eps,
        text_token_mask is not None,
        triton.next_power_of_2(hidden_size),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output_streams
