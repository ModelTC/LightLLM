"""V4.1 conditional memory, following DeepSeek's MIT-licensed inference/engram.py."""

import math
import mmap
import time
import os
import json
import hashlib
from pathlib import Path
from contextlib import nullcontext
from filelock import FileLock
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_dp_size
from lightllm.utils.shm_utils import create_or_link_shm, ServiceSharedMemory
from lightllm.common.weight_cache import is_preloading_weights, initializing_weight_client

logger = init_logger(__name__)


def build_compressed_token_map(tokenizer):
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
    backend = tokenizer.backend_tokenizer
    vocabulary, mapping = {}, []
    for token in range(len(tokenizer)):
        text = backend.decode([token], skip_special_tokens=False)
        key = backend.id_to_token(token) if "\ufffd" in text else normalizer.normalize_str(text) or text
        mapping.append(vocabulary.setdefault(key, len(vocabulary)))
    return mapping, len(vocabulary)


class EngramHasher:
    def __init__(self, config, tokenizer, token_slots):
        mapping, primes, offsets, multipliers = self._load_cpu_metadata(config, tokenizer)
        self.token_map = torch.tensor(mapping, dtype=torch.int64, device="cuda")
        self.has_vision = config.get("vision_n_layers", 0) > 0
        self.pad = int(mapping[config["engram_pad_token_id"]])
        self.history = torch.zeros(token_slots, dtype=torch.int64, device="cuda")
        self.order = config["engram_max_ngram_size"]
        self.primes = torch.tensor(primes, device="cuda")
        self.offsets = torch.tensor(offsets, device="cuda")
        self.multipliers = torch.tensor(multipliers, device="cuda")
        self.heads = config["engram_n_heads"]

    @staticmethod
    def _build_cpu_metadata(config, tokenizer):
        mapping, size = build_compressed_token_map(tokenizer)
        assert size == config["engram_compressed_vocab_size"], (size, config["engram_compressed_vocab_size"])
        order = config["engram_max_ngram_size"]
        primes, offsets, multipliers, used = [], [], [], set()
        bound = max(1, (np.iinfo(np.int64).max // size) // 2)
        for lid in config["engram_layer_ids"]:
            generator = np.random.default_rng(10007 * lid)
            multipliers.append(generator.integers(0, bound, size=order, dtype=np.int64) * 2 + 1)
            layer_primes = []
            for _ in range(order - 1):
                prime = config["engram_vocab_size"] - 1
                for _ in range(config["engram_n_heads"]):
                    prime += 1
                    while prime in used or prime < 2 or any(prime % d == 0 for d in range(2, math.isqrt(prime) + 1)):
                        prime += 1
                    used.add(prime)
                    layer_primes.append(prime)
            primes.append(layer_primes)
            offsets.append(np.cumsum([0, *layer_primes[:-1]]))
        return tuple(np.asarray(value, dtype=np.int64) for value in (mapping, primes, offsets, multipliers))

    @classmethod
    def _load_cpu_metadata(cls, config, tokenizer):
        cache_root = os.environ.get("LIGHTLLM_ENGRAM_CACHE_DIR")
        if not cache_root:
            return cls._build_cpu_metadata(config, tokenizer)
        import tokenizers

        identity = hashlib.sha256(Path(__file__).read_bytes())
        identity.update(tokenizer.backend_tokenizer.to_str().encode())
        identity.update(str((len(tokenizer), tokenizers.__version__, np.__version__)).encode())
        identity.update(
            json.dumps({k: v for k, v in config.items() if k.startswith("engram_")}, sort_keys=True).encode()
        )
        root = Path(cache_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{identity.hexdigest()}.npz"
        names = ("mapping", "primes", "offsets", "multipliers")
        with FileLock(str(path) + ".lock"):
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    result = tuple(data[name] for name in names)
                logger.info(f"Loaded Engram CPU metadata: {path}")
                return result
            started = time.monotonic()
            result = cls._build_cpu_metadata(config, tokenizer)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as file:
                np.savez(file, **dict(zip(names, result)))
            os.replace(temporary, path)
            logger.info(f"Cached Engram CPU metadata in {time.monotonic() - started:.3f}s: {path}")
            return result

    def _map_token_ids(self, ids):
        if self.has_vision:
            # Image cache IDs are out of vocabulary and break the n-gram history.
            return torch.where(
                ids < self.token_map.numel(), self.token_map[ids.clamp_max(self.token_map.numel() - 1)], -1
            )
        return self.token_map[ids]

    def restore_history(self, lookback, req_manager):
        rows = torch.tensor(
            [
                (req, start - len(ids) + offset, token)
                for req, start, ids in lookback
                for offset, token in enumerate(ids)
            ],
            dtype=torch.int64,
            device=self.history.device,
        )
        slots = req_manager.req_to_token_indexs[rows[:, 0], rows[:, 1]].long()
        self.history[slots] = self._map_token_ids(rows[:, 2])

    def __call__(self, state):
        count = state.input_ids.numel()
        layers, columns = self.primes.shape
        # All rows must reach history before other CTAs read their predecessors:
        # a prefill chunk or MTP verification can contain several rows per request.
        _store_engram_tokens[(triton.cdiv(count, 256),)](
            state.input_ids,
            state.mem_index,
            self.token_map,
            self.history,
            count,
            self.token_map.numel(),
            self.has_vision,
            256,
        )
        out = torch.empty((count, layers, columns), dtype=torch.int64, device=state.input_ids.device)
        _engram_hash[(triton.cdiv(count, 16), layers)](
            state.position_ids,
            state.dsv4_sparse_req_idx,
            state.req_manager.req_to_token_indexs,
            self.history,
            self.multipliers,
            self.primes,
            self.offsets,
            out,
            count,
            state.req_manager.req_to_token_indexs.stride(0),
            self.pad,
            self.order,
            self.heads,
            layers,
            16,
            triton.next_power_of_2(self.heads),
        )
        return out


@triton.jit
def _store_engram_tokens(
    IDS, SLOTS, TOKEN_MAP, HISTORY, N, VOCAB: tl.constexpr, VISION: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < N
    token = tl.load(IDS + row, valid, 0)
    slot = tl.load(SLOTS + row, valid, 0).to(tl.int64)
    mapped = tl.load(TOKEN_MAP + token, valid & (token < VOCAB) if VISION else valid, -1)
    tl.store(HISTORY + slot, mapped, valid)


@triton.jit
def _engram_hash(
    POSITIONS,
    REQS,
    REQ_TO_TOKEN,
    HISTORY,
    MULTIPLIERS,
    PRIMES,
    OFFSETS,
    OUT,
    N,
    REQ_STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    ORDER: tl.constexpr,
    HEADS: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    head = tl.arange(0, BLOCK_H)
    valid = row < N
    position = tl.load(POSITIONS + row, valid, 0)
    req = tl.load(REQS + row, valid, 0).to(tl.int64)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    columns: tl.constexpr = (ORDER - 1) * HEADS
    for shift in tl.static_range(ORDER):
        slot = tl.load(REQ_TO_TOKEN + req * REQ_STRIDE + tl.maximum(position - shift, 0), valid, 0).to(tl.int64)
        source = tl.load(HISTORY + slot, valid, 0)
        blocked = blocked | (position < shift) | (source == -1)
        token = tl.where(blocked, PAD, source)
        multiplier = tl.load(MULTIPLIERS + layer * ORDER + shift)
        rolling = rolling ^ (token * multiplier)
        if shift > 0:
            col = (shift - 1) * HEADS + head
            prime = tl.load(PRIMES + layer * columns + col, head < HEADS, 1)
            offset = tl.load(OFFSETS + layer * columns + col, head < HEADS, 0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            tl.store(
                OUT + row[:, None] * LAYERS * columns + layer * columns + col[None, :],
                hashed,
                valid[:, None] & (head[None, :] < HEADS),
            )


@triton.jit
def _gather_table(
    IDS, TABLE, SCALE, OUT, START: tl.constexpr, END: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    index = tl.load(IDS + row)
    valid = (index >= START) & (index < END)
    col = tl.arange(0, BLOCK)
    value = tl.load(TABLE + (index - START) * DIM + col, valid & (col < DIM), 0.0).to(tl.float32)
    exponent = tl.load(SCALE + (index - START) * (DIM // 32) + col // 32, valid & (col < DIM), 0).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    tl.store(OUT + row * DIM + col, value * scale, col < DIM)


class EngramEmbeddingWeight(BaseWeightTpl):
    """One row-sharded FP8 table in pinned host memory; the GPU reads it through UVA."""

    supports_resident_cache = False

    def __init__(self, prefix, rows, dim):
        super().__init__()
        self.prefix, self.dim = prefix, dim
        self.part_rows = (rows + self.tp_world_size_ - 1) // self.tp_world_size_
        self.start = self.tp_rank_ * self.part_rows
        self.end = min(rows, self.start + self.part_rows)
        self._create_weight()

    def _create_weight(self):
        # GPU weight skeletons are fake tensors; this shared CPU backing must be
        # real so each worker can register its own view of the owner's loaded table.
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            self._create_host_weight()

    def _create_host_weight(self):
        weight_bytes = self.part_rows * self.dim
        scale_bytes = self.part_rows * (self.dim // 32)
        logger.info(
            f"Allocating {self.prefix}: TP rank {self.tp_rank_}, "
            f"pinned table {(weight_bytes + scale_bytes) / 2**30:.2f} GiB"
        )
        table_bytes = weight_bytes + scale_bytes
        client = initializing_weight_client()
        if get_dp_size() > 1 or is_preloading_weights() or client is not None:
            # DP ranks on this node read the same TP shard; pin one physical table per node.
            if client is None:
                self._host_shm = create_or_link_shm(f"{self.prefix}_tp{self.tp_rank_}", table_bytes + 2)
            else:
                self._host_shm = ServiceSharedMemory(
                    name=f"{client.host_namespace}_{self.prefix}_tp{self.tp_rank_}", create=False
                )
                assert self._host_shm.size == table_bytes + 2, "Resident Engram table size differs"
            self._host_mapping = self._host_shm._mmap
            self.loaded = np.ndarray((2,), dtype=np.uint8, buffer=self._host_mapping, offset=table_bytes)
            self._load_lock = FileLock(f"/tmp/{self._host_shm.name}.load.lock")
        else:
            self._host_mapping = mmap.mmap(-1, table_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            self.loaded = [False, False]
            self._load_lock = nullcontext()
        self._host_mapping.madvise(mmap.MADV_HUGEPAGE)
        from lightllm.utils.kv_cache_utils import interleave_pages_across_numa_nodes
        import ctypes

        interleave_pages_across_numa_nodes(
            ctypes.CDLL(None, use_errno=True),
            ctypes.addressof(ctypes.c_uint8.from_buffer(self._host_mapping)),
            table_bytes,
            use_hugetlb=False,
        )
        storage = torch.frombuffer(self._host_mapping, dtype=torch.uint8, count=table_bytes)
        self.weight = storage[:weight_bytes].view(torch.float8_e4m3fn).view(self.part_rows, self.dim)
        self.scale = storage[weight_bytes:].view(torch.float8_e8m0fnu).view(self.part_rows, self.dim // 32)
        self._host_registration = None

    def finalize_host_registration(self):
        """Map this process's table only after the shared checkpoint data is complete."""
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        assert self.verify_load(), f"Engram table {self.prefix} is not fully loaded"
        if self._host_registration is not None:
            return
        started = time.monotonic()
        table_bytes = self.weight.numel() + self.scale.numel()
        from lightllm.utils.host_memory import HostMemoryRegistration

        registration = HostMemoryRegistration(self.weight.data_ptr(), table_bytes, table_bytes, workers=1)
        try:
            device_weight = get_accelerator_view_from_cpu_tensor(self.weight)
            device_scale = get_accelerator_view_from_cpu_tensor(self.scale)
        except BaseException:
            registration.close()
            raise
        self.device_weight, self.device_scale = device_weight, device_scale
        self._host_registration = registration
        logger.info(f"Engram {self.prefix}: registered {table_bytes} bytes in {time.monotonic() - started:.3f}s")

    def load_hf_weights(self, weights):
        for i, (suffix, target) in enumerate((("weight", self.weight), ("scale", self.scale))):
            name = f"{self.prefix}.{suffix}"
            if name in weights:
                with self._load_lock:
                    if not self.loaded[i]:
                        started = time.monotonic()
                        target[: self.end - self.start].copy_(weights[name][self.start : self.end])
                        self.loaded[i] = True
                        logger.info(f"Engram {name}: populated shared data in {time.monotonic() - started:.3f}s")

    def verify_load(self):
        return all(self.loaded)

    def gather_local(self, indices, out=None):
        """Read this rank's host shard; no collective is issued on the prefetch stream."""
        indices = indices.contiguous()
        if out is None:
            out = torch.empty((*indices.shape, self.dim), dtype=torch.bfloat16, device=indices.device)
        _gather_table[(indices.numel(),)](
            indices,
            self.device_weight,
            self.device_scale.view(torch.uint8),
            out,
            self.start,
            self.end,
            self.dim,
            triton.next_power_of_2(self.dim),
        )
        return out.flatten(1)

    def lookup(self, indices, local_embedding=None):
        out = self.gather_local(indices) if local_embedding is None else local_embedding
        if self.tp_world_size_ > 1:
            all_reduce(out)
        return out


@triton.jit
def _engram_gate(
    STREAMS,
    KV,
    Q,
    K,
    MASK,
    OUT,
    STREAM_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    MULT: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, stream = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < DIM
    h = tl.load(STREAMS + row * STREAM_STRIDE + stream * DIM + col, valid, 0).to(tl.float32)
    key = tl.load(KV + row * KV_STRIDE + stream * DIM + col, valid, 0).to(tl.float32)
    q = tl.load(Q + stream * DIM + col, valid, 0).to(tl.float32)
    k = tl.load(K + stream * DIM + col, valid, 0).to(tl.float32)
    dot = tl.sum(h * key * q * k, 0)
    dot *= tl.rsqrt(tl.sum(h * h, 0) / DIM + EPS) * tl.rsqrt(tl.sum(key * key, 0) / DIM + EPS) * DIM ** -0.5
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), 1e-6)), dot))
    if HAS_MASK:
        gate = tl.where(tl.load(MASK + row), gate, 0.0)
    value = tl.load(KV + row * KV_STRIDE + MULT * DIM + col, valid, 0).to(tl.float32)
    tl.store(OUT + row * OUT_STRIDE + stream * DIM + col, h + gate * value, valid)


def engram_forward(
    streams,
    indices,
    weight,
    eps,
    token_mask=None,
    out=None,
    local_embedding=None,
    prefetch_stream=None,
):
    # Two bounded buffers: 48 MiB total at 24 heads x 256 BF16 dimensions.
    chunk_rows = 2048
    if prefetch_stream is not None and streams.shape[0] > chunk_rows:
        if out is None:
            out = torch.empty_like(streams)
        current = torch.cuda.current_stream()
        buffers = [
            indices.new_empty(
                (chunk_rows, indices.shape[1], weight.engram_embed_.dim),
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        ]
        ready = [torch.cuda.Event() for _ in range(2)]
        consumed = [torch.cuda.Event() for _ in range(2)]
        prefetch_stream.wait_stream(current)
        with torch.cuda.stream(prefetch_stream):
            weight.engram_embed_.gather_local(indices[:chunk_rows], out=buffers[0])
            ready[0].record(prefetch_stream)
        for chunk, start in enumerate(range(0, streams.shape[0], chunk_rows)):
            slot = chunk % 2
            end = min(start + chunk_rows, streams.shape[0])
            if end < streams.shape[0]:
                next_slot = 1 - slot
                next_end = min(end + chunk_rows, streams.shape[0])
                with torch.cuda.stream(prefetch_stream):
                    if chunk:
                        prefetch_stream.wait_event(consumed[next_slot])
                    weight.engram_embed_.gather_local(indices[end:next_end], out=buffers[next_slot][: next_end - end])
                    ready[next_slot].record(prefetch_stream)
            current.wait_event(ready[slot])
            engram_forward(
                streams[start:end],
                indices[start:end],
                weight,
                eps,
                token_mask=None if token_mask is None else token_mask[start:end],
                out=out[start:end],
                local_embedding=buffers[slot][: end - start].flatten(1),
            )
            consumed[slot].record(current)
        for buffer in buffers:
            buffer.record_stream(prefetch_stream)
        return out
    embedded = (
        weight.engram_embed_.lookup(indices)
        if local_embedding is None
        else weight.engram_embed_.lookup(indices, local_embedding=local_embedding)
    )
    kv = weight.engram_wkv_.mm(embedded)
    mult, dim = streams.shape[-2:]
    if out is None:
        out = torch.empty(streams.shape, dtype=streams.dtype, device=streams.device)
    _engram_gate[(streams.shape[0], mult)](
        streams,
        kv,
        weight.engram_q_.weight,
        weight.engram_k_.weight,
        token_mask,
        out,
        streams.stride(0),
        kv.stride(0),
        out.stride(0),
        mult,
        dim,
        eps,
        token_mask is not None,
        triton.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
