"""V4.1 conditional memory, following DeepSeek's MIT-licensed inference/engram.py."""

import math
import mmap
import weakref
from contextlib import nullcontext
from filelock import FileLock
import numpy as np
import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_dp_size
from lightllm.utils.shm_utils import create_or_link_shm

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
        mapping, size = build_compressed_token_map(tokenizer)
        assert size == config["engram_compressed_vocab_size"], (size, config["engram_compressed_vocab_size"])
        self.token_map = torch.tensor(mapping, dtype=torch.int64, device="cuda")
        self.has_vision = config.get("vision_n_layers", 0) > 0
        self.pad = mapping[config["engram_pad_token_id"]]
        self.history = torch.zeros(token_slots, dtype=torch.int64, device="cuda")
        self.order = config["engram_max_ngram_size"]
        primes, offsets, multipliers, used = [], [], [], set()
        bound = max(1, (np.iinfo(np.int64).max // size) // 2)
        for lid in config["engram_layer_ids"]:
            generator = np.random.default_rng(10007 * lid)
            multipliers.append(generator.integers(0, bound, size=self.order, dtype=np.int64) * 2 + 1)
            layer_primes = []
            for _ in range(self.order - 1):
                prime = config["engram_vocab_size"] - 1
                for _ in range(config["engram_n_heads"]):
                    prime += 1
                    while prime in used or prime < 2 or any(prime % d == 0 for d in range(2, math.isqrt(prime) + 1)):
                        prime += 1
                    used.add(prime)
                    layer_primes.append(prime)
            primes.append(layer_primes)
            offsets.append(np.cumsum([0, *layer_primes[:-1]]))
        self.primes = torch.tensor(primes, device="cuda")
        self.offsets = torch.tensor(np.asarray(offsets), device="cuda")
        self.multipliers = torch.tensor(np.asarray(multipliers), device="cuda")
        self.heads = config["engram_n_heads"]

    def __call__(self, state):
        positions = state.position_ids.long()
        reqs = state.dsv4_sparse_req_idx.long()
        slots = state.mem_index.long()
        ids = state.input_ids.long()
        if self.has_vision:
            # Image cache IDs are out of vocabulary and break the n-gram history.
            self.history[slots] = torch.where(
                ids < self.token_map.numel(), self.token_map[ids.clamp_max(self.token_map.numel() - 1)], -1
            )
        else:
            self.history[slots] = self.token_map[ids]
        tokens = []
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.order):
            previous = (positions - shift).clamp_min(0)
            previous_slots = state.req_manager.req_to_token_indexs[reqs, previous].long()
            source = self.history[previous_slots]
            blocked = blocked | (positions < shift) | (source == -1)
            tokens.append(torch.where(blocked, self.pad, source))
        products = torch.stack(tokens, -1)[:, None, :] * self.multipliers[None, :, :]
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.order):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling[..., None] % self.primes[:, (shift - 1) * self.heads : shift * self.heads])
        return torch.cat(hashes, -1) + self.offsets


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

    def __init__(self, prefix, rows, dim):
        super().__init__()
        self.prefix, self.dim = prefix, dim
        self.part_rows = (rows + self.tp_world_size_ - 1) // self.tp_world_size_
        self.start = self.tp_rank_ * self.part_rows
        self.end = min(rows, self.start + self.part_rows)
        self._create_weight()

    def _create_weight(self):
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        weight_bytes = self.part_rows * self.dim
        scale_bytes = self.part_rows * (self.dim // 32)
        logger.info(
            f"Allocating {self.prefix}: TP rank {self.tp_rank_}, "
            f"pinned table {(weight_bytes + scale_bytes) / 2**30:.2f} GiB"
        )
        table_bytes = weight_bytes + scale_bytes
        if get_dp_size() > 1:
            # DP ranks on this node read the same TP shard; pin one physical table per node.
            self._host_shm = create_or_link_shm(f"{self.prefix}_tp{self.tp_rank_}", table_bytes + 2)
            self._host_mapping = self._host_shm._mmap
            self.loaded = np.ndarray((2,), dtype=np.uint8, buffer=self._host_mapping, offset=table_bytes)
            self._load_lock = FileLock(f"/tmp/{self._host_shm.name}.load.lock")
        else:
            self._host_mapping = mmap.mmap(-1, table_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            self.loaded = [False, False]
            self._load_lock = nullcontext()
        self._host_mapping.madvise(mmap.MADV_HUGEPAGE)
        storage = torch.frombuffer(self._host_mapping, dtype=torch.uint8, count=table_bytes)
        # Prefault the exact-sized mapping before registration; the pinned allocator rounds
        # each 23.6 GiB table up to 32 GiB, increasing the host-registration footprint.
        # Read faults are safe when another DP rank has already started populating the table.
        np.frombuffer(self._host_mapping, dtype=np.uint8)[:: mmap.PAGESIZE].sum()
        runtime = torch.cuda.cudart()
        error = runtime.cudaHostRegister(storage.data_ptr(), storage.numel(), 3)
        if error != 0:
            raise RuntimeError(f"Registering {self.prefix} host table failed: {error}")
        self._host_registration = weakref.finalize(self, runtime.cudaHostUnregister, storage.data_ptr())
        self.weight = storage[:weight_bytes].view(torch.float8_e4m3fn).view(self.part_rows, self.dim)
        self.scale = storage[weight_bytes:].view(torch.float8_e8m0fnu).view(self.part_rows, self.dim // 32)
        self.device_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self.device_scale = get_accelerator_view_from_cpu_tensor(self.scale)

    def load_hf_weights(self, weights):
        for i, (suffix, target) in enumerate((("weight", self.weight), ("scale", self.scale))):
            name = f"{self.prefix}.{suffix}"
            if name in weights:
                with self._load_lock:
                    if not self.loaded[i]:
                        target[: self.end - self.start].copy_(weights[name][self.start : self.end])
                        self.loaded[i] = True

    def verify_load(self):
        return all(self.loaded)

    def lookup(self, indices):
        indices = indices.contiguous()
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
        if self.tp_world_size_ > 1:
            all_reduce(out)
        return out.flatten(1)


def engram_forward(streams, indices, weight, eps, token_mask=None):
    embedded = weight.engram_embed_.lookup(indices)
    kv = weight.engram_wkv_.mm(embedded)
    mult, dim = streams.shape[-2:]
    key, value = kv.split((mult * dim, dim), -1)
    key = key.float().view(-1, mult, dim)
    h = streams.float()
    dot = (h * key * weight.engram_q_.weight * weight.engram_k_.weight).sum(-1)
    dot *= torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps) * dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = torch.where(token_mask[:, None], gate, 0)
    return (h + gate[..., None] * value.float()[:, None, :]).to(streams.dtype)
