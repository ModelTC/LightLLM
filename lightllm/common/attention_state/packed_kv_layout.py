"""Byte geometry for a shared KV vector with a separately stored BF16 RoPE tail."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedKVLayout:
    kv_dim: int
    rope_dim: int
    format: str = "fp8_group64_e8m0"

    def __post_init__(self):
        assert self.format in ("fp8_group64_e8m0", "nvfp4_block16")
        assert self.kv_dim > self.rope_dim >= 0
        assert (self.kv_dim - self.rope_dim) % self.quant_group_size == 0

    @property
    def quant_group_size(self):
        return 16 if self.format == "nvfp4_block16" else 64

    @property
    def data_bytes(self):
        nope = self.kv_dim - self.rope_dim
        return (nope // 2 if self.format == "nvfp4_block16" else nope) + self.rope_dim * 2

    @property
    def scale_bytes(self):
        groups = (self.kv_dim - self.rope_dim) // self.quant_group_size
        return (groups + 7) // 8 * 8

    @property
    def entry_bytes(self):
        return self.data_bytes + self.scale_bytes

    @property
    def alignment_bytes(self):
        # Native FP8 FlashMLA requires a whole data-row stride. The NVFP4
        # consumer accepts byte strides, so retain only vector alignment.
        return 16 if self.format == "nvfp4_block16" else self.data_bytes

    def packed_bytes(self, tokens):
        alignment = self.alignment_bytes
        return (tokens * self.entry_bytes + alignment - 1) // alignment * alignment
