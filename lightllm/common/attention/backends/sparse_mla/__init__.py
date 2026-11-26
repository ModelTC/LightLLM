"""稀疏 MLA：在潜在缓存上吸收 MLA 注意力，限制为每个查询的 top-k 缓存槽。

选择槽的索引器属于模型；这个家族接受其解析后的索引。
"""

from dataclasses import dataclass
from typing import Optional

import torch

KV_CACHE_FORMATS = ("bf16", "fp8_ds_mla", "fp8_group64_e8m0")


@dataclass(frozen=True)
class SparseMLAAttConfig:
    """读取压缩 latent cache 的稀疏 MLA 静态层配置。

    注意力头数按 rank 计算。query 和缓存行都包含 kv_lora_rank 个 latent 通道和
    qk_rope_head_dim 个 RoPE 通道，latent 部分提供 value。NoPE 模型（如 GLM-5）的
    qk_rope_head_dim 为 0。kv_cache_format 支持以下格式。

    - "bf16" 每槽存储一行 [kv_lora_rank + qk_rope_head_dim]。
    - "fp8_ds_mla" 为 DeepSeek V3.2 的打包 FP8 行，每 128 个通道有一个 fp32 scale，
      RoPE 部分以 bf16 存储。
    - "fp8_group64_e8m0" 为 GLM-5 的打包 NoPE 行，每 64 个通道有一个 E8M0 scale，
      后面跟索引器字段。

    topk 是索引器为每个 query 选择缓存槽的上限，层实现无需此参数时为 None。
    对每组 kpool 个 key 评分的索引器会在 top-k 宽度内选择整组，再纳入最新的未满组。
    GLM-5 的索引器已经限制了 key 的可见性，因此 causal=False 时 query 可以读取全部选中的 key。
    """

    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    softmax_scale: float
    dtype: torch.dtype
    kv_cache_format: str = "bf16"
    topk: Optional[int] = None
    kpool: int = 1
    causal: bool = True

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.kv_lora_rank <= 0 or self.kpool <= 0:
            raise ValueError("num_heads, kv_lora_rank and kpool must be positive")
        if self.topk is not None and self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.qk_rope_head_dim < 0:
            raise ValueError(f"qk_rope_head_dim must be >= 0, got {self.qk_rope_head_dim}")
        if self.kv_cache_format not in KV_CACHE_FORMATS:
            raise ValueError(f"kv_cache_format must be one of {KV_CACHE_FORMATS}, got {self.kv_cache_format!r}")


@dataclass(frozen=True)
class SparseMLAPrefillAttInputs:
    """单层 prefill 输入。

    q 的形状为 [query_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]。
    topk_mem_indices 为 [query_tokens, topk] 的缓存槽号，未使用项为 -1。
    "fp8_ds_mla" 还需要本步的全精度 current_kv，形状为
    [new_tokens, 1, kv_lora_rank + qk_rope_head_dim]。topk_indices 表示同一组选中项
    在本步变长 key 布局中的位置。q_kv_start_loc 给出每个 query 所属请求的起始位置，
    仅用于 "fp8_group64_e8m0"。
    """

    q: torch.Tensor
    kv_cache: torch.Tensor
    topk_mem_indices: torch.Tensor
    topk_indices: Optional[torch.Tensor] = None
    current_kv: Optional[torch.Tensor] = None
    q_kv_start_loc: Optional[torch.Tensor] = None
    fp8_workspace: Optional[dict[str, torch.Tensor]] = None


@dataclass(frozen=True)
class SparseMLADecodeAttInputs:
    """一层的解码操作数：q_nope [rows, num_heads, kv_lora_rank]，q_rope [rows, num_heads, qk_rope_head_dim]，缓存和
    topk_mem_indices [rows, topk] 缓存槽位（-1 表示未使用）。
    """

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    kv_cache: torch.Tensor
    topk_mem_indices: torch.Tensor


@dataclass(frozen=True)
class SWASparseMLAAttConfig:
    """稀疏 MLA 在滑动窗口缓存加上可选的压缩缓存（DeepSeek V4）上的静态配置。

    num_heads 是每个 rank 的注意力头数，按 FlashMLA 的 64 头分块补齐。
    每个 query 读取选中的滑窗行。compress_ratio 非零时，它还读取选中的压缩缓存行，
    每 compress_ratio 个 token 压缩成一行。has_attn_sink 为每个头添加一个学习得到的 sink logit。
    """

    num_heads: int
    head_dim_v: int
    softmax_scale: float
    dtype: torch.dtype
    compress_ratio: int = 0
    has_attn_sink: bool = False

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.head_dim_v <= 0:
            raise ValueError("num_heads and head_dim_v must be positive")
        if self.compress_ratio < 0:
            raise ValueError(f"compress_ratio must be >= 0, got {self.compress_ratio}")


@dataclass(frozen=True)
class SWASparseMLAAttInputs:
    """一层的操作数：q [rows, num_heads, head_dim]，FlashMLA 注意力视图的打包 FP8 滑动窗口缓存和压缩缓存（无压缩时为 None），每个缓存的选定行索引及其有效长度，以及当层有
    sink 时的 attn_sink [num_heads]。
    """

    q: torch.Tensor
    swa_cache: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    compressed_cache: Optional[torch.Tensor] = None
    extra_indices: Optional[torch.Tensor] = None
    extra_lengths: Optional[torch.Tensor] = None
    attn_sink: Optional[torch.Tensor] = None
