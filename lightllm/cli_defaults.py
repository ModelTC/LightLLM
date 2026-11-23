"""Small, conservative defaults for the one-line ``lightllm serve`` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lightllm.common.model_config import normalize_model_config


_QWEN_HYBRID_MODEL_TYPES = {"qwen3_next", "qwen3_5", "qwen3_5_moe"}
_KV_REPLICATION_MODEL_TYPES = {
    "qwen2",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    *_QWEN_HYBRID_MODEL_TYPES,
}


def _gpu_memories_mib() -> list[int]:
    """Return visible GPU memory sizes without making CUDA a hard CLI dependency."""
    try:
        import torch

        if torch.cuda.is_available():
            return [
                int(torch.cuda.get_device_properties(i).total_memory // (1024 * 1024))
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass

    # This fallback keeps the command useful in a minimal install and in
    # containers where importing torch fails before the server does.
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _model_weight_bytes(model_dir: str) -> Optional[int]:
    """Estimate weight storage from local files, without opening model tensors."""
    root = Path(model_dir)
    if not root.is_dir():
        return None

    # An index avoids counting files that are present but not referenced by the
    # model (for example, a stale shard left by a previous download).
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = root / index_name
        if index_path.is_file():
            try:
                weight_map = json.loads(index_path.read_text()).get("weight_map", {})
                shards = {root / name for name in weight_map.values()}
                sizes = [path.stat().st_size for path in shards if path.is_file()]
                if sizes:
                    return sum(sizes)
            except (OSError, ValueError, TypeError):
                pass

    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    try:
        sizes = [path.stat().st_size for path in root.rglob("*") if path.is_file() and path.suffix in suffixes]
    except OSError:
        return None
    return sum(sizes) or None


def _tp_head_constraints(model_dir: str) -> Optional[tuple[list[int], int, bool]]:
    try:
        model_config = json.loads((Path(model_dir) / "config.json").read_text())
        model_type = model_config.get("model_type")
        config = normalize_model_config(model_config)
        query_heads = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads", query_heads))
        sharded_heads = [query_heads]
        if model_type in _QWEN_HYBRID_MODEL_TYPES:
            for key in ("linear_num_key_heads", "linear_num_value_heads"):
                if key in config:
                    sharded_heads.append(int(config[key]))
        if kv_heads <= 0 or any(heads <= 0 for heads in sharded_heads):
            return None
        return sharded_heads, kv_heads, model_type in _KV_REPLICATION_MODEL_TYPES
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def resolve_tp(model_dir: str) -> int:
    """Choose the smallest visible TP that conservatively fits the weights.

    The estimate deliberately leaves room for KV cache and runtime buffers. If
    hardware or model metadata cannot be inspected, the legacy single-GPU
    default is retained.
    """
    memories = _gpu_memories_mib()
    if len(memories) <= 1:
        return 1
    weight_bytes = _model_weight_bytes(model_dir)
    if not weight_bytes:
        return 1

    head_constraints = _tp_head_constraints(model_dir)
    if head_constraints is None:
        return 1
    sharded_heads, kv_heads, allow_kv_replication = head_constraints
    usable_per_gpu = min(memories) * 1024 * 1024 * 0.75
    # Runtime allocations and allocator fragmentation are not represented by
    # weight files. The factor is intentionally conservative for a zero-config
    # launcher; users can override it with --tp and --mem_fraction.
    required_bytes = weight_bytes * 1.20
    largest_valid_tp = 1
    for tp in range(1, len(memories) + 1):
        if any(heads % tp for heads in sharded_heads):
            continue
        if kv_heads % tp and not (allow_kv_replication and tp % kv_heads == 0):
            continue
        largest_valid_tp = tp
        if required_bytes / tp <= usable_per_gpu:
            return tp
    return largest_valid_tp
