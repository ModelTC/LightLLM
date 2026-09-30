#!/usr/bin/env python3
"""Convert DeepSeek-V4 block-FP8 scales to UE8M0-equivalent FP32 values.

The output scale tensors remain FP32, but every value is rounded upward to a
power of two with the same rule used by LightLLM's ``USE_UE8M0_SCALE`` path::

    scale = ceil_to_power_of_two(max(amax, 1e-4) / 448)

Only scales paired with 2-D F8_E4M3 weights and matching the block-128 layout
are converted.  Paired FP8 weights and all unrelated tensors are preserved.
The source model is never modified.

Rounding a non-power-of-two scale without requantizing its paired FP8 weight
changes the effective dequantized weight.  This is intentional for this tool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP8_MAX = 448.0
MIN_AMAX = 1.0e-4


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    source_shard: str
    source_dtype: str
    shape: Tuple[int, ...]


def _load_index(model_dir: Path) -> Tuple[Path, dict]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    with index_path.open("r", encoding="utf-8") as file:
        index = json.load(file)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"invalid weight_map in {index_path}")
    return index_path, index


def _inspect_tensor_metadata(model_dir: Path, weight_map: Mapping[str, str]) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    names_by_shard: Dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        names_by_shard.setdefault(shard, []).append(name)

    metadata: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    for shard, names in names_by_shard.items():
        shard_path = model_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            shard_keys = set(file.keys())
            missing = set(names) - shard_keys
            if missing:
                raise ValueError(f"{shard} is missing indexed tensors: {sorted(missing)[:5]}")
            for name in names:
                tensor_slice = file.get_slice(name)
                metadata[name] = (
                    tensor_slice.get_dtype(),
                    tuple(int(dim) for dim in tensor_slice.get_shape()),
                )
    return metadata


def _scale_candidates(weight_name: str) -> Tuple[str, str]:
    base = weight_name[: -len(".weight")]
    return base + ".scale", base + ".weight_scale_inv"


def _inspect_scale_specs(
    weight_map: Mapping[str, str],
    tensor_metadata: Mapping[str, Tuple[str, Tuple[int, ...]]],
    block_size: int,
) -> list[ScaleSpec]:
    specs: list[ScaleSpec] = []
    for weight_name, (weight_dtype, weight_shape) in tensor_metadata.items():
        if weight_dtype != "F8_E4M3" or not weight_name.endswith(".weight"):
            continue
        if len(weight_shape) != 2:
            raise ValueError(f"FP8 weight must be 2-D: {weight_name} has shape {weight_shape}")

        scale_names = [name for name in _scale_candidates(weight_name) if name in tensor_metadata]
        if len(scale_names) != 1:
            raise ValueError(f"expected one paired scale for {weight_name}, found {scale_names}")
        scale_name = scale_names[0]
        scale_dtype, scale_shape = tensor_metadata[scale_name]
        if scale_dtype not in {"F32", "F8_E8M0"}:
            raise ValueError(f"{scale_name} has unsupported dtype {scale_dtype}; expected F32 or F8_E8M0")

        expected_shape = (
            math.ceil(weight_shape[0] / block_size),
            math.ceil(weight_shape[1] / block_size),
        )
        if scale_shape != expected_shape:
            raise ValueError(f"{scale_name} shape is {scale_shape}, expected {expected_shape}")
        specs.append(
            ScaleSpec(
                name=scale_name,
                source_shard=weight_map[scale_name],
                source_dtype=scale_dtype,
                shape=scale_shape,
            )
        )

    if not specs:
        raise ValueError("no block-FP8 scale tensors found")
    return specs


def ceil_to_ue8m0_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Return FP32 powers of two using LightLLM's UE8M0 rounding rule."""

    scale = scale.to(dtype=torch.float32, device="cpu").contiguous()
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("scale contains a non-finite value")
    if bool((scale < 0).any()):
        raise ValueError("scale contains a negative value")

    scale = scale.clamp_min(MIN_AMAX / FP8_MAX)
    bits = scale.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp_(1, 254)
    return (exponent << 23).view(torch.float32)


def _find_specs_needing_conversion(model_dir: Path, specs: Sequence[ScaleSpec]) -> list[ScaleSpec]:
    specs_by_shard: Dict[str, list[ScaleSpec]] = {}
    for spec in specs:
        specs_by_shard.setdefault(spec.source_shard, []).append(spec)

    needed: list[ScaleSpec] = []
    for shard, shard_specs in specs_by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as file:
            for spec in shard_specs:
                source_scale = file.get_tensor(spec.name)
                converted_scale = ceil_to_ue8m0_fp32(source_scale)
                if source_scale.dtype != torch.float32 or not torch.equal(source_scale, converted_scale):
                    needed.append(spec)
    return needed


def _copy_or_link(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepare_output_dir(
    source_dir: Path,
    output_dir: Path,
    *,
    affected_shards: set[str],
    link_mode: str,
) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    for entry in source_dir.iterdir():
        if not entry.is_file() or entry.name == "model.safetensors.index.json":
            continue
        if entry.suffix == ".safetensors" and entry.name in affected_shards:
            continue
        _copy_or_link(entry, output_dir / entry.name, link_mode)


def _rewrite_shard(
    source_path: Path,
    output_path: Path,
    scale_names: set[str],
) -> Tuple[int, int]:
    changed_tensors = 0
    changed_values = 0
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with safe_open(source_path, framework="pt", device="cpu") as source_file:
            source_keys = set(source_file.keys())
            missing = scale_names - source_keys
            if missing:
                raise ValueError(f"{source_path.name} is missing scales: {sorted(missing)[:5]}")

            tensors = {}
            for name in source_file.keys():
                tensor = source_file.get_tensor(name)
                if name in scale_names:
                    converted = ceil_to_ue8m0_fp32(tensor)
                    changed = int(torch.count_nonzero(converted != tensor.to(torch.float32)).item())
                    changed_values += changed
                    changed_tensors += int(changed > 0 or tensor.dtype != torch.float32)
                    tensor = converted
                tensors[name] = tensor.contiguous()
            metadata = source_file.metadata()
            save_file(tensors, temporary_path, metadata=metadata)
        shutil.copymode(source_path, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return changed_tensors, changed_values


def _verify_output(output_dir: Path, specs: Sequence[ScaleSpec]) -> None:
    specs_by_shard: Dict[str, list[ScaleSpec]] = {}
    for spec in specs:
        specs_by_shard.setdefault(spec.source_shard, []).append(spec)

    for shard, shard_specs in specs_by_shard.items():
        with safe_open(output_dir / shard, framework="pt", device="cpu") as file:
            for spec in shard_specs:
                tensor_slice = file.get_slice(spec.name)
                if tensor_slice.get_dtype() != "F32":
                    raise ValueError(f"{spec.name} is {tensor_slice.get_dtype()}, expected F32")
                if tuple(tensor_slice.get_shape()) != spec.shape:
                    raise ValueError(f"{spec.name} shape changed")
                scale = file.get_tensor(spec.name)
                if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                    raise ValueError(f"{spec.name} contains an invalid scale")
                log2_scale = torch.log2(scale)
                if not torch.equal(log2_scale, torch.round(log2_scale)):
                    raise ValueError(f"{spec.name} contains a non-power-of-two value")


def convert(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_model_dir).expanduser().resolve(strict=True)
    output_dir = Path(args.output_model_dir).expanduser().resolve(strict=False)
    if source_dir == output_dir:
        raise ValueError("source and output directories must be different")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    _, index = _load_index(source_dir)
    weight_map: Dict[str, str] = index["weight_map"]
    tensor_metadata = _inspect_tensor_metadata(source_dir, weight_map)
    specs = _inspect_scale_specs(weight_map, tensor_metadata, args.block_size)
    needed_specs = _find_specs_needing_conversion(source_dir, specs)
    specs_by_shard: Dict[str, set[str]] = {}
    for spec in needed_specs:
        specs_by_shard.setdefault(spec.source_shard, set()).add(spec.name)

    source_dtype_counts: Dict[str, int] = {}
    for spec in needed_specs:
        source_dtype_counts[spec.source_dtype] = source_dtype_counts.get(spec.source_dtype, 0) + 1
    rewrite_bytes = sum((source_dir / shard).stat().st_size for shard in specs_by_shard)
    print(f"source: {source_dir}")
    print(f"output: {output_dir}")
    print(f"block-FP8 scales: {len(specs)}; requiring conversion: {len(needed_specs)}")
    print(f"source dtypes requiring conversion: {source_dtype_counts}")
    print(f"affected shards: {len(specs_by_shard)}; rewrite size: {rewrite_bytes / 1024**3:.2f} GiB")
    print("output scales: FP32 powers of two; paired FP8 weights are not requantized")
    if args.dry_run:
        return

    _prepare_output_dir(
        source_dir,
        output_dir,
        affected_shards=set(specs_by_shard),
        link_mode=args.link_mode,
    )

    changed_tensors = 0
    changed_values = 0
    for index_in_plan, shard in enumerate(sorted(specs_by_shard), start=1):
        print(f"[{index_in_plan}/{len(specs_by_shard)}] rewriting {shard}", flush=True)
        shard_changed_tensors, shard_changed_values = _rewrite_shard(
            source_dir / shard,
            output_dir / shard,
            specs_by_shard[shard],
        )
        changed_tensors += shard_changed_tensors
        changed_values += shard_changed_values

    metadata = dict(index.get("metadata") or {})
    if "total_size" in metadata:
        added_bytes = sum(math.prod(spec.shape) * 3 for spec in needed_specs if spec.source_dtype == "F8_E8M0")
        metadata["total_size"] = int(metadata["total_size"]) + added_bytes
    output_index = {**index, "metadata": metadata, "weight_map": weight_map}
    index_path = output_dir / "model.safetensors.index.json"
    temporary_index_path = output_dir / ".model.safetensors.index.json.tmp"
    with temporary_index_path.open("w", encoding="utf-8") as file:
        json.dump(output_index, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_index_path, index_path)

    if not args.no_verify:
        print("verifying converted scales ...", flush=True)
        _verify_output(output_dir, specs)
    print(
        f"conversion complete: {changed_tensors} scale tensors and "
        f"{changed_values} values changed; output: {output_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_model_dir", help="source Hugging Face model directory")
    parser.add_argument("output_model_dir", help="new output model directory")
    parser.add_argument("--block-size", type=int, default=128, help="FP8 block size (default: 128)")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="how unchanged model files and shards are placed in the output directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="inspect and print the conversion plan only")
    parser.add_argument("--no-verify", action="store_true", help="skip final scale verification")
    return parser


def main() -> None:
    convert(build_parser().parse_args())


if __name__ == "__main__":
    main()
