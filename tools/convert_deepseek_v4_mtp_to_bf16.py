#!/usr/bin/env python3
"""Convert DeepSeek-V4 ``mtp.*`` checkpoint tensors to BF16.

The DeepSeek-V4-Flash checkpoint stores dense MTP matrices as block-FP8 and
routed-expert matrices as packed MXFP4.  Casting those tensors directly would
not recover their values.  This tool performs the corresponding dequantization,
removes the paired scale tensors from the output index, and writes every MTP
tensor as BF16.

Non-MTP shards are linked or copied into a new model directory.  The source
model is never modified.  This changes checkpoint storage only; the runtime
must select the no-quantization path for the converted MTP layers while keeping
the target layers on their original quantization path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    source_shard: str
    source_dtype: str
    source_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    output_nbytes: int
    conversion: str
    scale_name: str | None = None


def _numel(shape: Sequence[int]) -> int:
    return math.prod(int(dim) for dim in shape)


def _tensor_nbytes(dtype: str, shape: Sequence[int]) -> int:
    try:
        item_size = _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype: {dtype}") from exc
    return _numel(shape) * item_size


def _scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"quantized tensor does not end in .weight: {weight_name}")
    return weight_name[: -len(".weight")] + ".scale"


def _parse_size(value: str) -> int:
    text = value.strip().upper().replace(" ", "")
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            if not number:
                raise argparse.ArgumentTypeError(f"invalid size: {value}")
            size = int(float(number) * units[suffix])
            if size <= 0:
                raise argparse.ArgumentTypeError(f"size must be positive: {value}")
            return size
    try:
        size = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size: {value}") from exc
    if size <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive: {value}")
    return size


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device, non_blocking=device.type == "cuda")


def dequantize_fp8_block(
    weight_slice,
    scale: torch.Tensor,
    *,
    output_shape: Sequence[int],
    block_size: int,
    row_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Dequantize a 2-D E4M3 matrix with a 2-D E8M0 block scale."""

    rows, cols = (int(dim) for dim in output_shape)
    expected_scale_shape = (math.ceil(rows / block_size), math.ceil(cols / block_size))
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"FP8 scale shape mismatch: expected {expected_scale_shape}, got {tuple(scale.shape)}"
        )

    output = torch.empty((rows, cols), dtype=torch.bfloat16, device="cpu")
    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        scale_row_start = row_start // block_size
        scale_row_end = math.ceil(row_end / block_size)
        scale_row_offset = row_start - scale_row_start * block_size

        weight_chunk = _to_device(weight_slice[row_start:row_end], device).to(
            torch.float32
        )
        scale_chunk = _to_device(scale[scale_row_start:scale_row_end], device).to(
            torch.float32
        )
        expanded_scale = scale_chunk.repeat_interleave(block_size, dim=0)
        expanded_scale = expanded_scale[
            scale_row_offset : scale_row_offset + row_end - row_start
        ].repeat_interleave(block_size, dim=1)[:, :cols]
        converted = (weight_chunk * expanded_scale).to(torch.bfloat16).cpu()
        output[row_start:row_end].copy_(converted)
        del weight_chunk, scale_chunk, expanded_scale, converted
    return output


def dequantize_mxfp4(
    packed_weight_slice,
    scale: torch.Tensor,
    *,
    packed_shape: Sequence[int],
    block_size: int,
    row_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Unpack a 2-D MXFP4 matrix and apply its per-K-block E8M0 scale."""

    rows, packed_cols = (int(dim) for dim in packed_shape)
    logical_cols = packed_cols * 2
    if logical_cols % block_size != 0:
        raise ValueError(
            f"MXFP4 logical K dimension {logical_cols} is not divisible by block size {block_size}"
        )
    expected_scale_shape = (rows, logical_cols // block_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"MXFP4 scale shape mismatch: expected {expected_scale_shape}, got {tuple(scale.shape)}"
        )

    output = torch.empty((rows, logical_cols), dtype=torch.bfloat16, device="cpu")
    lookup = torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)
    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        packed = _to_device(packed_weight_slice[row_start:row_end], device).view(
            torch.uint8
        )
        low = (packed & 0x0F).to(torch.long)
        high = (packed >> 4).to(torch.long)

        values = torch.empty(
            (row_end - row_start, logical_cols), dtype=torch.float32, device=device
        )
        values[:, 0::2] = lookup[low]
        values[:, 1::2] = lookup[high]
        scale_chunk = _to_device(scale[row_start:row_end], device).to(torch.float32)
        expanded_scale = scale_chunk.repeat_interleave(block_size, dim=1)
        converted = (values * expanded_scale).to(torch.bfloat16).cpu()
        output[row_start:row_end].copy_(converted)
        del packed, low, high, values, scale_chunk, expanded_scale, converted
    return output


def _load_index(model_dir: Path) -> Tuple[Path, dict]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    with index_path.open("r", encoding="utf-8") as file:
        index = json.load(file)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"invalid weight_map in {index_path}")
    return index_path, index


def _inspect_specs(
    model_dir: Path,
    weight_map: Mapping[str, str],
    *,
    prefix: str,
    fp8_block_size: int,
    mxfp4_block_size: int,
) -> Tuple[List[TensorSpec], set[str], int]:
    mtp_weight_map = {
        name: shard for name, shard in weight_map.items() if name.startswith(prefix)
    }
    if not mtp_weight_map:
        raise ValueError(f"no checkpoint tensors start with {prefix!r}")

    names_by_shard: Dict[str, List[str]] = {}
    for name, shard in mtp_weight_map.items():
        names_by_shard.setdefault(shard, []).append(name)

    tensor_meta: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    original_mtp_nbytes = 0
    for shard, names in names_by_shard.items():
        shard_path = model_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            shard_keys = set(file.keys())
            missing = set(names) - shard_keys
            if missing:
                raise ValueError(
                    f"{shard} is missing indexed tensors: {sorted(missing)[:5]}"
                )
            for name in names:
                tensor_slice = file.get_slice(name)
                dtype = tensor_slice.get_dtype()
                shape = tuple(int(dim) for dim in tensor_slice.get_shape())
                tensor_meta[name] = (dtype, shape)
                original_mtp_nbytes += _tensor_nbytes(dtype, shape)

    paired_scales: set[str] = set()
    specs: List[TensorSpec] = []
    for name in mtp_weight_map:
        dtype, shape = tensor_meta[name]
        if dtype == "F8_E8M0":
            continue
        if dtype == "F8_E4M3":
            if len(shape) != 2:
                raise ValueError(f"FP8 tensor must be 2-D: {name} has shape {shape}")
            scale_name = _scale_name(name)
            if tensor_meta.get(scale_name, (None,))[0] != "F8_E8M0":
                raise ValueError(
                    f"missing E8M0 scale for {name}: expected {scale_name}"
                )
            scale_shape = tensor_meta[scale_name][1]
            expected_scale_shape = (
                math.ceil(shape[0] / fp8_block_size),
                math.ceil(shape[1] / fp8_block_size),
            )
            if scale_shape != expected_scale_shape:
                raise ValueError(
                    f"FP8 scale shape mismatch for {name}: expected {expected_scale_shape}, got {scale_shape}"
                )
            paired_scales.add(scale_name)
            output_shape = shape
            conversion = "fp8_block"
        elif dtype == "I8":
            if len(shape) != 2:
                raise ValueError(
                    f"packed MXFP4 tensor must be 2-D: {name} has shape {shape}"
                )
            scale_name = _scale_name(name)
            if tensor_meta.get(scale_name, (None,))[0] != "F8_E8M0":
                raise ValueError(
                    f"missing E8M0 scale for {name}: expected {scale_name}"
                )
            output_shape = (shape[0], shape[1] * 2)
            expected_scale_shape = (
                output_shape[0],
                output_shape[1] // mxfp4_block_size,
            )
            if tensor_meta[scale_name][1] != expected_scale_shape:
                raise ValueError(
                    f"MXFP4 scale shape mismatch for {name}: expected {expected_scale_shape}, "
                    f"got {tensor_meta[scale_name][1]}"
                )
            paired_scales.add(scale_name)
            conversion = "mxfp4"
        elif dtype in {"BF16", "F16", "F32"}:
            scale_name = None
            output_shape = shape
            conversion = "cast"
        else:
            raise ValueError(
                f"cannot convert {name} with dtype {dtype} to BF16 without model-specific semantics"
            )

        specs.append(
            TensorSpec(
                name=name,
                source_shard=mtp_weight_map[name],
                source_dtype=dtype,
                source_shape=shape,
                output_shape=output_shape,
                output_nbytes=_numel(output_shape) * 2,
                conversion=conversion,
                scale_name=scale_name,
            )
        )

    orphan_scales = {
        name for name, (dtype, _) in tensor_meta.items() if dtype == "F8_E8M0"
    } - paired_scales
    if orphan_scales:
        raise ValueError(f"orphan MTP E8M0 scales: {sorted(orphan_scales)[:10]}")
    return specs, paired_scales, original_mtp_nbytes


def _plan_shards(
    specs: Sequence[TensorSpec], max_shard_size: int
) -> List[List[TensorSpec]]:
    shards: List[List[TensorSpec]] = []
    current: List[TensorSpec] = []
    current_size = 0
    for spec in specs:
        if current and current_size + spec.output_nbytes > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(spec)
        current_size += spec.output_nbytes
    if current:
        shards.append(current)
    return shards


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
    referenced_non_mtp_shards: Iterable[str],
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
        if entry.suffix == ".safetensors":
            continue
        _copy_or_link(entry, output_dir / entry.name, link_mode)

    for shard in sorted(set(referenced_non_mtp_shards)):
        _copy_or_link(source_dir / shard, output_dir / shard, link_mode)


def _convert_spec(
    spec: TensorSpec,
    source_file,
    *,
    fp8_block_size: int,
    mxfp4_block_size: int,
    row_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if spec.conversion == "cast":
        return source_file.get_tensor(spec.name).to(torch.bfloat16).contiguous()

    assert spec.scale_name is not None
    weight_slice = source_file.get_slice(spec.name)
    scale = source_file.get_tensor(spec.scale_name)
    if spec.conversion == "fp8_block":
        return dequantize_fp8_block(
            weight_slice,
            scale,
            output_shape=spec.output_shape,
            block_size=fp8_block_size,
            row_chunk_size=row_chunk_size,
            device=device,
        )
    if spec.conversion == "mxfp4":
        return dequantize_mxfp4(
            weight_slice,
            scale,
            packed_shape=spec.source_shape,
            block_size=mxfp4_block_size,
            row_chunk_size=row_chunk_size,
            device=device,
        )
    raise AssertionError(f"unknown conversion: {spec.conversion}")


def _write_converted_shards(
    source_dir: Path,
    output_dir: Path,
    shard_plan: Sequence[Sequence[TensorSpec]],
    *,
    fp8_block_size: int,
    mxfp4_block_size: int,
    row_chunk_size: int,
    device: torch.device,
) -> Dict[str, str]:
    source_shards = sorted(
        {spec.source_shard for shard in shard_plan for spec in shard}
    )
    output_weight_map: Dict[str, str] = {}
    with ExitStack() as stack:
        source_files = {
            shard: stack.enter_context(
                safe_open(source_dir / shard, framework="pt", device="cpu")
            )
            for shard in source_shards
        }
        shard_count = len(shard_plan)
        for shard_index, shard_specs in enumerate(shard_plan, start=1):
            output_name = f"mtp-bf16-{shard_index:05d}-of-{shard_count:05d}.safetensors"
            planned_bytes = sum(spec.output_nbytes for spec in shard_specs)
            print(
                f"[{shard_index}/{shard_count}] converting {len(shard_specs)} tensors "
                f"({planned_bytes / 1024**3:.2f} GiB) -> {output_name}",
                flush=True,
            )
            tensors = {
                spec.name: _convert_spec(
                    spec,
                    source_files[spec.source_shard],
                    fp8_block_size=fp8_block_size,
                    mxfp4_block_size=mxfp4_block_size,
                    row_chunk_size=row_chunk_size,
                    device=device,
                )
                for spec in shard_specs
            }
            temporary_path = output_dir / f".{output_name}.tmp"
            output_path = output_dir / output_name
            save_file(tensors, temporary_path, metadata={"format": "pt"})
            os.replace(temporary_path, output_path)
            for spec in shard_specs:
                output_weight_map[spec.name] = output_name
            del tensors
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return output_weight_map


def _verify_output(
    output_dir: Path,
    specs: Sequence[TensorSpec],
    converted_weight_map: Mapping[str, str],
) -> None:
    expected_by_shard: Dict[str, Dict[str, TensorSpec]] = {}
    for spec in specs:
        expected_by_shard.setdefault(converted_weight_map[spec.name], {})[
            spec.name
        ] = spec

    for shard, expected in expected_by_shard.items():
        with safe_open(output_dir / shard, framework="pt", device="cpu") as file:
            actual_keys = set(file.keys())
            if actual_keys != set(expected):
                raise ValueError(
                    f"output key mismatch in {shard}: missing={set(expected) - actual_keys}, "
                    f"unexpected={actual_keys - set(expected)}"
                )
            for name, spec in expected.items():
                tensor_slice = file.get_slice(name)
                if tensor_slice.get_dtype() != "BF16":
                    raise ValueError(
                        f"{name} is {tensor_slice.get_dtype()}, expected BF16"
                    )
                if tuple(tensor_slice.get_shape()) != spec.output_shape:
                    raise ValueError(
                        f"{name} shape is {tuple(tensor_slice.get_shape())}, expected {spec.output_shape}"
                    )


def convert(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_model_dir).expanduser().resolve(strict=True)
    output_dir = Path(args.output_model_dir).expanduser().resolve(strict=False)
    if source_dir == output_dir:
        raise ValueError("source and output directories must be different")
    if args.row_chunk_size <= 0:
        raise ValueError("--row-chunk-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA conversion requested but torch.cuda.is_available() is false"
        )

    _, index = _load_index(source_dir)
    weight_map: Dict[str, str] = index["weight_map"]
    specs, paired_scales, original_mtp_nbytes = _inspect_specs(
        source_dir,
        weight_map,
        prefix=args.prefix,
        fp8_block_size=args.fp8_block_size,
        mxfp4_block_size=args.mxfp4_block_size,
    )
    shard_plan = _plan_shards(specs, args.max_shard_size)
    output_mtp_nbytes = sum(spec.output_nbytes for spec in specs)
    conversion_counts: Dict[str, int] = {}
    for spec in specs:
        conversion_counts[spec.conversion] = (
            conversion_counts.get(spec.conversion, 0) + 1
        )

    print(f"source: {source_dir}")
    print(f"output: {output_dir}")
    print(f"MTP output tensors: {len(specs)}; removed scales: {len(paired_scales)}")
    print(f"conversions: {conversion_counts}")
    print(
        f"MTP size: {original_mtp_nbytes / 1024**3:.2f} GiB -> "
        f"{output_mtp_nbytes / 1024**3:.2f} GiB in {len(shard_plan)} shards"
    )
    if args.dry_run:
        return

    non_mtp_weight_map = {
        name: shard
        for name, shard in weight_map.items()
        if not name.startswith(args.prefix)
    }
    _prepare_output_dir(
        source_dir,
        output_dir,
        referenced_non_mtp_shards=non_mtp_weight_map.values(),
        link_mode=args.link_mode,
    )
    converted_weight_map = _write_converted_shards(
        source_dir,
        output_dir,
        shard_plan,
        fp8_block_size=args.fp8_block_size,
        mxfp4_block_size=args.mxfp4_block_size,
        row_chunk_size=args.row_chunk_size,
        device=device,
    )

    new_weight_map: Dict[str, str] = {}
    for name, shard in weight_map.items():
        if not name.startswith(args.prefix):
            new_weight_map[name] = shard
        elif name in converted_weight_map:
            new_weight_map[name] = converted_weight_map[name]
        elif name not in paired_scales:
            raise AssertionError(
                f"MTP tensor was neither converted nor removed: {name}"
            )

    metadata = dict(index.get("metadata") or {})
    if "total_size" in metadata:
        metadata["total_size"] = (
            int(metadata["total_size"]) - original_mtp_nbytes + output_mtp_nbytes
        )
    new_index = {"metadata": metadata, "weight_map": new_weight_map}
    index_path = output_dir / "model.safetensors.index.json"
    temporary_index_path = output_dir / ".model.safetensors.index.json.tmp"
    with temporary_index_path.open("w", encoding="utf-8") as file:
        json.dump(new_index, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_index_path, index_path)

    manifest = {
        "source_model_dir": str(source_dir),
        "weight_prefix": args.prefix,
        "output_dtype": "bfloat16",
        "runtime_note": (
            "Configure converted MTP layers with quant type 'none'; target layers "
            "remain on their original quantization path."
        ),
        "fp8_block_size": args.fp8_block_size,
        "mxfp4_block_size": args.mxfp4_block_size,
        "converted_tensor_count": len(specs),
        "removed_scale_count": len(paired_scales),
        "original_mtp_bytes": original_mtp_nbytes,
        "output_mtp_bytes": output_mtp_nbytes,
    }
    with (output_dir / "mtp_bf16_conversion.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    if not args.no_verify:
        print("verifying converted shards ...", flush=True)
        _verify_output(output_dir, specs, converted_weight_map)
    print(f"conversion complete: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_model_dir", help="source Hugging Face model directory")
    parser.add_argument("output_model_dir", help="new output model directory")
    parser.add_argument(
        "--prefix", default="mtp.", help="checkpoint key prefix to convert"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="conversion device, for example cpu or cuda:0 (output shards are always stored on CPU)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=_parse_size,
        default=_parse_size("2GiB"),
        help="maximum planned size of each converted shard (default: 2GiB)",
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=256,
        help="number of matrix rows converted at once (default: 256)",
    )
    parser.add_argument("--fp8-block-size", type=int, default=128)
    parser.add_argument("--mxfp4-block-size", type=int, default=32)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="how unchanged model files are placed in the output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect and print the conversion plan only",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the final dtype/key/shape verification",
    )
    return parser


def main() -> None:
    convert(build_parser().parse_args())


if __name__ == "__main__":
    main()
