import argparse
import os
import shutil
import tempfile
from pathlib import Path

from tvm_ffi.cpp import load_inline


def build(source: Path, destination: Path):
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0"
    destination.mkdir(parents=True, exist_ok=True)
    targets = [
        ("repack", "marlin/awq_marlin_repack.cuh", "awq_marlin_repack", "awq_marlin_repack"),
        ("dense_fp16", "marlin/gptq_marlin.cuh", "gptq_marlin_gemm", "gptq_marlin_gemm<fp16_t>"),
        ("dense_bf16", "marlin/gptq_marlin.cuh", "gptq_marlin_gemm", "gptq_marlin_gemm<bf16_t>"),
        ("moe_fp16", "marlin_moe/moe_wna16_marlin.cuh", "moe_wna16_marlin_gemm", "moe_wna16_marlin_gemm<fp16_t>"),
        ("moe_bf16", "marlin_moe/moe_wna16_marlin.cuh", "moe_wna16_marlin_gemm", "moe_wna16_marlin_gemm<bf16_t>"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for name, header, symbol, template in targets:
            build_dir = Path(directory) / name
            build_dir.mkdir()
            load_inline(
                name,
                cpp_sources=[],
                cuda_sources=[
                    f'#include "{source / "csrc" / "gemm" / header}"',
                    f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({symbol}, ({template}));",
                ],
                extra_cflags=["-std=c++20", "-O3"],
                extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr", "-DSGL_CUDA_ARCH=900"],
                extra_include_paths=[str(source / "include")],
                build_directory=str(build_dir),
            )
            shutil.copyfile(build_dir / f"{name}.so", destination / f"{name}.so")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)
