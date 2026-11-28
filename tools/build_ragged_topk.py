import argparse
import os
import shutil
import tempfile
from pathlib import Path

from tvm_ffi.cpp import load_inline


def build(source: Path, destination: Path):
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        load_inline(
            "topk",
            cpp_sources=[],
            cuda_sources=[
                f'#include "{source / "csrc" / "deepseek_v4" / "topk_v2.cuh"}"',
                "TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, sglang::TopKKernel<true>::plan);",
                "TVM_FFI_DLL_EXPORT_TYPED_FUNC(select_topk, sglang::TopKKernel<true>::transform_paged);",
            ],
            extra_cflags=["-std=c++20", "-O3"],
            extra_cuda_cflags=[
                "-std=c++20",
                "-O3",
                "--expt-relaxed-constexpr",
                "-DSGL_CUDA_ARCH=900",
                "-DSGL_TOPK_V2_MAX_C8_OCC2=0",
                "-DSGL_TOPK_V2_MAX_C16_OCC1=0",
                "-DSGL_TOPK_MODE=1",
            ],
            extra_include_paths=[str(source / "include")],
            build_directory=directory,
        )
        shutil.copyfile(Path(directory) / "topk.so", destination / "topk.so")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)
