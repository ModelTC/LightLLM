import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch


source = Path("/build/DeepGEMM")
package = source / "sgl_deep_gemm"
for name in ("utils", "testing", "legacy", "mega"):
    shutil.copytree(source / "deep_gemm" / name, package / name, dirs_exist_ok=True)
sys.path.insert(0, str(source))
import sgl_deep_gemm  # noqa: E402


for name in ("fp8_mega_moe", "mega_moe_pre_dispatch_sm90", "get_symm_buffer_for_sm90_mega_moe"):
    assert callable(getattr(sgl_deep_gemm, name)), name

destination = Path("/build/sgl_deep_gemm")
destination.mkdir()
for name in ("__init__.py", "cuda_helpers.py", "VERSION", "_C.so", "LICENSE"):
    shutil.copy2(package / name, destination / name)
for name in ("utils", "testing", "legacy", "mega"):
    shutil.copytree(package / name, destination / name, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(source / "deep_gemm/include", destination / "include")
for name in ("cute", "cutlass"):
    shutil.copytree(source / "third-party/cutlass/include" / name, destination / "include" / name)
receipt = {
    "flash_kda": "d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72",
    "sgl_deep_gemm": "dbbb0f4745af11c4c899ea7648888af687394ede",
    "flash_kda_cuda_archs": os.environ["FLASH_KDA_CUDA_ARCHS"],
    "sgl_deep_gemm_cuda_archs": os.environ["TVM_FFI_CUDA_ARCH_LIST"],
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "sgl_deep_gemm_native_sha256": hashlib.sha256((destination / "_C.so").read_bytes()).hexdigest(),
}
Path("/build/kernel_build.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
