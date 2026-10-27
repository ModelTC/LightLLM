"""Rebuild the pinned DeepEP with its NVSHMEM 3.6.5 RC-QP layout fix."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

BASE_REVISION = "60d44037a702f651a6e18bd4aea65ed8409051c2"
PATCH_SHA256 = "592bafb79261941eba1d126bfa9fdb89ed4cd0e28f8983c57836492a65612126"


def main():
    import torch

    assert torch.__version__ == "2.11.0+cu130", torch.__version__
    assert importlib.metadata.version("nvidia-nvshmem-cu13") == "3.6.5"
    assert importlib.metadata.version("nvidia-nccl-cu13") == "2.30.4"
    patch = Path(__file__).with_name("patches") / "deepep-pr696-rc-qp-layout.patch"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == PATCH_SHA256
    original = Path("/root/DeepEP-60d4403")
    assert subprocess.check_output(["git", "-C", str(original), "rev-parse", "HEAD"], text=True).strip() == BASE_REVISION
    assert not subprocess.check_output(["git", "-C", str(original), "status", "--porcelain"], text=True).strip()
    output = Path("/opt/lightllm-build")
    wheels = output / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="deepep-rcqp-") as temporary:
        source = Path(temporary) / "source"
        subprocess.run(["git", "clone", "--shared", str(original), str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", BASE_REVISION], check=True)
        subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
        include_order = patch.with_name("deepep-torch-half-include-order.patch")
        subprocess.run(["git", "-C", str(source), "apply", "--check", str(include_order)], check=True)
        subprocess.run(["git", "-C", str(source), "apply", str(include_order)], check=True)
        fmt = source / "third-party/fmt"
        fmt.rmdir()
        shutil.copytree(original / "third-party/fmt", fmt, ignore=shutil.ignore_patterns(".git"))
        environment = dict(os.environ, CUDA_HOME="/usr/local/cuda", TORCH_CUDA_ARCH_LIST="9.0")
        environment.setdefault("MAX_JOBS", "16")
        subprocess.run(
            ["/opt/conda/bin/python", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "--wheel-dir", str(wheels), str(source)],
            env=environment, check=True,
        )
    built = list(wheels.glob("deep_ep-*.whl"))
    assert len(built) == 1, built
    wheel = built[0]
    with zipfile.ZipFile(wheel) as archive:
        native = [name for name in archive.namelist() if name.startswith("deep_ep/_C.") and name.endswith(".so")]
        assert len(native) == 1, native
        native_hash = hashlib.sha256(archive.read(native[0])).hexdigest()
    subprocess.run(
        ["/opt/conda/bin/python", "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True,
    )
    receipt = {
        "base_revision": BASE_REVISION,
        "upstream_patch": "https://github.com/deepseek-ai/DeepEP/pull/696",
        "patch_sha256": PATCH_SHA256,
        "include_order_patch_sha256": hashlib.sha256(include_order.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "nvshmem": importlib.metadata.version("nvidia-nvshmem-cu13"),
        "nccl": importlib.metadata.version("nvidia-nccl-cu13"),
        "cuda_arch": "9.0",
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "native_path": native[0],
        "native_sha256": native_hash,
    }
    (output / "deepep_rcqp.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
