import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PINS = {
    "flash-kda": ("https://github.com/MoonshotAI/FlashKDA.git", "d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72"),
    "DeepGEMM": ("https://github.com/sgl-project/DeepGEMM.git", "dbbb0f4745af11c4c899ea7648888af687394ede"),
}


def prepare(root):
    root.mkdir(parents=True)
    for name, (url, commit) in PINS.items():
        directory = root / name
        subprocess.run(["git", "init", str(directory)], check=True)
        for args in (
            ["remote", "add", "origin", url],
            ["fetch", "--depth=1", "origin", commit],
            ["checkout", "--detach", "FETCH_HEAD"],
            ["submodule", "update", "--init", "--recursive", "--depth=1"],
        ):
            subprocess.run(["git", "-C", str(directory), *args], check=True)
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        assert actual == commit
    shutil.copy2(Path(__file__).with_name("build_glm53_kernels.py"), root / "build_glm53_kernels.py")
    (root / ".dockerignore").write_text("**/.git\n**/__pycache__\n**/_C_build\n**/*.so\n")
    licenses = root / "licenses"
    licenses.mkdir()
    for source, name in (
        ("DeepGEMM/LICENSE", "sgl-deep-gemm-LICENSE"),
        ("DeepGEMM/third-party/cutlass/LICENSE.txt", "sgl-cutlass-LICENSE.txt"),
        ("DeepGEMM/third-party/fmt/LICENSE", "sgl-fmt-LICENSE"),
    ):
        shutil.copy2(root / source, licenses / name)
    receipt = {
        "providers": {name: {"url": url, "commit": commit} for name, (url, commit) in PINS.items()},
        "submodules": {
            name: subprocess.check_output(
                ["git", "-C", str(root / name), "submodule", "status", "--recursive"], text=True
            ).strip()
            for name in PINS
        },
        "files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for name in PINS
            for path in sorted((root / name).rglob("*"))
            if path.is_file() and ".git" not in path.parts
        },
    }
    (root / "kernel_inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output.resolve())
