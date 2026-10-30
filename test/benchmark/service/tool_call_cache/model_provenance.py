"""Hash the actual local model files during an untimed preparation interval.

Run under exp, before service startup or between phases. Reading all weights
during a measured replay would compete for host memory and filesystem bandwidth.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def inspect_model(directory):
    paths = set(directory.glob("*.safetensors"))
    paths.update(directory.glob("*.json"))
    paths.update(directory.glob("*.jinja"))
    for name in ["merges.txt", "vocab.txt", "tokenizer.model", "spiece.model"]:
        if (directory / name).is_file():
            paths.add(directory / name)
    files = []
    for path in sorted(paths):
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"model file changed while hashing: {path}")
        files.append(
            {"name": path.name, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}
        )
        print(f"Hashed {path.name}: {digest.hexdigest()}", flush=True)
    revision = None
    if (directory / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-c", f"safe.directory={directory}", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
    metadata = directory / ".cache/huggingface/download/config.json.metadata"
    if revision is None and metadata.exists():
        revision = metadata.read_text().splitlines()[0]
    return {"directory": str(directory), "recorded_source_revision": revision, "files": files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "EXP_RUN_DIR" not in os.environ:
        parser.error("run this untimed preparation under exp")
    shutil.copy2(__file__, Path(os.environ["EXP_RUN_DIR"]) / "model_provenance.py")
    result = {"models": [inspect_model(path.resolve()) for path in args.models]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(args.output, Path(os.environ["EXP_RUN_DIR"]) / "model-provenance.json")


if __name__ == "__main__":
    main()
