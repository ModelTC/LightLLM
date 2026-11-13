"""Freeze the supplied launcher baseline and this working tree for isolated A/B."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("baseline_manifest.json")


def verify_baseline():
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Baseline file changed: {name}: expected {expected}, got {actual}")
    return manifest


def package_hashes(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "lightllm").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def freeze_bundle(root, output):
    subprocess.run(
        ["tar", "--exclude=__pycache__", "--exclude=*.pyc", "-czf", str(output), "-C", str(root), "lightllm"],
        check=True,
    )
    return {"path": str(output), "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest()}


def prepare(destination, precision=False):
    manifest = verify_baseline()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    # Original deployment runs its code on local disk. Keep frozen package
    # bundles on AFS for review, while staging/executing code on local disk.
    scratch = Path(tempfile.mkdtemp(prefix="bsh-nsa-snapshot-"))
    runtimes = {}
    hashes = {}
    bundles = {}
    candidate_source_files = None
    for role, source in (("baseline", Path(manifest["runtime_source"])), ("candidate", REPO)):
        target = scratch / role
        target.mkdir()
        if role == "baseline":
            archive_root = Path("/mnt/afs-toolcall/yankuihao/code/lightllm_dsa_perf_20261107")
            for archive in ("source", "original_changes", "changes"):
                subprocess.run(
                    ["tar", "--no-same-owner", "-xzf", str(archive_root / f"{archive}.tar.gz"), "-C", str(target)],
                    check=True,
                )
            overlay = Path(
                "/mnt/afs-toolcall/yankuihao/code/dsa_quick_perf_20261107/accepted/lightllm/models/sensenova_loop"
            )
            for name in ("dsa_layer_infer.py", "quick_rope.py"):
                shutil.copyfile(overlay / name, target / "lightllm/models/sensenova_loop" / name)
        else:
            archive = scratch / "candidate-source.tar.gz"
            freeze_bundle(source, archive)
            subprocess.run(["tar", "--no-same-owner", "-xzf", str(archive), "-C", str(target)], check=True)
            candidate_source_files = package_hashes(target)
        if precision:
            model = target / "lightllm/models/sensenova_loop/model.py"
            shutil.copyfile(Path(__file__).with_name("precision_hooks.py"), model.with_name("bsh_precision_hooks.py"))
            with model.open("a") as f:
                f.write(
                    "\nfrom .bsh_precision_hooks import install as _bsh_install\n"
                    "_bsh_install(SenseNovaLoopTpPartModel)\n"
                )
        runtimes[role] = str(target)
        hashes[role] = package_hashes(target)
        bundles[role] = freeze_bundle(target, destination / (role + ".tar.gz"))
    result = {
        "baseline": manifest,
        "runtimes": runtimes,
        "files": hashes,
        "precision_hooks": precision,
        "candidate_source_files": candidate_source_files,
        "snapshot_bundles": bundles,
        "candidate_git_head": subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO}", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
        ).strip(),
    }
    (destination / "runtime_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_baseline(), indent=2))
    else:
        if args.output is None:
            parser.error("--output is required unless --verify-only is set")
        print(json.dumps(prepare(args.output, args.precision)["runtimes"], indent=2))


if __name__ == "__main__":
    main()
