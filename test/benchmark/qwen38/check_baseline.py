import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def check_runtime(framework, model_directory):
    report = {
        "framework": framework,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "glibc": platform.libc_ver(),
        "executable": sys.executable,
        "scope": "Dependency/import/GPU smoke checks only; no serving or performance result",
        "software": {},
        "status": "checking",
    }
    for package in (framework, "torch", "triton", "transformers", "flashinfer-python"):
        try:
            report["software"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["software"][package] = None
    try:
        dependencies = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        report["dependencies"] = {
            "exit_code": dependencies.returncode,
            "output": dependencies.stdout + dependencies.stderr,
        }
        if dependencies.returncode:
            raise RuntimeError("Baseline dependency validation failed")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Baseline cannot access CUDA")
        report.update(cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(), gpu=torch.cuda.get_device_name())
        a = torch.ones(16, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(64, 32, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(a @ b, torch.full((16, 32), 64, device="cuda", dtype=torch.bfloat16), atol=0, rtol=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = a @ b
        a.fill_(2)
        graph.replay()
        torch.testing.assert_close(captured, torch.full_like(captured, 128), atol=0, rtol=0)
        graph.reset()
        report["bf16_matmul_and_graph"] = "passed"
        architecture = "Qwen4ExpForConditionalGeneration"
        if framework == "vllm":
            from vllm.config import ModelConfig
            from vllm.model_executor.models import ModelRegistry

            config = ModelConfig(
                model=str(model_directory.resolve()), skip_tokenizer_init=True, dtype="bfloat16", max_model_len=32
            )
            model_class, resolved_architecture = ModelRegistry.resolve_model_cls(architecture, config)
            if resolved_architecture != architecture:
                raise RuntimeError("Baseline resolved a different model architecture")
        else:
            from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

            model_class = Qwen4ExpForConditionalGeneration
        source = Path(inspect.getfile(model_class))
        report["model"] = {
            "architecture": architecture,
            "module": model_class.__module__,
            "source": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    return report


def main():
    if not os.environ.get("EXP_ACTIVE"):
        raise RuntimeError("Run baseline checks through exp")
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, default=Path("artifacts/references/qwen4_exp/layer0"))
    args = parser.parse_args()
    result = check_runtime(args.framework, args.model_directory)
    payload = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    (Path(os.environ["EXP_RUN_DIR"]) / "baseline-readiness.json").write_text(payload)
    print(payload, flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
