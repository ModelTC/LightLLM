import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import torch
import triton
from safetensors import safe_open

from check_generation import REVISION, verified_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument(
        "--tp", type=int, default=4, help="Vocabulary partition count; this process runs rank zero only"
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 3, 5, 7, 8, 9, 24, 64, 128])
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=40)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        raise RuntimeError("Run through exp")
    if min(args.rows + [args.tp, args.samples, args.rep_ms]) < 1:
        parser.error("Partition count, rows, samples and repeat duration must be positive")
    verified = verified_checkpoint(args.model_directory)
    checkpoint = json.loads((args.model_directory / "config.json").read_text())
    config = checkpoint.get("text_config", checkpoint)
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    if args.tp > vocab:
        parser.error("Partition count cannot exceed vocabulary size")

    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args

    start = StartArgs()
    start.hardware_platform, start.data_type, start.tp, start.dp = "cuda", args.dtype, args.tp, 1
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(vars(start))
    os.environ.update(
        LIGHTLLM_DP_WORLD_SIZE=str(args.tp), LIGHTLLM_CURRENT_RANK_IN_DP="0", LIGHTLLM_CURRENT_DEVICE_ID="0"
    )
    get_env_start_args.cache_clear()
    from lightllm.common.basemodel.layer_weights.meta_weights import LMHeadWeight
    from lightllm.models.qwen4_exp.layer_weights.lm_head_weight import Qwen4ExpLMHeadWeight

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    baseline = LMHeadWeight(hidden, vocab, "lm_head.weight", dtype)
    index = json.loads((args.model_directory / "model.safetensors.index.json").read_text())["weight_map"]
    filename = index["lm_head.weight"]
    with safe_open(args.model_directory / filename, framework="pt", device="cpu") as source:
        baseline.load_hf_weights({"lm_head.weight": source.get_tensor("lm_head.weight")})
    assert baseline.verify_load()
    optimized = Qwen4ExpLMHeadWeight(hidden, vocab, "lm_head.weight", dtype, embedding_weight=baseline)
    assert optimized.verify_load() and optimized.weight is baseline.weight
    torch.manual_seed(3107)
    inputs = torch.randn(max(args.rows), hidden, device="cuda", dtype=dtype)
    reference = torch.cat([baseline(row[:, None]) for row in inputs], dim=1)
    run = Path(os.environ["EXP_RUN_DIR"])
    report = dict(
        status="running",
        scope="trained rank-zero head component; no distributed serving or vLLM/SGLang performance claim",
        revision=REVISION,
        configuration=vars(args) | {"model_directory": str(args.model_directory)},
        local_weight_shape=list(baseline.weight.shape),
        hardware=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.free", "--format=csv,noheader"], text=True
        ),
        software={name: importlib.metadata.version(name) for name in ("torch", "triton")},
        cuda=torch.version.cuda,
        bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        source_file=verified["files"][filename],
        results=[],
    )
    path = run / "results.json"
    try:
        for rows in args.rows:
            tokens = inputs[:rows].T
            old, new = baseline(tokens), optimized(tokens)
            torch.testing.assert_close(new, reference[:, :rows], atol=0, rtol=0)
            timings = {}
            for name, weight in (("baseline", baseline), ("optimized", optimized)):
                timings[name] = [
                    triton.testing.do_bench_cudagraph(lambda: weight(tokens), rep=args.rep_ms) * 1000
                    for _ in range(args.samples)
                ]
            item = dict(
                rows=rows,
                scalar_matches=True,
                baseline_max_abs=float((old - reference[:, :rows]).float().abs().max()),
                microseconds=timings,
                speedup=sorted(timings["baseline"])[args.samples // 2]
                / sorted(timings["optimized"])[args.samples // 2],
            )
            report["results"].append(item)
            path.write_text(json.dumps(report, indent=2))
            print(json.dumps(item), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
