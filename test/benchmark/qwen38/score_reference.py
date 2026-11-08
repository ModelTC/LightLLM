import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors import safe_open

from check_generation import REVISION, cases_for, verified_checkpoint

SOURCE_SHA256 = "87c27c88424cea73d5d8d9179f76a77a97e6e4249ce43ecff79a8b255fab0504"


def load_source(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Reference equations do not match the pinned source")
    name = "transformers.models.qwen4_exp.modeling_qwen4_exp"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_embedding_rows(model, root):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    weight = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight
    if weight.is_meta or tuple(weight.shape) != (320001536, 160) or weight.dtype != torch.bfloat16:
        raise ValueError("Reference lexical table has the wrong shape, residency or dtype")
    checked = 0
    for shard in range(128):
        key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        with safe_open(root / index[key], framework="pt", device="cpu") as source:
            tensor = source.get_slice(key)
            rows, width = tensor.get_shape()
            if (rows, width) != (2500012, 160):
                raise ValueError(f"Unexpected source shard shape: {key}")
            for row in (0, rows // 2, rows - 1):
                actual = weight[shard * rows + row].cpu()
                torch.testing.assert_close(actual, tensor[row : row + 1][0], atol=0, rtol=0)
                checked += 1
    return {"shards": 128, "exact_rows": checked, "device": str(weight.device)}


def forward_reference(model, inputs, cache, length, chunk_size=None):
    count = inputs.shape[1]
    if count < 1 or length < count or (chunk_size is not None and chunk_size < 1):
        raise ValueError("Reference forward requires nonempty inputs and valid lengths")
    chunk_size = count if chunk_size is None else chunk_size
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        absolute_start, absolute_stop = length - count + start, length - count + stop
        output = model(
            input_ids=inputs[:, start:stop],
            position_ids=torch.arange(absolute_start, absolute_stop, device=inputs.device).unsqueeze(0),
            attention_mask=torch.ones((1, absolute_stop), device=inputs.device, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
    return output


@torch.inference_mode()
def score_case(model, prompt, result, directory, case, prefill_chunk_size=None):
    tokens = result["token_ids"]
    native = result["response"]["tokens"]
    if [item["id"] for item in native] != tokens:
        raise ValueError("Native probability records do not match generated IDs")
    device = model.model.language_model.embed_tokens.weight.device
    inputs = torch.tensor([prompt], device=device, dtype=torch.long)
    cache = None
    rows, scores = [], []
    for step, target in enumerate(tokens):
        length = len(prompt) + step
        output = forward_reference(model, inputs, cache, length, prefill_chunk_size)
        cache = output.past_key_values
        logits = output.logits[0, -1].float().cpu()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError(f"Nonfinite reference logits at {case} step {step}")
        scores.append(logits)
        reference_logprob = float(logits[target] - logits.logsumexp(0))
        native_logprob = float(native[step]["logprob"])
        greedy = int(logits.argmax())
        rows.append(
            dict(
                step=step,
                token_id=target,
                reference_greedy_id=greedy,
                reference_logprob=reference_logprob,
                native_logprob=native_logprob,
                logprob_delta=native_logprob - reference_logprob,
                reference_greedy_gap=float(logits[greedy] - logits[target]),
            )
        )
        inputs = torch.tensor([[target]], device=device, dtype=torch.long)
    torch.save(torch.stack(scores), directory / f"{case}-scores.pt")
    return dict(
        steps=rows,
        greedy_matches=sum(row["token_id"] == row["reference_greedy_id"] for row in rows),
        reference_mean_nll=-sum(row["reference_logprob"] for row in rows) / len(rows),
        native_mean_nll=-sum(row["native_logprob"] for row in rows) / len(rows),
        mean_absolute_logprob_delta=sum(abs(row["logprob_delta"]) for row in rows) / len(rows),
        maximum_absolute_logprob_delta=max(abs(row["logprob_delta"]) for row in rows),
    )


def main():
    if not os.environ.get("EXP_ACTIVE") or os.environ.get("USE_HUB_KERNELS") != "NO":
        raise RuntimeError("Run through exp with USE_HUB_KERNELS=NO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device-plan", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["english", "chinese", "code_unicode"])
    parser.add_argument("--prefill-chunk-size", type=int)
    args = parser.parse_args()
    if args.prefill_chunk_size is not None and args.prefill_chunk_size < 1:
        parser.error("Prefill chunk size must be positive")
    root, run = args.model_directory, Path(os.environ["EXP_RUN_DIR"])
    verified_checkpoint(root)
    trace = json.loads(args.generation.read_text())
    cases = cases_for(root)
    if trace["revision"] != REVISION or trace["cases"] != cases or trace["service"]["framework"] != "lightllm":
        raise ValueError("Generation must contain native outputs for the pinned shared inputs")
    for case in args.cases:
        if not trace["results"][case]["token_ids"]:
            raise ValueError(f"Missing generated IDs: {case}")
    plan = json.loads(args.device_plan.read_text())
    device_map = plan["device_map"]
    if not device_map or not all(type(device) is int for device in device_map.values()):
        raise ValueError("This reference diagnostic requires a complete GPU placement")
    module = load_source(args.source)
    report = dict(
        status="loading",
        scope="identical native token histories; diagnostic only, no quality parity or performance claim",
        revision=REVISION,
        source_sha256=SOURCE_SHA256,
        source=str(args.source.resolve()),
        generation=str(args.generation.resolve()),
        device_plan=plan,
        command=sys.argv,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        software={name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
        hardware=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.free", "--format=csv,noheader"], text=True
        ),
        configuration=dict(
            dtype="bfloat16",
            attention="sdpa",
            experts="eager",
            use_hub_kernels=False,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            prefill_chunk_size=args.prefill_chunk_size,
        ),
        results={},
    )
    path = run / "reference-scores.json"
    path.write_text(json.dumps(report, indent=2))
    started = time.monotonic()
    try:
        model, loading = module.Qwen4ExpForConditionalGeneration.from_pretrained(
            str(root),
            dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="sdpa",
            experts_implementation="eager",
            output_loading_info=True,
        )
        model.eval()
        (run / "loading-info.json").write_text(json.dumps(loading, indent=2, default=str))
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError("Reference checkpoint loading is incomplete")
        if any(not key.startswith("mtp.") for key in loading.get("unexpected_keys", [])):
            raise ValueError("Unexpected non-MTP reference weights")
        report["lexical_row_verification"] = verify_embedding_rows(model, root)
        report.update(status="scoring", load_seconds=time.monotonic() - started)
        path.write_text(json.dumps(report, indent=2))
        for case in args.cases:
            result = score_case(model, cases[case], trace["results"][case], run, case, args.prefill_chunk_size)
            report["results"][case] = result
            path.write_text(json.dumps(report, indent=2))
            print(
                json.dumps({"case": case, **{key: value for key, value in result.items() if key != "steps"}}),
                flush=True,
            )
        report["status"] = "recorded"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
