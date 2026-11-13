"""Exercise score tiles and exact TopK using the supplied checkpoint's Indexer."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch

from benchmark import load_baseline, unpack_scores
from lightllm.models.sensenova_loop.dsa import project_indexer, yarn_cos_sin
from lightllm.models.sensenova_loop.quick_rope import apply_partial_mrope
from lightllm.models.sensenova_loop.triton_kernel.dsa_perf import ranked_scores
from lightllm.models.sensenova_loop.triton_kernel.dsa_select import exact_float_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.3.self_attn.indexer."
    weights = {}
    for suffix in ["wq.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias"]:
        name = prefix + suffix
        with safe_open(args.model_dir / index[name], framework="pt", device="cuda:0") as handle:
            weights[suffix] = handle.get_tensor(name)
    torch.manual_seed(519)
    keys = 16384
    hidden_k = torch.randn(keys, 4096, device="cuda", dtype=torch.bfloat16)
    ik = torch.nn.functional.layer_norm(
        torch.nn.functional.linear(hidden_k, weights["wk.weight"]),
        (128,),
        weights["k_norm.weight"],
        weights["k_norm.bias"],
        1e-6,
    )
    del hidden_k
    cos, sin = yarn_cos_sin(config["text_config"], keys, "cuda")
    apply_partial_mrope(None, ik[:, None], cos, sin, 64)
    kv = torch.zeros(keys, 2, 384, device="cuda", dtype=torch.bfloat16)
    kv[:, 0, 256:] = ik
    table = torch.stack((torch.arange(keys, device="cuda"), torch.arange(keys - 1, -1, -1, device="cuda"))).int()
    baseline = load_baseline(args.baseline_code)
    results = []
    for rows, mixed in [(1, False), (5, True), (17, True), (129, False), (512, False)]:
        hidden = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
        iq, _, iw = project_indexer(
            hidden,
            weights["wq.weight"],
            weights["wk.weight"],
            weights["weights_proj.weight"],
            weights["k_norm.weight"],
            weights["k_norm.bias"],
            8,
        )
        dummy = torch.zeros(rows, 1, 128, device="cuda", dtype=torch.bfloat16)
        apply_partial_mrope(iq, dummy, cos[-rows:], sin[-rows:], 64)
        req = (torch.arange(rows, device="cuda") % (2 if mixed else 1)).int()
        pos = torch.full((rows,), keys - 1, device="cuda", dtype=torch.int32)
        if mixed:
            pos[::4] = -1
            pos[1::4] = 2049
        lengths = (pos + 1).clamp_min(0)
        packed = baseline.ranked_scores(iq, iw, kv, table, req, pos, keys, 256)
        reference = baseline.packed_topk(packed, 2048)
        ref_scores = unpack_scores(packed)
        for tile in [4, 8, 16]:
            scores = ranked_scores(iq, iw, ik[:, None], table, req, pos, keys, 0, packed=False, query_tile=tile)
            torch.testing.assert_close(scores, ref_scores, rtol=0, atol=0)
            for backend in ["torch", "radix", "auto"]:
                ids = exact_float_topk(scores, lengths, 2048, backend=backend)
                torch.testing.assert_close(ids, reference, rtol=0, atol=0)
            item = {"rows": rows, "mixed_requests": mixed, "tile": tile, "scores_exact": True, "ids_exact": True}
            results.append(item)
            print(json.dumps(item), flush=True)
    args.output.write_text(
        json.dumps({"model_dir": str(args.model_dir), "cases": results, "complete": True}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
