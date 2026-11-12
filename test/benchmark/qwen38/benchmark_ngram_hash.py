import argparse
import json
import os

import torch
import triton
from triton.testing import do_bench_cudagraph

from lightllm.common.ngram_embedding import NGramHasher


def eager(hasher, tokens, history):
    batch, length = tokens.shape
    full = torch.cat((history, tokens.long()), dim=-1)
    positions = torch.arange(full.shape[1], device=full.device, dtype=torch.int64)
    eos_positions = torch.where(full == hasher.eos_token_id, positions, -1)
    inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat((eos_positions.new_full((batch, 1), -1), inclusive[:, :-1]), dim=1)
    relative = positions.unsqueeze(0) - previous_eos - 1
    shifted = [full]
    for shift in range(1, hasher.context_len + 1):
        source = positions - shift
        gathered = full.gather(1, source.clamp_min(0).unsqueeze(0).expand(batch, -1))
        valid = (relative >= shift) & (source.unsqueeze(0) >= 0)
        shifted.append(torch.where(valid, gathered, hasher.eos_token_id))
    ids = []
    per_order = hasher.heads // hasher.context_len
    for order in range(2, hasher.context_len + 2):
        mixed = shifted[0] * hasher.multipliers[0]
        for index in range(1, order):
            mixed = torch.bitwise_xor(mixed, shifted[index] * hasher.multipliers[index])
        begin = (order - 2) * per_order
        sizes = hasher.head_vocab_sizes[begin : begin + per_order]
        offsets = hasher.head_offsets[begin : begin + per_order]
        ids.append(torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1)) + offsets.view(1, 1, -1))
    return torch.cat(ids, dim=-1)[:, -length:].reshape(-1, hasher.heads), full[:, -hasher.context_len :].contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["1x1", "16x1", "128x1", "1x4096", "16x256"])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Wrap this benchmark with exp -m "purpose"')
    torch.manual_seed(37)

    sizes = torch.tensor(
        [
            20000003,
            20000023,
            20000033,
            20000047,
            20000059,
            20000063,
            20000069,
            20000077,
            20000081,
            20000093,
            20000107,
            20000147,
            20000153,
            20000159,
            20000161,
            20000171,
        ],
        device="cuda",
        dtype=torch.int64,
    )
    offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
    multipliers = torch.tensor([23703573157769, 20109073645365, 8052911324071], device="cuda", dtype=torch.int64)
    hasher = NGramHasher(multipliers, sizes, offsets, 248044)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "ngram_size": 3,
                "heads_per_ngram": 8,
                "multipliers": multipliers.tolist(),
                "head_vocab_sizes": sizes.tolist(),
                "eos_token_id": hasher.eos_token_id,
                "timing": "CUDA graph, median of interleaved repetitions",
                "includes_embedding_lookup": False,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        ),
        flush=True,
    )
    for shape in args.shapes:
        batch, length = map(int, shape.split("x"))
        tokens = torch.randint(0, 248320, (batch, length), dtype=torch.int64, device="cuda")
        tokens[:, ::31] = hasher.eos_token_id
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
        history = hasher.empty_history(batch)
        history[:, -1] = 77

        def fused():
            return hasher.hash(tokens.view(-1), starts, history)

        def reference():
            return eager(hasher, tokens, history)

        for actual, expected in zip(fused(), reference()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        samples = {"eager": [], "fused": []}
        for repeat in range(args.repetitions):
            functions = [("eager", reference), ("fused", fused)]
            for name, func in functions[:: 1 if repeat % 2 == 0 else -1]:
                samples[name].append(do_bench_cudagraph(func, rep=100) * 1000)
        medians = {name + "_us": sorted(values)[len(values) // 2] for name, values in samples.items()}
        print(
            json.dumps(
                {
                    "batch": batch,
                    "tokens_per_request": length,
                    **medians,
                    "speedup": medians["eager_us"] / medians["fused_us"],
                    "samples_us": samples,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
