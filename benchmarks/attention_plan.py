import json
import statistics
import time

import torch
import triton.testing

from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_kv_pair


def eager_time(run, repeats=200):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeats


def main():
    results = []
    for heads, dim in ((2, 128), (16, 256), (4, 512)):
        for tokens in (1, 64, 1024):
            kv = torch.randn(tokens, 2 * heads, dim, dtype=torch.bfloat16, device="cuda")
            ids = torch.randperm(4096, device="cuda", dtype=torch.int32)[:tokens]
            k = torch.empty(4096, heads, dim, dtype=kv.dtype, device="cuda")
            v = torch.empty_like(k)
            inputs_k, inputs_v = kv[:, :heads], kv[:, heads:]

            def separate():
                destindex_copy_kv(inputs_k, ids, k)
                destindex_copy_kv(inputs_v, ids, v)

            def paired():
                destindex_copy_kv_pair(kv, ids, k, v)

            separate()
            paired()
            torch.testing.assert_close(k[ids.long()], inputs_k, atol=0, rtol=0)
            torch.testing.assert_close(v[ids.long()], inputs_v, atol=0, rtol=0)
            timings = {name: {mode: [] for mode in ("eager_us", "graph_us")} for name in ("separate", "paired")}
            for trial in range(5):
                choices = [("separate", separate), ("paired", paired)]
                for name, run in choices[:: 1 if trial % 2 else -1]:
                    timings[name]["eager_us"].append(eager_time(run))
                    timings[name]["graph_us"].append(1000 * triton.testing.do_bench_cudagraph(run, rep=100))
            results.append(
                dict(
                    heads=heads,
                    dim=dim,
                    tokens=tokens,
                    **{
                        name: {mode: statistics.median(values) for mode, values in modes.items()}
                        for name, modes in timings.items()
                    },
                )
            )
    print(json.dumps(dict(device=torch.cuda.get_device_name(), torch=torch.__version__, results=results), indent=2))


if __name__ == "__main__":
    main()
