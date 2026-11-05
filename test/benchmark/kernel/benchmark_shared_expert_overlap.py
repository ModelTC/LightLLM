"""A/B benchmark of the real DeepEP decode/shared-expert call chain on Hopper.

Run from the checkout with its pinned DeepEP/DeepGEMM dependencies and RDMA access::

    PYTHONPATH=. NVSHMEM_QP_DEPTH=1024 python test/benchmark/kernel/benchmark_shared_expert_overlap.py \
        --world-size 8 --output /mnt/vqa/wanzihao/shared-expert-overlap/result.json

Fixed routes and synthetic FP8 weights isolate a single FFN layer. Defaults use
DSV4 Flash dimensions (4096 hidden, 2048 intermediate, 256 experts over 8 ranks,
top-6). Width 4 tests MTP-expanded row counts, not the full speculative pipeline.
Each mode is checked in eager and CUDA Graph replay, including changed inputs and
routes. Timings exclude routing and report the slowest rank, with alternating A/B
order. This is not a complete model accuracy or end-to-end service benchmark.

Keep NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE at the service default of 256;
request count times width must fit that capacity. Mode 0 computes the shared expert
before routed experts. Mode 1 is the default overlap schedule.
"""
import argparse
import json
import os
import socket
import statistics
from datetime import timedelta
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, args, port):
    torch.cuda.set_device(rank)
    from lightllm.utils import dist_utils as du

    du.set_global_rank(rank)
    du.set_global_world_size(args.world_size)
    du.set_dp_size(args.world_size)
    du.set_dp_world_size(1)
    du.set_current_rank_in_dp(0)
    du.set_current_device_id(rank)
    du.set_current_rank_in_node(rank)
    du.set_node_world_size(args.world_size)
    from lightllm.server.core.objs.start_args_type import StartArgs

    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(
        asdict(
            StartArgs(
                model_dir=args.config_dir,
                enable_ep_moe=True,
                enable_return_routed_experts=False,
                enable_tpsp_mix_mode=False,
                batch_max_tokens=8192,
                mtp_mode="dspark",
                mtp_step=3,
            )
        )
    )
    os.environ["NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE"] = str(args.max_rows)
    os.environ["LIGHTLLM_USE_UE8M0_SCALES"] = "1"
    from lightllm.utils.envs_utils import get_env_start_args

    get_env_start_args.cache_clear()
    import deep_ep
    from lightllm.distributed import dist_group_manager
    import lightllm.common.basemodel
    from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod
    from lightllm.common.quantization.quantize_method import WeightPack
    from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_quant_kernel import weight_quant
    from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM
    from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer

    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=args.world_size,
        timeout=timedelta(seconds=120),
    )
    group = dist.new_group(backend="gloo")
    dist.barrier(device_ids=[rank])
    experts = args.experts_per_rank * args.world_size
    print(rank, "creating DeepEP ElasticBuffer", args.max_rows, args.hidden, flush=True)
    buffer = deep_ep.ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=args.max_rows,
        hidden=args.hidden,
        num_topk=args.topk,
        use_fp8_dispatch=True,
        allow_multiple_reduction=True,
        explicitly_destroy=True,
    )
    dist_group_manager.ep_buffer = buffer
    print(rank, "DeepEP ready", flush=True)
    try:
        torch.manual_seed(1013 + rank)
        quant = DeepGEMMFP8w8a8B128QuantizationMethod()
        quant.use_ue8m0_scales = True
        quant.use_packed_ue8m0 = False

        def pack(shape):
            value = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
            q, s = weight_quant(value, 128, use_ue8m0_scales=True)
            return WeightPack(weight=q, weight_scale=s)

        owner = FusedMoeWeight.__new__(FusedMoeWeight)
        owner.layer_num_ = 0
        owner.w13 = pack((args.experts_per_rank, 2 * args.intermediate, args.hidden))
        owner.w2 = pack((args.experts_per_rank, args.hidden, args.intermediate))
        owner.fuse_moe_impl = FuseMoeDeepGEMM(experts, 0, 1.0, quant, layer_index=0)

        def shared_weight(shape):
            w = MMWeightTpl.__new__(MMWeightTpl)
            w.quant_method = quant
            w.mm_param = pack(shape)
            w.bias = None
            return w

        weights = NS(
            experts=owner,
            gate_up_proj=shared_weight((2 * args.intermediate, args.hidden)),
            down_proj=shared_weight((args.hidden, args.intermediate)),
        )
        config = dict(
            rms_norm_eps=1e-6,
            hidden_size=args.hidden,
            num_attention_heads=32,
            head_dim=512,
            qk_rope_head_dim=64,
            o_groups=8,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            compress_ratios=[0],
            num_hash_layers=0,
            vocab_size=129280,
            n_layer=1,
            num_experts_per_tok=args.topk,
            routed_scaling_factor=1.0,
            swiglu_limit=args.limit,
            index_head_dim=128,
            index_n_heads=32,
            index_topk=512,
        )
        layer = DeepseekV4TransformerLayerInfer(0, config)
        original_shared = layer._ffn_shared_expert
        stats = []
        for requests in args.requests:
            for width in args.widths:
                rows = requests * width
                x = torch.randn(rows, args.hidden, device="cuda", dtype=torch.bfloat16) * 0.1
                ids = torch.rand(rows, experts, device="cuda").topk(args.topk, dim=-1).indices
                probs = torch.randn(rows, args.topk, device="cuda").softmax(-1)
                infer = NS(is_prefill=False, microbatch_index=0)
                counts = {0: 0, 1: 0}
                layer._select_experts = lambda x, infer, w: (probs, ids)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())

                def shared(*a, mode, **kw):
                    assert torch.cuda.current_stream().cuda_stream == stream.cuda_stream
                    counts[mode] += 1
                    return original_shared(*a, **kw)

                def run_baseline(mode=0):
                    layer._ffn_shared_expert = lambda *a, **kw: shared(*a, mode=mode, **kw)
                    shared_out = layer._ffn_shared_expert(x, infer, weights)
                    routed_out = weights.experts.experts_with_topk(
                        input_tensor=x,
                        topk_weights=probs,
                        topk_ids=ids,
                        is_prefill=False,
                        infer_state=infer,
                        alpha=1.0,
                        limit=args.limit,
                        clamp_up_add_one=False,
                    )
                    return routed_out + shared_out

                def run_overlap(mode=1):
                    layer._ffn_shared_expert = lambda *a, **kw: shared(*a, mode=mode, **kw)
                    return layer._ffn(x, infer, weights)

                funcs = {0: run_baseline, 1: run_overlap}
                print(rank, "case", requests, width, "warming", flush=True)
                with torch.cuda.stream(stream):
                    before = x.clone()
                    baseline = funcs[0]()
                    actual = funcs[1]()
                    stream.synchronize()
                    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
                    torch.testing.assert_close(x, before, rtol=0, atol=0)
                    assert counts == {0: 1, 1: 1}
                    for _ in range(args.warmup):
                        funcs[0]()
                        funcs[1]()
                    stream.synchronize()
                    dist.barrier(group=group)
                    graphs = {}
                    outputs = {}
                    for mode in (0, 1):
                        graph = torch.cuda.CUDAGraph()
                        prev = counts[mode]
                        with torch.cuda.graph(graph, stream=stream):
                            outputs[mode] = funcs[mode]()
                        assert counts[mode] == prev + 1
                        graphs[mode] = graph
                        graph.replay()
                        stream.synchronize()
                        dist.barrier(group=group)
                    # A replay must consume changed inputs and routing, without running Python callbacks again.
                    x.mul_(0.875)
                    ids.add_(1).remainder_(experts)
                    counts_before = counts.copy()
                    graphs[0].replay()
                    graphs[1].replay()
                    stream.synchronize()
                    assert counts == counts_before
                    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                    eager_changed = funcs[0]()
                    stream.synchronize()
                    torch.testing.assert_close(outputs[1], eager_changed, rtol=0, atol=0)
                    assert torch.isfinite(outputs[1]).all()

                    def measure(fn):
                        dist.barrier(group=group)
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record(stream)
                        for _ in range(args.iterations):
                            fn()
                        b.record(stream)
                        b.synchronize()
                        value = torch.tensor(a.elapsed_time(b) * 1000 / args.iterations, device="cuda")
                        dist.all_reduce(value, op=dist.ReduceOp.MAX)
                        return value.item()

                    values = {"eager": [], "graph": []}
                    for kind, call in [("eager", funcs), ("graph", {mode: graphs[mode].replay for mode in (0, 1)})]:
                        for order in [(0, 1), (1, 0), (0, 1), (1, 0)]:
                            values[kind].append({str(mode): measure(call[mode]) for mode in order})
                    result = {
                        "requests_per_rank": requests,
                        "rows_per_request": width,
                        "token_rows_per_rank": rows,
                        "exact_output_match": True,
                        "input_unchanged": True,
                        "eager_us": {
                            str(mode): statistics.median(r[str(mode)] for r in values["eager"]) for mode in (0, 1)
                        },
                        "graph_us": {
                            str(mode): statistics.median(r[str(mode)] for r in values["graph"]) for mode in (0, 1)
                        },
                        "samples": values,
                    }
                    stats.append(result)
                    if rank == 0:
                        print(json.dumps(result), flush=True)
                    for graph in graphs.values():
                        graph.reset()
                torch.cuda.synchronize()
                dist.barrier(group=group)
        if rank == 0:
            output = {
                "world_size": args.world_size,
                "gpu": torch.cuda.get_device_name(rank),
                "torch": torch.__version__,
                "deep_ep_module": deep_ep.__file__,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "experts": experts,
                "topk": args.topk,
                "scope": (
                    "real single-layer routed/shared FFN; fixed routes, synthetic FP8 weights; "
                    "no complete model or server"
                ),
                "cases": stats,
            }
            Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--requests", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts-per-rank", type=int, default=32)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.max_rows = int(os.getenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", "256"))
    assert max(args.requests) * max(args.widths) <= args.max_rows
    config_dir = Path(args.output).resolve().parent / "shared-expert-benchmark-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    args.config_dir = str(config_dir)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=args.world_size, join=True)
