"""八卡真实 routed-MoE 对照：Graph 内包含路由、DeepEP、DeepGEMM，不包含 attention。

PYTHONPATH=. EP_DISABLE_GIN=1 python -m torch.distributed.run --standalone --nproc-per-node=8 \
  test/benchmark/kernel/benchmark_eplb_decode_rank_weighted_moe.py --output /tmp/weighted_moe.json

每源 rank 16 token，全局 128，4 套真实尺寸合成 FP8 权重轮转。各算法共享布局、输入和权重，
同一进程交替计时；不把预计算路由的 MoE 时间冒充完整路由路径，也不把单节点 NVLink 视为 RDMA。
"""

import argparse
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from benchmark_eplb_decode_compute import make_weights, make_forward, ThermalGuard
from eplb_decode_benchmark_utils import make_traffic, make_placement
from benchmark_eplb_decode_rank_weighted import validate_route_cpu


PHASES = ("routing", "input_quant", "dispatch_wait", "w1", "silu_quant", "w2", "local_reduce", "combine_wait")
VARIANTS = ("current", "weighted")


def check_gpu_availability(control_group):
    """排除其他进程或容器的显存占用；容器外 PID 可能不出现在 NVML 进程表中。"""
    import pynvml

    pynvml.nvmlInit()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    handle = pynvml.nvmlDeviceGetHandleByUUID(str(device.uuid))
    # 总显存与进程显存是两个独立 NVML 快照，刚释放 / 申请的内存可能更新不同步。
    # 可见外部 PID 立即拒绝；未知占用需短暂复核，避免把快照更新延迟当成外部任务。
    for attempt in range(3):
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        known_bytes = sum(process.usedGpuMemory for process in processes)
        unaccounted_mib = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - known_bytes) // (1024 ** 2)
        occupied = dict(
            rank=dist.get_rank(),
            other_pids=[process.pid for process in processes if process.pid != os.getpid()],
            unaccounted_mib=unaccounted_mib,
        )
        if occupied["other_pids"] or unaccounted_mib <= 1024:
            break
        if attempt < 2:
            time.sleep(0.1)
    snapshots = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(snapshots, occupied, group=control_group)
    conflicts = [snapshot for snapshot in snapshots if snapshot["other_pids"] or snapshot["unaccounted_mib"] > 1024]
    if conflicts:
        raise RuntimeError(f"GPU 被其他任务占用，无法进行独占性能对照；请释放资源后复测：{conflicts}")


def capture(forward, args, profiled):
    stamps = (
        [
            [torch.cuda.Event(enable_timing=True, external=True) for _ in range(len(PHASES) + 1)]
            for _ in range(args.graph_steps)
        ]
        if profiled
        else None
    )
    if stamps is not None:
        for row in stamps:
            for event in row:
                event.record()
        torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in range(args.graph_steps):
            output, _, _, _ = forward(bank=step % args.banks, stamps=stamps[step] if profiled else None)
    return graph, stamps, output


def prepare_variant(name, placement, logical, inputs, weights, buffer, args, communication_sms):
    from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
        eplb_hash_targets_topk_ids,
    )
    from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_rank_weighted import (
        eplb_rank_weighted_topk_ids,
    )
    from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
        build_logical_to_physical_map,
        build_logical_to_rank_physical_map,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    logical_to_rank_physical_map = torch.tensor(
        build_logical_to_rank_physical_map(placement, args.experts), dtype=torch.int32, device="cuda"
    )
    rows = torch.tensor(
        build_logical_to_physical_map(placement, args.experts, current_rank=rank, node_world_size=world),
        dtype=torch.int32,
        device="cuda",
    )
    selectors = []
    for ids in logical:
        common = dict(
            logical_topk_ids=ids, logical_to_rank_physical_map=logical_to_rank_physical_map, current_rank=rank
        )
        if name == "weighted":
            selector = partial(eplb_rank_weighted_topk_ids, **common, node_world_size=world)
        else:
            assert name == "current"
            common.update(logical_to_physical_map=rows, num_physical_experts_per_rank=len(placement[0]))
            selector = partial(eplb_hash_targets_topk_ids, **common, node_world_size=world)
        selectors.append(selector)
    routes = [selector() for selector in selectors]
    for bank, route in enumerate(routes):
        validate_route_cpu(logical[bank].cpu().numpy(), route.cpu().numpy(), placement, rank, world, name)
    # 复用已验证的 GEMM/通信流水线，只在它之前加入路由，不复制低层计算实现。
    base_forward = make_forward(buffer, inputs, routes, weights, args, communication_sms)

    def forward(bank, stamps=None):
        if stamps is not None:
            stamps[0].record()
        routes[bank] = selectors[bank]()
        return base_forward(bank=bank, stamps=None if stamps is None else stamps[1:])

    return forward, routes


def run_group(workload, redundant, buffer, args, control_group):
    import deep_gemm

    rank, world = dist.get_rank(), dist.get_world_size()
    placement = make_placement(args.experts, world, redundant, workload, args.topk)
    weights = make_weights(placement[rank], args.hidden, args.intermediate, args.banks)
    reserved_sms = buffer.get_theoretical_num_sms(len(placement[0]) * world, args.topk)
    deep_gemm.set_num_sms(torch.cuda.get_device_properties(rank).multi_processor_count - reserved_sms)
    communication_sms = torch.cuda.get_device_properties(rank).multi_processor_count * 3 // 5 // 2 * 2
    local_tokens = args.global_tokens // world
    logical, inputs = [], []
    for bank in range(args.banks):
        # 性能验证使用未参加系数筛选的种子；每个源 rank 的行号依旧从 0 开始。
        ids = make_traffic(args.global_tokens, args.experts, args.topk, workload, args.evaluation_seed + bank)
        logical.append(
            ids[rank * local_tokens : (rank + 1) * local_tokens].contiguous().to(device="cuda", dtype=torch.int64)
        )
        generator = torch.Generator(device="cuda").manual_seed(100 + rank + bank * world)
        inputs.append(
            torch.randn((local_tokens, args.hidden), dtype=torch.bfloat16, device="cuda", generator=generator)
        )
    variants, reference_outputs = {}, []
    for name in args.variants:
        forward, routes = prepare_variant(name, placement, logical, inputs, weights, buffer, args, communication_sms)
        snapshots = []
        for bank in range(args.banks):
            output, handle, _, _ = forward(bank=bank)
            torch.cuda.synchronize()
            assert bool(output.isfinite().all())
            if name == "current":
                reference_outputs.append(output.clone())
            else:
                torch.testing.assert_close(output, reference_outputs[bank], rtol=0.02, atol=0.02)
            snapshots.append(
                dict(
                    counts=handle.num_unaligned_recv_tokens_per_expert.cpu().tolist(),
                    physical_ids=routes[bank].cpu().tolist(),
                )
            )
        plain, _, plain_output = capture(forward, args, profiled=False)
        profiled, stamps, profiled_output = capture(forward, args, profiled=True)
        for _ in range(3):
            plain.replay()
            profiled.replay()
        torch.cuda.synchronize()
        assert bool(plain_output.isfinite().all()) and bool(profiled_output.isfinite().all())
        variants[name] = dict(
            plain=plain,
            profiled=profiled,
            stamps=stamps,
            snapshots=snapshots,
            # 保留闭包/输出，避免 Graph 输入地址被其他变体复用。
            forward=forward,
            outputs=(plain_output, profiled_output),
            plain_us=[],
            profiled_us=[],
            phases_us=[],
            thermal=[],
        )
        if rank == 0:
            print(json.dumps(dict(captured=name, workload=workload, R=redundant)), flush=True)

    guard = ThermalGuard(rank, control_group, args.cool_temperature)
    for round_index in range(args.rounds):
        # 外部任务也可能在评测中途启动；检查放在计时区间外。
        check_gpu_availability(control_group)
        order = list(args.variants) if round_index % 2 == 0 else list(reversed(args.variants))
        for name in order:
            item = variants[name]
            for profiled in (False, True) if round_index % 2 == 0 else (True, False):
                samples = []
                graph_times = []
                for chunk in range(args.steps // args.graph_steps):
                    guard.cool()
                    # 先启动采样线程，再同步各 rank，避免线程启动延迟进入其他 rank 的通信等待。
                    guard.begin()
                    dist.barrier(device_ids=[rank])
                    torch.cuda.synchronize()
                    if profiled:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        item["profiled"].replay()
                        end.record()
                        end.synchronize()
                        graph_times.append(start.elapsed_time(end) * 1000 / args.graph_steps)
                        samples.extend(
                            [
                                [row[index].elapsed_time(row[index + 1]) * 1000 for index in range(len(PHASES))]
                                for row in item["stamps"]
                            ]
                        )
                    else:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        item["plain"].replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000 / args.graph_steps)
                    item["thermal"].append(dict(round=round_index, profiled=profiled, chunk=chunk, **guard.end()))
                item["phases_us" if profiled else "plain_us"].append(samples if profiled else float(np.mean(samples)))
                if profiled:
                    item["profiled_us"].append(float(np.mean(graph_times)))
        if rank == 0:
            print(
                json.dumps(dict(progress=workload, R=redundant, round=round_index + 1, rounds=args.rounds)), flush=True
            )

    local = {
        name: {key: item[key] for key in ("plain_us", "profiled_us", "phases_us", "snapshots", "thermal")}
        for name, item in variants.items()
    }
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local, group=control_group)
    results = []
    if rank == 0:
        for name in args.variants:
            phases = np.asarray([data[name]["phases_us"] for data in gathered])  # [rank, round, step, phase]
            plain = np.asarray([data[name]["plain_us"] for data in gathered])
            profiled = np.asarray([data[name]["profiled_us"] for data in gathered])
            counts = np.asarray([[snapshot["counts"] for snapshot in data[name]["snapshots"]] for data in gathered])
            for bank in range(args.banks):
                physical = [
                    expert
                    for data in gathered
                    for token in data[name]["snapshots"][bank]["physical_ids"]
                    for expert in token
                ]
                expected = np.bincount(physical, minlength=world * counts.shape[-1]).reshape(world, -1)
                assert np.array_equal(expected, counts[:, bank]) and expected.sum() == args.global_tokens * args.topk
            padded = ((counts + 127) // 128 * 128).sum(axis=2)
            total = phases.sum(axis=-1)
            # 使用同一个最慢 rank 的完整阶段向量，不能把各阶段独立最大值相加来算占比。
            critical_rank = total.argmax(axis=0)
            critical_phases = np.take_along_axis(phases, critical_rank[None, :, :, None], axis=0)[0]
            critical_total = critical_phases.sum(axis=-1)
            remote = sum(
                len({expert // len(placement[0]) for expert in token} - {source})
                for source, data in enumerate(gathered)
                for bank in data[name]["snapshots"]
                for token in bank["physical_ids"]
            )
            result = dict(
                workload=workload,
                R=redundant,
                variant=name,
                plain_moe_us=float(np.median(plain.max(axis=0))),
                plain_samples_rank_round_us=plain.tolist(),
                profiled_moe_us=float(np.median(profiled.max(axis=0))),
                profiled_samples_rank_round_us=profiled.tolist(),
                # 单步阶段总跨度不含相邻 step 之间的 event 成本，不能拿它代替整张图计时。
                profiled_step_span_us=float(np.median(total.mean(axis=2).max(axis=0))),
                # 各阶段独立 rank 最大值不可相加；占比从同一个关键 rank 的向量计算。
                phases_critical_us={
                    phase: float(np.median(phases[..., index].max(axis=0))) for index, phase in enumerate(PHASES)
                },
                critical_compute_share_pct=float(
                    np.median(100 * critical_phases[..., 3:6].sum(axis=-1) / critical_total)
                ),
                critical_communication_share_pct=float(
                    np.median(100 * critical_phases[..., [2, 7]].sum(axis=-1) / critical_total)
                ),
                critical_phase_share_pct={
                    phase: float(np.median(100 * critical_phases[..., index] / critical_total))
                    for index, phase in enumerate(PHASES)
                },
                mean_remote_ranks=remote / (args.global_tokens * args.banks),
                max_rank_rows_over_mean=float(
                    np.median(counts.sum(axis=2).max(axis=0) / counts.sum(axis=2).mean(axis=0))
                ),
                max_rank_padding_over_mean=float(np.median(padded.max(axis=0) / padded.mean(axis=0))),
                counts_rank_bank_expert=counts.tolist(),
                thermal_slowdown_observed=any(
                    sample["thermal_slowdown"] for data in gathered for sample in data[name]["thermal"]
                ),
                max_temperature=max(sample["max_temperature"] for data in gathered for sample in data[name]["thermal"]),
            )
            result["profiled_overhead_pct"] = 100 * (result["profiled_moe_us"] / result["plain_moe_us"] - 1)
            results.append(result)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in ("counts_rank_bank_expert", "plain_samples_rank_round_us")
                    }
                ),
                flush=True,
            )
    dist.barrier(device_ids=[rank])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redundant", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--workloads", nargs="+", default=["uniform", "cooccurrence"])
    parser.add_argument("--cases", nargs="+", help="只评测指定组合，例如 uniform_R64 cooccurrence_R16")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--global-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=272)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--graph-steps", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--evaluation-seed", type=int, default=2031)
    parser.add_argument("--cool-temperature", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.variants[0] == "current" and set(args.variants) <= set(VARIANTS)
    assert len(args.variants) == len(set(args.variants))
    assert args.steps > 0 and args.graph_steps > 0 and args.banks > 0 and args.rounds > 0
    assert args.hidden % 128 == args.intermediate % 128 == 0 and args.capacity % 8 == 0
    assert not args.cases or set(args.cases) <= {
        f"{workload}_R{redundant}" for workload in args.workloads for redundant in args.redundant
    }
    assert args.steps % args.graph_steps == 0 and args.graph_steps % args.banks == 0
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=600))
    world = dist.get_world_size()
    assert world == 8 and args.experts == 256 and args.topk == 8 and args.global_tokens % world == 0
    assert args.global_tokens > 0 and args.global_tokens // world <= args.capacity
    import deep_ep
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import (
        set_mk_alignment_for_contiguous_layout,
    )

    # NCCL communicator 是惰性初始化的；先执行 collective，DeepEP 才能取得有效句柄。
    dist.barrier(device_ids=[rank])
    torch.cuda.synchronize()
    check_gpu_availability(control_group)
    buffer = deep_ep.ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=args.capacity,
        hidden=args.hidden,
        num_topk=args.topk,
        use_fp8_dispatch=True,
        allow_multiple_reduction=True,
        prefer_overlap_with_compute=True,
        explicitly_destroy=True,
    )
    set_mk_alignment_for_contiguous_layout(128)
    result = dict(
        metadata=dict(
            gpu=torch.cuda.get_device_name(),
            global_tokens=args.global_tokens,
            tokens_per_source=args.global_tokens // world,
            hidden=args.hidden,
            intermediate=args.intermediate,
            banks=args.banks,
            steps=args.steps,
            graph_steps=args.graph_steps,
            rounds=args.rounds,
            torch=torch.__version__,
            alignment=128,
            cool_temperature=args.cool_temperature,
            evaluation_seed=args.evaluation_seed,
            scope="single-node routed MoE including routing; synthetic weights; not whole-model or RDMA",
        ),
        cases=[],
    )
    try:
        for workload in args.workloads:
            for redundant in args.redundant:
                if args.cases and f"{workload}_R{redundant}" not in args.cases:
                    continue
                result["cases"].extend(run_group(workload, redundant, buffer, args, control_group))
                if rank == 0:
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        dist.barrier(device_ids=[rank])
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
