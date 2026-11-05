"""八卡 decode routed-MoE 计算分布评测，不包含 attention/shared expert/router。

PYTHONPATH=. EP_DISABLE_GIN=1 torchrun --standalone --nproc-per-node=8 \
  test/benchmark/kernel/benchmark_eplb_decode_compute.py --output /tmp/decode_compute.json

默认全局 128 token（每源 rank 16），DeepSeek-R1 形状，FP8 专家权重，
128-row alignment。四套独立权重轮转，避免单层热点权重常驻 L2 的假收益。
无 instrumentation 的 CUDA Graph 测 MoE 跨度；另一个相同的 graph 用
外部 CUDA Event 分段计时。计算占比的分母是后者，不混用两种计时口径。
"""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torch.distributed as dist


PHASES = ("input_quant", "dispatch_wait", "w1", "silu_quant", "w2", "local_reduce", "combine_wait")


class ThermalGuard:
    """计时外被动冷却，计时内只读采样；不改变 GPU 时钟或功耗设置。"""

    def __init__(self, rank, control_group, temperature_limit):
        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        # 用 UUID 匹配 CUDA device，兼容 CUDA_VISIBLE_DEVICES 的重排。
        device = torch.cuda.get_device_properties(rank)
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(device.uuid))
        self.control_group = control_group
        self.temperature_limit = temperature_limit
        self.samples = []
        self.stop = threading.Event()

    def sample(self):
        nvml = self.nvml
        thermal_bits = nvml.nvmlClocksThrottleReasonSwThermalSlowdown | nvml.nvmlClocksThrottleReasonHwThermalSlowdown
        sample = dict(
            temperature=nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU),
            sm_clock=nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM),
            thermal_slowdown=bool(nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle) & thermal_bits),
        )
        return sample

    def cool(self):
        if not self.temperature_limit:
            return
        started = time.monotonic()
        while True:
            warm = torch.tensor([int(self.sample()["temperature"] > self.temperature_limit)])
            dist.all_reduce(warm, op=dist.ReduceOp.MAX, group=self.control_group)
            if not warm.item():
                return
            if time.monotonic() - started > 120:
                raise RuntimeError("GPU 未能被动冷却到阈值；停止评测，避免将热降频误判为算法差异")
            time.sleep(0.2)

    def _poll(self):
        while not self.stop.wait(0.01):
            self.samples.append(self.sample())

    def begin(self):
        self.samples = [self.sample()]
        self.stop.clear()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def end(self):
        self.stop.set()
        self.thread.join()
        self.samples.append(self.sample())
        return dict(
            max_temperature=max(sample["temperature"] for sample in self.samples),
            thermal_slowdown=any(sample["thermal_slowdown"] for sample in self.samples),
            samples=self.samples,
        )


def case_specs():
    specs = [
        dict(name=f"spread_{groups}", pattern="spread", groups=groups, R=0, routing="none")
        for groups in (1, 2, 4, 8, 16, 32)
    ]
    for pattern in ("pair_compact", "pair_spread8"):
        specs.extend(
            dict(name=f"{pattern}_{left}_{8-left}", pattern=pattern, left=left, R=0, routing="none")
            for left in (4, 6, 7)
        )
    specs.append(dict(name="one_target", pattern="one_target", R=0, routing="none"))
    specs.append(dict(name="uniform", pattern="uniform", R=0, routing="none"))
    specs.append(dict(name="cooccurrence_R0", pattern="cooccurrence", R=0, routing="none"))
    for redundant in (1, 2, 4, 16, 64):
        specs.append(
            dict(name=f"cooccurrence_R{redundant}_current", pattern="cooccurrence", R=redundant, routing="current")
        )
    return specs


def build_logical_routes(spec, global_tokens, bank, experts=256, world_size=8, topk=8):
    """构造唯一的 logical top-k；受控组保持全局有效计算量不变。"""
    token = torch.arange(global_tokens, dtype=torch.int64)
    slots = torch.arange(topk, dtype=torch.int64)
    primary = experts // world_size
    pattern = spec["pattern"]
    if pattern == "spread":
        groups = spec["groups"]
        expert = token[:, None] % groups + (bank * groups) % primary
        return (slots[None, :] * primary + expert).to(torch.int32)
    if pattern.startswith("pair_"):
        left = spec["left"]
        pair = token % (world_size // 2)
        rank = pair[:, None] * 2 + (slots[None, :] >= left).to(torch.int64)
        local_slot = torch.where(slots < left, slots, slots - left)[None, :]
        if pattern == "pair_spread8":
            local_slot = (local_slot + token[:, None] // (world_size // 2)) % 8
        local_slot = local_slot + (bank * 8) % primary
        return (rank * primary + local_slot).to(torch.int32)
    if pattern == "one_target":
        return ((token[:, None] % world_size) * primary + slots[None, :] + (bank * 8) % primary).to(torch.int32)
    from eplb_decode_benchmark_utils import make_traffic

    return make_traffic(tokens=global_tokens, experts=experts, topk=topk, workload=pattern, seed=2026 + bank)


def build_placement(spec, experts, world_size, topk):
    from eplb_decode_benchmark_utils import make_placement

    return make_placement(
        experts=experts,
        world_size=world_size,
        redundant=spec["R"],
        workload="cooccurrence" if spec["pattern"] == "cooccurrence" else "uniform",
        topk=topk,
    )


def make_weights(logical_experts, hidden, intermediate, banks):
    """真实尺寸的 FP8 权重；同 bank、同 logical ID 的副本内容一致。

    使用可精确表示的整数 FP8 数据和 power-of-two block scales；只比较
    执行成本，不将这些合成权重冒充实际模型权重或任务精度评测。
    """
    packs = []
    for bank in range(banks):
        w1 = torch.empty((len(logical_experts), 2 * intermediate, hidden), dtype=torch.float8_e4m3fn, device="cuda")
        w2 = torch.empty((len(logical_experts), hidden, intermediate), dtype=torch.float8_e4m3fn, device="cuda")
        generator = torch.Generator(device="cuda")
        for local_index, logical_expert in enumerate(logical_experts):
            generator.manual_seed(10000 * bank + logical_expert + 42)
            w1[local_index].copy_(
                torch.randint(
                    -16, 17, (2 * intermediate, hidden), dtype=torch.int8, device="cuda", generator=generator
                ).to(torch.float8_e4m3fn)
            )
            w2[local_index].copy_(
                torch.randint(-16, 17, (hidden, intermediate), dtype=torch.int8, device="cuda", generator=generator).to(
                    torch.float8_e4m3fn
                )
            )
        scale1 = 2 ** round(math.log2(1 / (10 * math.sqrt(hidden))))
        scale2 = 2 ** round(math.log2(1 / (10 * math.sqrt(intermediate))))
        s1 = torch.full(
            (len(logical_experts), 2 * intermediate // 128, hidden // 128), scale1, dtype=torch.float32, device="cuda"
        )
        s2 = torch.full(
            (len(logical_experts), hidden // 128, intermediate // 128), scale2, dtype=torch.float32, device="cuda"
        )
        packs.append((w1, s1, w2, s2))
    return packs


def make_forward(buffer, inputs, routes, weights, args, communication_sms):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep.deepgemm import (
        deepgemm_grouped_fp8_nt_contiguous,
    )
    from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul_mix_quant_ep import (
        silu_and_mul_psum_post_quant_fwd,
    )
    from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import ep_reduce_decode_output
    from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8

    topk_weights = torch.full(routes[0].shape, 1 / args.topk, dtype=torch.float32, device="cuda")
    physical_experts = len(weights[0][0]) * dist.get_world_size()
    expected_m = math.ceil(args.global_tokens * args.topk / args.experts)

    def forward(bank, stamps=None):
        def record(index):
            if stamps is not None:
                stamps[index].record()

        record(0)
        quantized = per_token_group_quant_fp8(inputs[bank], group_size=128, dtype=torch.float8_e4m3fn)
        record(1)
        recv_x, _, recv_weights, handle, event = buffer.dispatch(
            quantized,
            topk_idx=routes[bank],
            topk_weights=topk_weights,
            num_experts=physical_experts,
            num_max_tokens_per_rank=args.capacity,
            expert_alignment=128,
            num_sms=communication_sms,
            async_with_compute_stream=True,
            allocate_on_comm_stream=False,
            do_cpu_sync=False,
            do_handle_copy=False,
            do_expand=True,
            do_zero_padding=True,
            use_tma_aligned_col_major_sf=True,
        )
        event.current_stream_wait()
        record(2)
        w1, s1, w2, s2 = weights[bank]
        rows = recv_x[0].shape[0]
        first_output = torch.empty((rows, 2 * args.intermediate), dtype=torch.bfloat16, device="cuda")
        deepgemm_grouped_fp8_nt_contiguous(
            input_tuple=recv_x,
            w_tuple=(w1, s1),
            out=first_output,
            m_indices=handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m,
        )
        record(3)
        middle = torch.empty((rows, args.intermediate), dtype=torch.float8_e4m3fn, device="cuda")
        scale_storage = torch.empty((args.intermediate // 128, (rows + 3) // 4 * 4), dtype=torch.float32, device="cuda")
        middle_scales = scale_storage[:, :rows].T
        silu_and_mul_psum_post_quant_fwd(
            input=first_output,
            output=middle,
            output_scale=middle_scales,
            expert_token_psum=handle.psum_num_recv_tokens_per_expert,
            expert_alignment=128,
            quant_group_size=128,
        )
        record(4)
        expert_output = torch.empty((rows, args.hidden), dtype=torch.bfloat16, device="cuda")
        deepgemm_grouped_fp8_nt_contiguous(
            input_tuple=(middle, middle_scales),
            w_tuple=(w2, s2),
            out=expert_output,
            m_indices=handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m,
        )
        record(5)
        dense, compact = ep_reduce_decode_output(
            expert_output=expert_output,
            route_weights=recv_weights,
            recv_src_metadata=handle.recv_src_metadata,
            num_valid_recv_tokens=handle.psum_num_recv_tokens_per_scaleup_rank[-1:],
        )
        handle.recv_src_metadata = compact
        record(6)
        output, _, event = buffer.combine(
            dense,
            handle=handle,
            topk_weights=None,
            num_sms=communication_sms,
            async_with_compute_stream=True,
            allocate_on_comm_stream=False,
        )
        event.current_stream_wait()
        record(7)
        operands = dict(recv_x=recv_x, psum=handle.psum_num_recv_tokens_per_expert, middle=(middle, middle_scales))
        return output, handle, rows, operands

    return forward


def capture(forward, steps, banks, instrument):
    stamps = (
        [[torch.cuda.Event(enable_timing=True, external=True) for _ in range(len(PHASES) + 1)] for _ in range(steps)]
        if instrument
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
        for step in range(steps):
            output, _, _, _ = forward(bank=step % banks, stamps=stamps[step] if instrument else None)
    return graph, stamps, output


def capture_compute(operands, weights, args, kind):
    """独立计时真实 GEMM/计算流水线，graph 内不插入分段 timing event。"""
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep.deepgemm import (
        decode_masked_group_gemm,
        deepgemm_grouped_fp8_nt_contiguous,
    )

    expected_m = math.ceil(args.global_tokens * args.topk / args.experts)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in range(args.graph_steps):
            bank = step % args.banks
            data = operands[bank]
            w1, s1, w2, s2 = weights[bank]
            if kind == "compute":
                output = decode_masked_group_gemm(
                    recv_x=data["recv_x"],
                    expert_token_psum=data["psum"],
                    expert_alignment=128,
                    dtype=torch.bfloat16,
                    w1=w1,
                    w1_scale=s1,
                    w2=w2,
                    w2_scale=s2,
                    expected_m=expected_m,
                )
            else:
                input_tuple, weight_tuple, width = (
                    (data["recv_x"], (w1, s1), 2 * args.intermediate)
                    if kind == "w1"
                    else (data["middle"], (w2, s2), args.hidden)
                )
                output = torch.empty((input_tuple[0].shape[0], width), dtype=torch.bfloat16, device="cuda")
                deepgemm_grouped_fp8_nt_contiguous(
                    input_tuple=input_tuple,
                    w_tuple=weight_tuple,
                    out=output,
                    m_indices=data["psum"],
                    use_psum_layout=True,
                    expected_m_for_psum_layout=expected_m,
                )
    return graph, output


def run_case(spec, placement, weights, buffer, args, control_group):
    import deep_gemm
    from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
        eplb_hash_targets_topk_ids,
    )
    from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
        build_logical_to_rank_physical_map,
        build_logical_to_physical_map,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    physical_experts = len(placement[0]) * world
    reserved_sms = buffer.get_theoretical_num_sms(physical_experts, args.topk)
    compute_sms = torch.cuda.get_device_properties(rank).multi_processor_count - reserved_sms
    deep_gemm.set_num_sms(compute_sms)
    communication_sms = args.communication_sms or (
        torch.cuda.get_device_properties(rank).multi_processor_count * 3 // 5 // 2 * 2
    )
    routing_rows = build_logical_to_physical_map(placement, args.experts, current_rank=rank, node_world_size=world)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, args.experts)
    logical_map = torch.tensor(routing_rows, dtype=torch.int32, device="cuda")
    logical_to_rank_physical_map = torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda")
    local_tokens = args.global_tokens // world
    inputs, routes, route_statistics = [], [], []
    all_physical = torch.tensor(placement, dtype=torch.int64, device="cuda").flatten()
    for bank in range(args.banks):
        logical_cpu = build_logical_routes(spec, args.global_tokens, bank, args.experts, world, args.topk)
        logical = logical_cpu[rank * local_tokens : (rank + 1) * local_tokens].contiguous().cuda()
        if spec["routing"] == "none":
            physical = logical.to(torch.int64)
        else:
            assert spec["routing"] == "current"
            physical = eplb_hash_targets_topk_ids(
                logical_topk_ids=logical,
                logical_to_physical_map=logical_map,
                logical_to_rank_physical_map=logical_to_rank_physical_map,
                current_rank=rank,
                num_physical_experts_per_rank=len(placement[0]),
            ).to(torch.int64)
        assert torch.equal(all_physical[physical], logical.to(torch.int64))
        routes.append(physical)
        generator = torch.Generator(device="cuda").manual_seed(100 + rank + bank * world)
        inputs.append(
            torch.randn((local_tokens, args.hidden), dtype=torch.bfloat16, device="cuda", generator=generator)
        )
        targets = torch.nn.functional.one_hot(physical // len(placement[0]), num_classes=world).any(dim=1)
        targets[:, rank] = False
        route_statistics.append(dict(remote_rank_sum=int(targets.sum().item()), expert_ids=physical.cpu().tolist()))

    forward = make_forward(buffer, inputs, routes, weights, args, communication_sms)
    snapshots, operands = [], []
    for bank in range(args.banks):
        output, handle, expanded_capacity, data = forward(bank=bank)
        torch.cuda.synchronize()
        assert bool(output.isfinite().all().item())
        counts = handle.num_unaligned_recv_tokens_per_expert.cpu().tolist()
        psum = handle.psum_num_recv_tokens_per_expert.cpu().tolist()
        padded_rows = sum((count + 127) // 128 * 128 for count in counts)
        assert (psum[-1] + 127) // 128 * 128 == padded_rows
        snapshots.append(
            dict(
                counts=counts,
                valid_rows=sum(counts),
                active_experts=sum(count > 0 for count in counts),
                padded_rows=padded_rows,
                deduplicated_received=int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()),
                recv_capacity_rows=handle.recv_src_metadata.shape[0],
                expanded_capacity_rows=expanded_capacity,
            )
        )
        # do_handle_copy=False 的元数据可能复用内部存储，独立计算图需固定
        # 每个 bank 的输入与 psum；clone 在计时外完成，保留 scale 的列主序。
        operands.append(
            dict(
                recv_x=tuple(tensor.clone() for tensor in data["recv_x"]),
                psum=data["psum"].clone(),
                middle=tuple(tensor.clone() for tensor in data["middle"]),
            )
        )

    plain, _, plain_output = capture(forward, args.graph_steps, args.banks, instrument=False)
    profiled, stamps, profiled_output = capture(forward, args.graph_steps, args.banks, instrument=True)
    isolated_graphs = {kind: capture_compute(operands, weights, args, kind)[0] for kind in ("w1", "w2", "compute")}
    for _ in range(3):
        plain.replay()
        profiled.replay()
        for graph in isolated_graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    assert bool(plain_output.isfinite().all().item()) and bool(profiled_output.isfinite().all().item())
    plain_us, phase_us = [], []
    isolated_us = {kind: [] for kind in isolated_graphs}
    thermal_guard = ThermalGuard(rank, control_group, args.cool_temperature)
    thermal_samples = []
    for round_index in range(args.rounds):
        kinds = ("plain", "profiled", "w1", "w2", "compute")
        for kind in kinds if round_index % 2 == 0 else kinds[::-1]:
            chunk_us, chunk_phases = [], []
            # 较短的图累计相同的 128 次调用；块间冷却不计入耗时，避免长图
            # 在一次计时内部触发降频，不能靠事后删掉慢样本掩盖硬件差异。
            for chunk in range(args.steps // args.graph_steps):
                thermal_guard.cool()
                dist.barrier(device_ids=[rank])
                thermal_guard.begin()
                if kind != "profiled":
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    (plain if kind == "plain" else isolated_graphs[kind]).replay()
                    end.record()
                    end.synchronize()
                    chunk_us.append(start.elapsed_time(end) * 1000 / args.graph_steps)
                else:
                    profiled.replay()
                    torch.cuda.synchronize()
                    chunk_phases.extend(
                        [
                            [row[index].elapsed_time(row[index + 1]) * 1000 for index in range(len(PHASES))]
                            for row in stamps
                        ]
                    )
                thermal_samples.append(dict(round=round_index, kind=kind, chunk=chunk, **thermal_guard.end()))
            if kind == "profiled":
                phase_us.append(chunk_phases)
            else:
                target = plain_us if kind == "plain" else isolated_us[kind]
                target.append(float(np.mean(chunk_us)))
        if rank == 0:
            print(json.dumps(dict(progress=spec["name"], round=round_index + 1, rounds=args.rounds)), flush=True)

    local = dict(
        rank=rank,
        plain_us=plain_us,
        phase_us=phase_us,
        snapshots=snapshots,
        routes=route_statistics,
        isolated_us=isolated_us,
        thermal_samples=thermal_samples,
    )
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local, group=control_group)
    if rank == 0:
        return summarize(spec, gathered, args, communication_sms, compute_sms)
    return None


def summarize(spec, ranks, args, communication_sms, compute_sms):
    # [rank, round, step, phase]；先按同一次调用取 rank 临界值，避免
    # 把不同层/不同调用的 rank 热点平均掉。
    phases = np.asarray([rank["phase_us"] for rank in ranks], dtype=np.float64)
    compute = phases[..., 2:5].sum(axis=-1)
    gemm = phases[..., 2] + phases[..., 4]
    total = phases.sum(axis=-1)
    critical_total = total.max(axis=0)
    critical_compute = compute.max(axis=0)
    critical_gemm = gemm.max(axis=0)
    plain = np.asarray([rank["plain_us"] for rank in ranks])
    counts = np.asarray([[snapshot["counts"] for snapshot in rank["snapshots"]] for rank in ranks])
    # 检查 dispatch 实际接收计数与全部源 rank 的 physical 路由严格对应。
    for bank in range(args.banks):
        assigned = [expert for rank in ranks for token in rank["routes"][bank]["expert_ids"] for expert in token]
        expected = np.bincount(assigned, minlength=counts.shape[0] * counts.shape[-1]).reshape(counts.shape[0], -1)
        assert np.array_equal(expected, counts[:, bank])
        assert expected.sum() == args.global_tokens * args.topk
    per_rank_rows = counts.sum(axis=-1)
    per_rank_active = (counts > 0).sum(axis=-1)
    padded = ((counts + 127) // 128 * 128).sum(axis=-1)
    remote_ranks = sum(bank["remote_rank_sum"] for rank in ranks for bank in rank["routes"]) / (
        args.global_tokens * args.banks
    )
    result = dict(
        **spec,
        communication_sms=communication_sms,
        compute_sms=compute_sms,
        plain_moe_us=float(np.median(plain.max(axis=0))),
        profiled_moe_us=float(np.median(critical_total)),
        critical_w1_us=float(np.median(phases[..., 2].max(axis=0))),
        critical_w2_us=float(np.median(phases[..., 4].max(axis=0))),
        critical_gemm_us=float(np.median(critical_gemm)),
        critical_compute_us=float(np.median(critical_compute)),
        critical_gemm_share_pct=float(np.median(100 * critical_gemm / critical_total)),
        critical_compute_share_pct=float(np.median(100 * critical_compute / critical_total)),
        phases_critical_us={
            name: float(np.median(phases[..., index].max(axis=0))) for index, name in enumerate(PHASES)
        },
        phase_rank_max_minus_min_us={
            name: float(np.median(np.ptp(phases[..., index], axis=0))) for index, name in enumerate(PHASES)
        },
        rank_compute_median_us=np.median(compute, axis=(1, 2)).tolist(),
        received_expert_rows_per_rank=per_rank_rows.tolist(),
        active_experts_per_rank=per_rank_active.tolist(),
        padded_rows_per_rank=padded.tolist(),
        average_remote_ranks=remote_ranks,
        max_rank_rows_over_mean=float(np.median(per_rank_rows.max(axis=0) / per_rank_rows.mean(axis=0))),
        max_rank_padding_over_mean=float(np.median(padded.max(axis=0) / padded.mean(axis=0))),
        plain_samples_rank_round_us=plain.tolist(),
        compute_samples_critical_round_us=np.median(critical_compute, axis=1).tolist(),
        gemm_samples_critical_round_us=np.median(critical_gemm, axis=1).tolist(),
        profile_samples_critical_round_us=np.median(critical_total, axis=1).tolist(),
        isolated_critical_us={
            kind: float(np.median(np.asarray([rank["isolated_us"][kind] for rank in ranks]).max(axis=0)))
            for kind in ("w1", "w2", "compute")
        },
        isolated_samples_rank_round_us={
            kind: [rank["isolated_us"][kind] for rank in ranks] for kind in ("w1", "w2", "compute")
        },
        thermal_samples_per_rank=[rank["thermal_samples"] for rank in ranks],
        max_measured_temperature=max(sample["max_temperature"] for rank in ranks for sample in rank["thermal_samples"]),
        thermal_slowdown_observed=any(
            sample["thermal_slowdown"] for rank in ranks for sample in rank["thermal_samples"]
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=272)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--graph-steps", type=int, default=0, help="单图调用数；0 自动使用最多 16 次，小图分块累计 steps 次调用")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--communication-sms", type=int, default=0)
    parser.add_argument("--cool-temperature", type=int, default=80, help="每次计时前等待所有卡被动冷却到此温度；0 表示禁用")
    parser.add_argument("--print-gemm-configs", action="store_true", help="仅 rank 0 打印实际 DeepGEMM tile 配置，供成本归因使用")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.graph_steps = args.graph_steps or min(args.steps, 16)
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    rank = int(os.environ["LOCAL_RANK"])
    if args.print_gemm_configs and rank == 0:
        os.environ["DG_PRINT_CONFIGS"] = "1"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=600))
    world = dist.get_world_size()
    assert world == 8 and args.experts == 256 and args.topk == 8
    assert args.global_tokens % world == 0 and args.global_tokens // world <= args.capacity
    assert args.steps % args.banks == 0 and args.hidden % 128 == args.intermediate % 128 == 0
    assert args.steps % args.graph_steps == 0 and args.graph_steps % args.banks == 0
    assert args.capacity % 8 == 0 and args.steps > 0 and args.rounds > 0
    import deep_ep
    import deep_gemm
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import (
        set_mk_alignment_for_contiguous_layout,
    )

    dist.barrier(device_ids=[rank])
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
    selected = [spec for spec in case_specs() if not args.cases or spec["name"] in args.cases]
    assert selected and (not args.cases or set(args.cases) == {spec["name"] for spec in selected})
    results = dict(
        metadata=dict(
            gpu=torch.cuda.get_device_name(rank),
            world_size=world,
            global_tokens=args.global_tokens,
            tokens_per_source_rank=args.global_tokens // world,
            hidden=args.hidden,
            intermediate=args.intermediate,
            experts=args.experts,
            topk=args.topk,
            capacity=args.capacity,
            alignment=128,
            banks=args.banks,
            steps=args.steps,
            rounds=args.rounds,
            torch=torch.__version__,
            deep_gemm=deep_gemm.__file__,
            graph_steps=args.graph_steps,
            scope="routed MoE only; precomputed routing; no attention/shared expert/router/whole-model inference",
            cache_policy="four distinct layer-weight banks rotated per call; not a hot single-layer benchmark",
            cool_temperature=args.cool_temperature,
        ),
        cases=[],
    )
    previous_layout, weights = None, None
    try:
        if rank == 0:
            print(json.dumps(results["metadata"]), flush=True)
        for spec in selected:
            placement = build_placement(spec, args.experts, world, args.topk)
            layout = tuple(placement[rank])
            if layout != previous_layout:
                weights = make_weights(layout, args.hidden, args.intermediate, args.banks)
                previous_layout = layout
            if rank == 0:
                print(json.dumps(dict(starting=spec["name"])), flush=True)
            result = run_case(spec, placement, weights, buffer, args, control_group)
            if rank == 0:
                results["cases"].append(result)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k
                            not in (
                                "received_expert_rows_per_rank",
                                "active_experts_per_rank",
                                "padded_rows_per_rank",
                                "plain_samples_rank_round_us",
                                "compute_samples_critical_round_us",
                                "gemm_samples_critical_round_us",
                                "profile_samples_critical_round_us",
                                "isolated_samples_rank_round_us",
                                "thermal_samples_per_rank",
                            )
                        }
                    ),
                    flush=True,
                )
            dist.barrier(group=control_group)
    finally:
        torch.cuda.synchronize()
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
