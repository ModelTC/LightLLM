import os
from collections import deque

os.environ["LIGHTLLM_DSV4_EPLB_TILE_ROUTING"] = "1"

import torch
import torch.distributed as dist

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_logical_to_physical_maps_for_layers,
    validate_physical_placement,
)
from lightllm.models.deepseek_v4.triton_kernel.eplb_tile_route import _load_cuda, route


def masks_for(mapping, replicas, slots):
    result = []
    for expert in range(256):
        mask = 0
        for index in range(int(replicas[expert])):
            mask |= 1 << (int(mapping[expert, index]) // slots)
        result.append(mask)
    return result


def cpu_quotas(counts, masks):
    q = [(int(value) + 127) // 128 for value in counts]
    subsets = [0] * 256
    for need, mask in zip(q, masks):
        subsets[mask] += need
    for bit in range(8):
        for subset in range(256):
            if subset & (1 << bit):
                subsets[subset] += subsets[subset ^ (1 << bit)]
    target = max((subsets[subset] + subset.bit_count() - 1) // subset.bit_count() for subset in range(1, 256))
    flow = [[0] * 8 for _ in range(256)]
    load = [0] * 8
    for expert in sorted(range(256), key=lambda value: (masks[value].bit_count() != 1, value)):
        remaining = q[expert]
        while remaining:
            parent = [None] * 8
            queue = deque()
            for rank in range(8):
                if masks[expert] & (1 << rank):
                    parent[rank] = (-1, -1)
                    queue.append(rank)
            sink = None
            while queue and sink is None:
                source = queue.popleft()
                if load[source] < target:
                    sink = source
                    break
                for old_expert in range(256):
                    if not flow[old_expert][source]:
                        continue
                    for destination in range(8):
                        if masks[old_expert] & (1 << destination) and parent[destination] is None:
                            parent[destination] = (source, old_expert)
                            queue.append(destination)
            assert sink is not None
            path = []
            node = sink
            while parent[node] != (-1, -1):
                source, old_expert = parent[node]
                path.append((source, node, old_expert))
                node = source
            delta = (
                min(remaining, target - load[sink], *(flow[item][source] for source, _, item in path))
                if path
                else min(remaining, target - load[sink])
            )
            assert delta > 0
            flow[expert][node] += delta
            for source, destination, old_expert in path:
                flow[old_expert][source] -= delta
                flow[old_expert][destination] += delta
            load[sink] += delta
            remaining -= delta
    return torch.tensor(flow, dtype=torch.int32), target


def layout(redundancy):
    slots = 32 + redundancy
    placement = torch.arange(256, dtype=torch.int32).reshape(8, 32)
    if redundancy:
        extras = torch.empty((8, redundancy), dtype=torch.int32)
        for rank in range(8):
            for extra in range(redundancy):
                extras[rank, extra] = ((rank + 1 + extra) % 8) * 32 + extra
        placement = torch.cat((placement, extras), dim=1)
    validate_physical_placement(placement.unsqueeze(0), 256)
    mapping, replicas = build_logical_to_physical_maps_for_layers(placement.unsqueeze(0), 256, full_layout=True)
    return placement, mapping[0], replicas[0], slots


def assert_route(placement, logical, physical, expected_quota, expected_t, slots):
    if logical.numel():
        assert torch.equal(placement.reshape(-1)[physical.cpu()], logical.cpu())
    actual = torch.zeros((256, 8), dtype=torch.int32, device=logical.device)
    if logical.numel():
        actual.index_put_(
            (logical.reshape(-1), torch.div(physical.reshape(-1), slots, rounding_mode="floor")),
            torch.ones(logical.numel(), dtype=torch.int32, device=logical.device),
            accumulate=True,
        )
    dist.all_reduce(actual)
    assert torch.equal(torch.div(actual + 127, 128, rounding_mode="floor").cpu(), expected_quota)
    assert int(actual.max().cpu()) >= 0
    assert int(expected_quota.sum(0).max()) == expected_t


def run_case(redundancy, rank, device):
    placement, mapping_cpu, replicas_cpu, slots = layout(redundancy)
    mapping = mapping_cpu.to(device).contiguous()
    replicas = replicas_cpu.to(device).contiguous()
    for expert in range(256):
        count = int(replicas_cpu[expert])
        if count > 1:
            mapping[expert, :count] = mapping[expert, :count].roll(rank % count)
    rows = 0 if rank == 0 else rank * 19
    logical = (
        ((torch.arange(rows * 6, device=device, dtype=torch.int64) * 17) + rank * 31)
        .remainder(256)
        .view(rows, 6)
        .contiguous()
    )
    physical = torch.empty_like(logical)
    route(logical, physical, mapping, replicas, slots, rank)
    counts = torch.zeros(256, dtype=torch.int32, device=device)
    if logical.numel():
        counts.index_add_(0, logical.reshape(-1), torch.ones(logical.numel(), dtype=torch.int32, device=device))
    gathered = torch.empty((8, 256), dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(gathered, counts)
    expected_quota, expected_t = cpu_quotas(gathered.sum(0).cpu().tolist(), masks_for(mapping_cpu, replicas_cpu, slots))
    assert_route(placement, logical, physical, expected_quota, expected_t, slots)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    # Production _load_cuda, not the standalone prototype module.
    assert "eplb_tile_route" in _load_cuda().__name__
    for redundancy in (0, 1, 2):
        run_case(redundancy, rank, device)
    empty = torch.empty((0, 6), dtype=torch.int64, device=device)
    placement, mapping, replicas, slots = layout(2)
    physical = torch.empty_like(empty)
    route(empty, physical, mapping.to(device).contiguous(), replicas.to(device).contiguous(), slots, rank)
    run_select_experts_cases(placement, mapping, replicas, slots, rank, device)
    dist.barrier()
    if rank == 0:
        print("production_wrapper_8rank passed R0/R1/R2 plus all-empty")
    dist.destroy_process_group()


def run_select_experts_cases(placement, mapping_cpu, replicas_cpu, slots, rank, device):
    """Exercise the production layer hook with legal distributed full EPLB state."""
    from types import SimpleNamespace

    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
        EPLBState,
        ExpertParallelState,
    )
    from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import (
        DeepseekV4TransformerLayerInfer,
    )

    vocab, topk = 32, 6

    def make_eplb(record_load):
        map_ = mapping_cpu.to(device).contiguous()
        for expert in range(256):
            copies = int(replicas_cpu[expert])
            if copies > 1:
                map_[expert, :copies] = map_[expert, :copies].roll(rank % copies)
        return EPLBState(
            num_redundant_experts_per_rank=2,
            initial_redundant_expert_ids_by_rank=placement.unsqueeze(0),
            logical_to_physical_map=map_,
            logical_replica_count=replicas_cpu.to(device).contiguous(),
            route_counter=torch.zeros((4, 256), dtype=torch.int64, device=device),
            recording=record_load,
            full_layout=True,
            physical_to_logical=placement,
        )

    def call(is_hash, capture, record_load, tile_enabled, empty=False):
        eplb = make_eplb(record_load)
        state = ExpertParallelState(num_logical_experts=256, world_size=8, eplb=eplb)
        experts = SimpleNamespace(expert_parallel_state=state, global_rank_=rank)
        text_bias = torch.zeros(256, dtype=torch.float32, device=device)
        vision_bias = torch.zeros(256, dtype=torch.float32, device=device)
        text_bias[20:26] = 100
        vision_bias[200:206] = 100
        table = torch.zeros((vocab, topk), dtype=torch.long, device=device)
        table[2] = torch.tensor([1, 4, 7, 10, 13, 16], device=device)
        layer_weight = SimpleNamespace(
            experts_=experts,
            gate_bias_=SimpleNamespace(weight=text_bias),
            gate_bias_vl_=SimpleNamespace(weight=vision_bias),
            gate_tid2eid_=SimpleNamespace(weight=table),
        )
        infer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
        infer.is_hash = is_hash
        infer.has_vision = True
        infer.vocab_size = vocab
        infer.num_experts_per_tok = topk
        infer.routed_scaling_factor = 1.0
        infer.alloc_tensor = torch.empty
        rows = 0 if empty or rank == 0 else 3
        infer_state = SimpleNamespace(
            is_prefill=True,
            input_ids=torch.tensor([2, vocab, vocab + 4096], dtype=torch.long, device=device)[:rows],
        )
        logits = torch.arange(rows * 256, dtype=torch.float32, device=device).view(rows, 256) / 1000
        os.environ["LIGHTLLM_DSV4_EPLB_TILE_ROUTING"] = "1" if tile_enabled else "0"
        from lightllm.utils.envs_utils import get_dsv4_eplb_tile_routing

        get_dsv4_eplb_tile_routing.cache_clear()
        weights, physical, logical = infer._select_experts(logits, infer_state, layer_weight, capture)
        return weights, physical, logical, eplb, logits, infer_state, layer_weight

    for is_hash in (False, True):
        for capture in (False, True):
            for record_load in (False, True):
                # Capture logical reference before enabling tile routing.
                ref_weights, _, ref_logical, ref_eplb, _, _, _ = call(is_hash, True, record_load, False)
                weights, physical, logical, eplb, _, _, _ = call(is_hash, capture, record_load, True)
                torch.testing.assert_close(weights, ref_weights)
                if physical.numel():
                    assert torch.equal(placement.reshape(-1)[physical.cpu()], ref_logical.cpu())
                if capture:
                    torch.testing.assert_close(logical, ref_logical)
                else:
                    assert logical is None
                torch.testing.assert_close(eplb.route_counter, ref_eplb.route_counter)
                assert eplb.recorded_sample_count == ref_eplb.recorded_sample_count
    # Every rank still enters the routing collective when the layer has no rows.
    _, physical, logical, eplb, _, _, _ = call(True, False, True, True, empty=True)
    assert physical.numel() == 0 and logical is None
    assert eplb.recorded_sample_count == 1


# Keep the torchrun entry point at the bottom so helpers are defined before use.

if __name__ == "__main__":
    main()
