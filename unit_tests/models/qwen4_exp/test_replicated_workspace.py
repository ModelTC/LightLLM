import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


class Projection:
    def __init__(self, weight):
        self.weight = weight

    def mm(self, value, out=None):
        return torch.mm(value, self.weight.t(), out=out)


def _reference(inputs, mask, ids, scores, layer_weight, rank, world):
    rows, hidden = inputs.shape
    up, down = layer_weight.experts.w13.weight, layer_weight.experts.w2.weight
    experts, first = up.shape[0], rank * up.shape[0]
    weights = F.softmax(scores, dim=-1)
    partial = torch.zeros_like(inputs, dtype=torch.float32)
    for route in range(ids.shape[1]):
        local = mask & (ids[:, route] >= first) & (ids[:, route] < first + experts)
        for expert in ids[local, route].unique().tolist():
            selected = local & (ids[:, route] == expert)
            gate, value = F.linear(inputs[selected], up[expert - first]).chunk(2, dim=-1)
            mid = F.silu(gate.float()).to(inputs.dtype) * value
            partial[selected] += F.linear(mid, down[expert - first]).float() * weights[selected, route, None]
    gathered = torch.empty(world * rows, hidden, dtype=inputs.dtype, device=inputs.device)
    dist.all_gather_into_tensor(gathered, partial.to(inputs.dtype))
    gathered = gathered.view(world, rows, hidden)
    routed = torch.zeros_like(inputs, dtype=torch.float32)
    for row, (live, routes) in enumerate(zip(mask.tolist(), ids.tolist())):
        if live:
            owners = [expert // experts for expert in routes]
            last = {owner: position for position, owner in enumerate(owners)}
            for position, owner in enumerate(owners):
                if last[owner] == position:
                    routed[row] += gathered[owner, row].float()
    gate, value = F.linear(inputs, layer_weight.gate_up_proj.weight).chunk(2, dim=-1)
    shared = F.linear(F.silu(gate.float()).to(inputs.dtype) * value, layer_weight.down_proj.weight)
    scalar = torch.sigmoid(F.linear(inputs, layer_weight.shared_expert_gate.weight).float()).to(inputs.dtype)
    shared = (shared.float() * scalar.float()).to(inputs.dtype)
    return (routed.to(inputs.dtype).float() + shared.float()).to(inputs.dtype)


def _worker(rank, world, rendezvous):
    from lightllm.server.core.objs.start_args_type import StartArgs

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    args = StartArgs()
    args.hardware_platform, args.data_type = "cuda", "bf16"
    args.enable_ep_moe = args.enable_tpsp_mix_mode = True
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(vars(args))
    for name, value in {
        "LIGHTLLM_DP_WORLD_SIZE": world,
        "LIGHTLLM_GLOBAL_WORLD_SIZE": world,
        "LIGHTLLM_NODE_WORLD_SIZE": world,
        "LIGHTLLM_CURRENT_DEVICE_ID": rank,
        "LIGHTLLM_CURRENT_RANK_IN_DP": rank,
        "LIGHTLLM_CURRENT_RANK_IN_NODE": rank,
        "LIGHTLLM_GLOBAL_RANK": rank,
        "LIGHTLLM_DP_SIZE": 1,
        "LIGHTLLM_GLOBAL_DP_RANK": 0,
        "LIGHTLLM_DP_RANK_IN_NODE": 0,
    }.items():
        os.environ[name] = str(value)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=180),
    )
    from lightllm.distributed.symm_mem_all_reduce import SymmMemAllreduce
    from lightllm.models.qwen4_exp.layer_infer.replicated_decoder import ReplicatedDecoderLayer

    graphs = []
    try:
        symmetric = SymmMemAllreduce(dist.group.WORLD, rank)
        available = torch.tensor(int(not symmetric.disabled), device=rank)
        dist.all_reduce(available, op=dist.ReduceOp.MIN)
        if not available.item():
            if rank == 0:
                Path(rendezvous + ".unavailable").touch()
            return
        group = SimpleNamespace(symm_mem_reduce=symmetric, device_group=dist.group.WORLD)
        experts, hidden, inter, topk = 512 // world, 2560, 640, 10
        tensor = dict(dtype=torch.bfloat16, device=rank)
        torch.manual_seed(701)
        shared_up = torch.randn(2 * inter, hidden, **tensor) * 0.02
        shared_down = torch.randn(hidden, inter, **tensor) * 0.02
        shared_gate = torch.randn(1, hidden, **tensor) * 0.1
        torch.manual_seed(811 + rank)
        weights = SimpleNamespace(
            experts=SimpleNamespace(
                w13=SimpleNamespace(weight=torch.randn(experts, 2 * inter, hidden, **tensor) * 0.02),
                w2=SimpleNamespace(weight=torch.randn(experts, hidden, inter, **tensor) * 0.02),
            ),
            gate_up_proj=Projection(shared_up),
            down_proj=Projection(shared_down),
            shared_expert_gate=Projection(shared_gate),
        )
        layer = ReplicatedDecoderLayer.__new__(ReplicatedDecoderLayer)
        layer.tp_rank_, layer.tp_world_size_ = rank, world
        pool = torch.cuda.graph_pool_handle()
        cases, pointers = [], None
        for rows in (3, 4, 5, 28, 63, 64):
            torch.manual_seed(913 + rows)
            inputs = torch.randn(rows, hidden, **tensor) * 0.5
            mask = torch.ones(rows, dtype=torch.bool, device=rank)
            logits = torch.full((rows, 512), -1000.0, device=rank)
            ids = torch.arange(topk, device=rank)[None, :].repeat(rows, 1)
            scores = (10 - torch.arange(topk, device=rank) * 0.125)[None, :].expand(rows, -1)
            logits.scatter_(1, ids, scores)
            weights.moe_gate = SimpleNamespace(mm=lambda _: logits)
            state = SimpleNamespace(dist_group=group, qwen4_exp_metadata=SimpleNamespace(token_mask=mask))
            layer._moe_ffn_edp(inputs, state, weights)
            current = {
                name: getattr(layer, name).data_ptr()
                for name in (
                    "routes",
                    "masked_ids",
                    "counts",
                    "mid",
                    "down",
                    "partial",
                    "shared_up",
                    "shared_mid",
                    "shared_output",
                    "shared_gate",
                )
            }
            if pointers is None:
                pointers = current
            assert current == pointers
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph, pool=pool):
                actual = layer._moe_ffn_edp(inputs, state, weights)
            cases.append((inputs, mask, logits, ids, scores, actual, graph))

        for change in range(4):
            for inputs, mask, logits, ids, scores, actual, graph in reversed(cases):
                rows = inputs.shape[0]
                inputs.mul_(0.9)
                if change == 2:
                    ids.copy_((world - 1) * experts + torch.arange(topk, device=rank)[None, :])
                elif change == 3:
                    ids.copy_(
                        torch.tensor(
                            [
                                experts - 1,
                                experts,
                                2 * experts - 1,
                                2 * experts,
                                3 * experts - 1,
                                3 * experts,
                                511,
                                0,
                                1,
                                2,
                            ],
                            device=rank,
                        )[None, :]
                    )
                else:
                    ids.copy_(
                        (torch.arange(rows, device=rank)[:, None] * 17 + torch.arange(topk, device=rank)[None, :] * 53)
                        % 512
                    )
                logits.fill_(-1000)
                logits.scatter_(1, ids, scores)
                mask.copy_(torch.arange(rows, device=rank) % 3 != 0)
                if change == 0:
                    mask.zero_()
                graph.replay()
                expected = _reference(inputs, mask, ids, scores, weights, rank, world)
                torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.002)
        dist.barrier()
    finally:
        for graph in graphs:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [4, 8])
def test_replicated_workspace_graphs(world, tmp_path):
    if torch.cuda.device_count() < world:
        pytest.skip(f"Requires {world} CUDA GPUs")
    if any(torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(world)):
        pytest.skip("Requires Hopper GPUs for the replicated decode profile")
    from lightllm.utils.device_utils import has_nvlink

    if not has_nvlink():
        pytest.skip("Requires an NVLink-connected node")
    rendezvous = str(tmp_path / "rendezvous")
    mp.spawn(_worker, args=(world, rendezvous), nprocs=world, join=True)
    if Path(rendezvous + ".unavailable").exists():
        pytest.skip("Symmetric multicast memory is unavailable")
