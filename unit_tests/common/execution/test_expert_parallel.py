"""Expert shard math and real two-rank DeepEP transfers."""

from lightllm.common.basemodel.weight_binding import bind_moe


from dataclasses import replace
from datetime import timedelta
import importlib.util
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightllm.common.execution.compute.moe.expert import ExpertComputeBackend
from lightllm.common.execution.runner.expert import ExpertRunner
from lightllm.common.execution.contracts.device import DeviceCaps
from lightllm.common.execution.contracts.state import FormatSpec
from lightllm.common.execution.compute.moe.spec import MoeSpec
from lightllm.common.execution.transfer.expert import ExpertDispatch, ExpertTransfer


def ep_format(rank=0):
    return FormatSpec(
        "fp8_block128",
        torch.float8_e4m3fn,
        "local_expert_output_input",
        ("gate_up", "down", "gate_up_scale", "down_scale"),
        1,
        (rank, 2),
        group_size=128,
        scale_dtype=torch.float32,
    )


def test_ep_offer_keeps_expert_sharding_separate_from_tp_and_shared_experts():
    backend, fmt, caps = ExpertComputeBackend(), ep_format(), DeviceCaps("cuda", 0, (9, 0))
    spec = MoeSpec(4096, 512, 8, 2, torch.bfloat16, shared_experts=1, gated_shared=True)
    offer = backend.probe(spec, fmt, caps).offers[0]
    assert offer.format.shard == (0, 2) and offer.spec.intermediate_size == 512
    assert not offer.spec.gated_shared and not offer.spec.shared_experts
    for bad_spec, bad_fmt, bad_caps in (
        (replace(spec, experts=7), fmt, caps),
        (spec, replace(fmt, layout="expert_output_input"), caps),
        (spec, replace(fmt, scale_dtype=torch.int32), caps),
        (replace(spec, dtype=torch.float16), fmt, caps),
        (spec, fmt, DeviceCaps("cuda", 0, (10, 0))),
    ):
        assert backend.probe(bad_spec, bad_fmt, bad_caps).reason
    rounded = backend.probe(replace(spec, round_intermediates=True), fmt, caps).offers[0]
    assert rounded.spec.round_intermediates


@pytest.mark.parametrize("prefill", [True, False])
def test_ep_receive_hook_and_completion_keep_the_source_until_ready(prefill):
    calls = []
    event = SimpleNamespace(query=lambda: "done" in calls, synchronize=lambda: calls.append("done"))
    runtime = SimpleNamespace(record_event=lambda: event)
    handle = SimpleNamespace(
        num_recv_tokens_per_expert_list=[128], num_unaligned_recv_tokens_per_expert=[2], recv_src_metadata="metadata"
    )
    received = torch.ones(2, 4)
    hook = lambda: calls.append("receive")
    overlap = SimpleNamespace(current_stream_wait=hook)
    buffer = SimpleNamespace(
        dispatch=lambda *a, **kw: (received, None, a[0], handle, overlap),
        combine=lambda *a, **kw: (received, None, overlap),
        low_latency_dispatch=lambda *a, **kw: (received, [2], handle, None, hook),
        low_latency_combine=lambda *a, **kw: (received, None, hook),
    )
    transfer = ExpertTransfer(
        buffer,
        buffer,
        experts=8,
        prefill_limit=8,
        decode_limit=8,
        num_sms=0,
        round_scales=False,
        runtime=runtime,
        capture=lambda: None,
    )
    source, ids, weights = torch.randn(2, 4), torch.zeros(2, 2, dtype=torch.int64), torch.ones(2, 2)
    exchange = transfer.dispatch(source, ids, weights, prefill=prefill, overlap=True)
    assert not exchange.completion.source_reusable.query()
    assert exchange.completion.keepalive[0] is source
    exchange.wait()
    exchange.wait()
    assert calls == ["receive"]
    assert not exchange.completion.destination_ready.query()
    exchange.completion.destination_ready.synchronize()
    assert exchange.completion.source_reusable.query()
    out, completion = transfer.combine(received, exchange, prefill=prefill, overlap=True)
    assert out is received and completion.keepalive[1] is exchange
    completion.destination_ready.wait()
    assert calls == ["receive", "done", "receive"]
    with pytest.raises(ValueError, match="capacity"):
        transfer.dispatch(source, ids.repeat(5, 1), weights.repeat(5, 1), prefill=prefill)


@pytest.mark.parametrize("failure_at", ["dispatch", "receive", "combine"])
def test_ep_partial_failure_quarantines_transport_and_retains_inputs(failure_at):
    calls = []
    source = torch.ones(1, 4)
    ids, scores = torch.zeros(1, 2, dtype=torch.int64), torch.ones(1, 2)

    def fail():
        calls.append("failed")
        raise RuntimeError("peer stopped")

    def dispatch(*args, **kwargs):
        if failure_at == "dispatch":
            fail()
        return source, [1], object(), None, fail if failure_at == "receive" else lambda: None

    def combine(*args, **kwargs):
        fail()

    buffer = SimpleNamespace(
        dispatch=dispatch, combine=combine, low_latency_dispatch=dispatch, low_latency_combine=combine
    )
    event = SimpleNamespace(query=lambda: True)
    driver = ExpertTransfer(
        buffer,
        buffer,
        experts=8,
        prefill_limit=8,
        decode_limit=8,
        num_sms=0,
        round_scales=False,
        capture=lambda: None,
        runtime=SimpleNamespace(record_event=lambda: event),
    )
    with pytest.raises(RuntimeError, match="peer stopped"):
        exchange = driver.dispatch(source, ids, scores, prefill=False, overlap=True)
        exchange.wait()
        driver.combine(source, exchange, prefill=False, overlap=True)
    assert driver.failure is not None
    assert source is driver.failure[1] or source is driver.failure[1][0]
    with pytest.raises(RuntimeError, match="quarantined"):
        driver.dispatch(source, ids, scores, prefill=False)
    assert calls == ["failed"]


@pytest.mark.parametrize("prefill", [False, True])
def test_ep_contribution_failure_retains_completed_and_inflight_buffers(prefill):
    calls, submitted_weights = [], []
    first_output = torch.ones(1, 4)

    def combine(*args, **kwargs):
        calls.append(args)
        if not prefill:
            submitted_weights.append(args[2])
        if len(calls) == 2:
            raise RuntimeError("second contribution failed")
        return (first_output, None, None)

    buffer = SimpleNamespace(
        dispatch=lambda: None, combine=combine, low_latency_dispatch=lambda: None, low_latency_combine=combine
    )
    transfer = ExpertTransfer(
        buffer,
        buffer,
        experts=8,
        prefill_limit=8,
        decode_limit=8,
        num_sms=0,
        round_scales=False,
        separate_contributions=True,
        runtime=object(),
        capture=lambda: None,
    )
    handle = SimpleNamespace(recv_src_metadata=torch.zeros(1, 4, dtype=torch.int32))
    exchange = ExpertDispatch(None, torch.tensor([[0, 4]]), torch.tensor([[0.5, 0.5]]), None, handle, None)
    inputs = torch.ones(2, 1, 4)
    with pytest.raises(RuntimeError, match="second contribution"):
        transfer.combine(inputs, exchange, prefill=prefill)
    retained = transfer.failure[1]
    assert any(item is first_output for item in retained)
    assert all(any(item is weights for item in retained) for weights in submitted_weights)
    with pytest.raises(RuntimeError, match="quarantined"):
        transfer.combine(inputs, exchange, prefill=prefill)
    assert len(calls) == 2


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3next", "deepseek2", "glm5_next"])
@pytest.mark.parametrize("prefill", [False, True])
def test_model_recipe_uses_bound_ep_and_preserves_shared_experts(family, prefill, monkeypatch):
    from importlib import import_module
    from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

    module = import_module(f"lightllm.models.{family}.layer_infer.transformer_layer_infer")
    classes = dict(
        qwen3_moe="Qwen3MOETransformerLayerInfer",
        qwen3next="Qwen3NextTransformerLayerInfer",
        deepseek2="Deepseek2TransformerLayerInfer",
        glm5_next="Glm5NextTransformerLayerInfer",
    )
    layer = getattr(module, classes[family]).__new__(getattr(module, classes[family]))
    layer.embed_dim_, layer.layer_num_, layer.n_shared_experts = 4, 3, 1
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state, captured = SimpleNamespace(is_prefill=prefill), []
    callback = captured.append
    monkeypatch.setattr(module, "get_moe_capture_callback", lambda s, i: callback if s is state and i == 3 else None)

    def run(x, logits, *, is_prefill, capture):
        assert x.data_ptr() == inputs.data_ptr() and is_prefill == prefill and capture is callback
        torch.testing.assert_close(logits, inputs[:, :2])
        capture("routed")
        return x * 2

    layer.bound_ep = SimpleNamespace(run=run)
    shared = torch.ones_like(inputs)
    layer._compute_shared_expert = lambda *a: shared.clone()
    layer._ffn_tp = lambda *a: shared.clone()
    monkeypatch.setattr(LlamaTransformerLayerInfer, "_ffn_tp", lambda *a: shared.clone())
    weight = SimpleNamespace(
        moe_gate=SimpleNamespace(mm=lambda x: x[:, :2], data_type_=inputs.dtype), num_fused_shared_experts=0
    )
    method = layer._moe_ffn if family == "glm5_next" else layer._moe_ffn_edp
    result = method(inputs, state, weight)
    torch.testing.assert_close(result, inputs * 2 + (0 if family == "qwen3_moe" else shared))
    assert captured == ["routed"]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("round_scales", [False, True])
def test_ep_two_rank_prefill_decode_overlap_and_graph(round_scales, tmp_path):
    _run_ep_workers(round_scales, tmp_path)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("round_scales", [False, True])
def test_ep_rounded_contributions_two_rank_overlap_and_graph(round_scales, tmp_path):
    _run_ep_workers(round_scales, tmp_path, round_intermediates=True)


def _run_ep_workers(round_scales, tmp_path, round_intermediates=False):
    if importlib.util.find_spec("deep_ep") is None or importlib.util.find_spec("deep_gemm") is None:
        pytest.skip("DeepEP and DeepGEMM required")
    context = mp.spawn(
        _ep_worker, args=(str(tmp_path / "rendezvous"), round_scales, round_intermediates), nprocs=2, join=False
    )
    try:
        for _ in range(120):
            if context.join(timeout=5):
                return
        raise TimeoutError("two-rank EP validation did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)


@torch.no_grad()
def _ep_worker(rank, rendezvous, round_scales, round_intermediates=False):
    import deep_ep
    from lightllm.common.execution.planner.planner import Planner
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
    from lightllm.common.quantization.quantize_method import WeightPack
    from lightllm.distributed import dist_group_manager
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import set_env_start_args
    from unit_tests.common.execution.test_moe_compute import quantize_blocks, route_reference

    from lightllm.utils import dist_utils

    for name, value in (
        ("global_rank", rank),
        ("global_world_size", 2),
        ("current_device_id", rank),
        ("dp_size", 1),
        ("dp_world_size", 2),
        ("global_dp_rank", 0),
        ("current_rank_in_dp", rank),
        ("current_rank_in_node", rank),
        ("node_world_size", 2),
        ("dp_rank_in_node", 0),
    ):
        getattr(dist_utils, "set_" + name)(value)
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    set_env_start_args(StartArgs(data_type="bf16", enable_ep_moe=True))
    dist.init_process_group(
        "nccl", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    prefill = decode = None
    try:
        dist.barrier(device_ids=[rank])
        spec = MoeSpec(
            4096,
            512,
            8,
            4 if round_intermediates else 2,
            torch.bfloat16,
            renormalize=True,
            routed_scale=1.3,
            round_intermediates=round_intermediates,
        )
        prefill = deep_ep.ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=64,
            hidden=4096,
            num_topk=spec.top_k,
            use_fp8_dispatch=True,
            allow_multiple_reduction=True,
            explicitly_destroy=True,
        )
        rdma_bytes = max(deep_ep.Buffer.get_low_latency_rdma_size_hint(64, 4096, 2, 8), 128 * 1024 * 1024)
        decode = deep_ep.Buffer(
            dist.group.WORLD,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=4,
            explicitly_destroy=True,
        )
        workspace = decode.get_local_buffer_tensor(torch.uint8, use_rdma_buffer=True)
        workspaces = tuple(workspace.chunk(2))
        torch.manual_seed(871)
        gate_up, a_scale, ref_a = quantize_blocks(torch.randn(8, 1024, 4096, device="cuda") * 0.02)
        down, b_scale, ref_b = quantize_blocks(torch.randn(8, 4096, 512, device="cuda") * 0.02)
        local = slice(rank * 4, (rank + 1) * 4)
        weight = FusedMoeWeight.__new__(FusedMoeWeight)
        weight.w13 = WeightPack(gate_up[local].contiguous(), a_scale[local].contiguous())
        weight.w2 = WeightPack(down[local].contiguous(), b_scale[local].contiguous())
        weight.w13.load_ok = weight.w2.load_ok = [True, True, True]
        weight.w1_list, weight.w2_list, weight.w3_list = [weight.w13], [weight.w2], []
        weight.hidden_size, weight.moe_intermediate_size, weight.n_routed_experts = 4096, 512, 8
        weight.num_experts_per_tok, weight.norm_topk_prob = spec.top_k, True
        weight.scoring_func, weight.routed_scaling_factor = "softmax", 1.3
        weight.n_group, weight.topk_group = 0, 0
        weight.enable_ep_moe, weight.global_rank_, weight.global_world_size = True, rank, 2
        weight.e_score_correction_bias = weight.per_expert_scale = None
        weight.redundancy_expert_ids_tensor = torch.empty(0, dtype=torch.int64, device="cuda")
        weight.routed_expert_counter_tensor = torch.zeros(8, dtype=torch.int64, device="cuda")
        weight.auto_update_redundancy_expert = False
        weight.quant_method = SimpleNamespace(method_name="fp8w8a8-b128-deepgemm")
        recipe = SimpleNamespace(layer_num_=3, is_moe=True)
        dist_group_manager.ep_buffer, dist_group_manager.ep_low_latency_buffer = prefill, decode
        dist_group_manager.groups = [None, None]
        dist_group_manager.ll_num_tokens = dist_group_manager.ll_decode_num_tokens = 64
        dist_group_manager.ep_num_sms = 0
        # Match the explicit scale policy during startup, then forbid runtime lookups.
        from lightllm.utils import envs_utils

        original_scale_policy = envs_utils.get_env_use_ue8m0_scales
        envs_utils.get_env_use_ue8m0_scales = lambda: round_scales
        second_recipe = SimpleNamespace(layer_num_=7, is_moe=True)
        plan = bind_moe({3: spec, 7: spec}, [SimpleNamespace(experts=weight)] * 2, [recipe, second_recipe])
        assert second_recipe.bound_ep.transfer is recipe.bound_ep.transfer
        envs_utils.get_env_use_ue8m0_scales = original_scale_policy
        assert not plan.legacy and recipe.bound_ep is plan.bindings[0][1]
        runner = recipe.bound_ep
        assert isinstance(runner, ExpertRunner)
        assert runner.compute.workspaces[0].data_ptr() == workspaces[0].data_ptr()
        dist_group_manager.ep_buffer = dist_group_manager.ep_low_latency_buffer = None
        dist_group_manager.get_deep_ep_prefill_moe_workspace = lambda *a: (_ for _ in ()).throw(
            AssertionError("runtime manager query")
        )
        weight.w13 = weight.w2 = weight.quant_method = None

        def quantized(x, *, transport=False):
            x = x.float().reshape(*x.shape[:-1], -1, 128)
            amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-4 if transport else 1e-10)
            scale = amax / 448
            if round_scales:
                scale = scale.log2().ceil().exp2()
            # DeepEP computes 448 / amax then multiplies. Dividing by the rounded
            # dequantization scale changes E4M3 tie rounding. Use FP64 division
            # followed by FP32 storage for the independent CUDA-float reference.
            normalized = x * (448 / amax.double()).float() if transport and not round_scales else x / scale
            return (normalized.clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).flatten(-2)

        def reference(x, logits, prefill):
            scores, ids = route_reference(spec, logits, None, None, None, 0)
            output = torch.zeros_like(x, dtype=torch.float32)
            source = quantized(x, transport=not prefill)
            for expert in range(8):
                rows, slot = torch.where(ids == expert)
                if not rows.numel():
                    continue
                projection = (source[rows] @ ref_a[expert].T).bfloat16()
                gate, up = projection.chunk(2, -1)
                activated = (torch.nn.functional.silu(gate.float()).bfloat16() * up).bfloat16()
                branch = (quantized(activated) @ ref_b[expert].T).bfloat16()
                weighted = branch.float() * scores[rows, slot, None]
                if round_intermediates:
                    weighted = weighted.bfloat16().float()
                output.index_add_(0, rows, weighted)
            return output.bfloat16()

        def check(out, x, logits, *, prefill=True):
            target = reference(x, logits, prefill)
            error = out.float() - target.float()
            rms = (error.square().mean() / target.float().square().mean()).sqrt()
            peak = error.abs().max() / target.abs().max()
            failed = (~((rms < 0.015) & (peak < 1 / 32))).int()
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            assert not failed.item(), f"rank={rank}, RMS={rms.item()}, peak={peak.item()}"

        torch.manual_seed(90 + rank)
        x = torch.randn(7 + rank * 4, 4096, device="cuda", dtype=torch.bfloat16) * 0.2
        logits = torch.randn(len(x), 8, device="cuda")
        if round_intermediates:
            _check_ep_contribution_rounding(runner, x, rank, decode)
        from lightllm.common.basemodel.moe_route_info_manager import MoeRouteInfoManager

        route_manager = MoeRouteInfoManager(1, spec.top_k, 1, {3: 0})
        route_manager.init_capture_buffer(64)
        route_binding = route_manager.bind_capture()
        route_rows = torch.arange(len(x), device="cuda")
        record = route_binding.bind_batch(route_rows)[3]

        def check_routes(rows=route_rows):
            torch.cuda.synchronize()
            expected_ids = route_reference(spec, logits, None, None, None, 0)[1].cpu().to(torch.uint8)
            actual_ids = route_binding.buffer[rows.cpu(), 0]
            failed = torch.tensor(int(not torch.equal(actual_ids, expected_ids)), device="cuda")
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            assert not failed.item(), "route recording changed logical expert IDs or token slots"

        check(runner.run(x, logits, is_prefill=True, capture=record), x, logits)
        decode.clean_low_latency_buffer(64, 4096, 8)
        check(runner.run(x, logits, is_prefill=False, capture=record), x, logits, prefill=False)
        # A rank may own no selected experts, while both ranks still send tokens.
        cold_logits = logits.clone()
        cold_logits[:, 4:] = -100
        check(runner.run(x, cold_logits, is_prefill=True), x, cold_logits)
        decode.clean_low_latency_buffer(64, 4096, 8)
        for _ in range(2):
            runner.run(x, logits, is_prefill=False, capture=record)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = runner.run(x, logits, is_prefill=False, capture=record)
        x.add_(0.03)
        route_rows.add_(16)
        logits.neg_()
        graph.replay()
        check(result, x, logits, prefill=False)
        check_routes()
        # Two in-flight microbatches exercise the same split steps used by recipes.
        for is_prefill in (True, False):
            if not is_prefill:
                decode.clean_low_latency_buffer(64, 4096, 8)
            sources = (x, x.mul(0.7))
            dispatches = []
            for lane, source in enumerate(sources):
                rows = torch.arange(len(x), device="cuda") + lane * 16
                callback = route_binding.bind_batch(rows)[3]
                if is_prefill:
                    scores, ids, encoded = runner.select_experts_and_quant_input(source, logits, capture=callback)
                    dispatches.append(runner.dispatch(encoded, ids, scores, runner.capture()))
                else:
                    dispatches.append(runner.low_latency_dispatch(source, logits, capture=callback))
                    # Recipes consume each receive hook before reusing the low-latency
                    # dispatch slot; the expert results remain live through combine.
                    dispatches[-1][-1]()
            results = []
            for lane, (inputs, counts_or_ids, scores, counts_or_weights, handle, ready) in enumerate(dispatches):
                ready()
                if is_prefill:
                    output = runner.prefilled_group_gemm(
                        counts_or_weights,
                        handle.num_unaligned_recv_tokens_per_expert,
                        handle.recv_src_metadata,
                        inputs,
                        counts_or_ids,
                        scores,
                        microbatch_index=lane,
                    )
                    combined, complete = runner.combine(output, handle, runner.capture())
                else:
                    output = runner.masked_group_gemm(inputs, counts_or_ids, spec.dtype, runner.expected_m(len(x)))
                    combined, complete = runner.low_latency_combine(output, scores, counts_or_weights, handle)
                complete()
                results.append(combined.clone())
            for lane, (out, source) in enumerate(zip(results, sources)):
                check(out, source, logits, prefill=is_prefill)
                check_routes(torch.arange(len(x), device="cuda") + lane * 16)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])
    finally:
        torch.cuda.synchronize()
        if decode is not None:
            decode.clean_low_latency_buffer(64, 4096, 8)
        torch.cuda.synchronize()
        if prefill is not None:
            prefill.destroy()
        if decode is not None:
            decode.destroy()
        dist.destroy_process_group()


def _check_ep_contribution_rounding(runner, source, rank, decode):
    from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import (
        ep_compact_metadata,
        ep_gather_chunk,
    )

    ids = torch.tensor([[0, 4, 1, 5]], device="cuda").repeat(len(source), 1)
    weights = torch.tensor([[0.333984375, 0.333984375, 1, 1]], device="cuda").repeat(len(source), 1)
    values = torch.tensor([3, -2, 0, 0, 3, -0.5, 0, 0], device="cuda", dtype=torch.bfloat16)
    weighted = values[ids].float() * weights
    target = weighted.bfloat16().float().sum(-1).bfloat16()
    assert not torch.equal(target, weighted.sum(-1).bfloat16())
    for prefill in (True, False):
        if not prefill:
            decode.clean_low_latency_buffer(64, 4096, 8)
        encoded = runner.compute.quantize(source) if prefill else source
        exchange = runner.transfer.dispatch(encoded, ids, weights, prefill=prefill)
        exchange.wait()
        if prefill:
            expert_values = values[rank * 4 : (rank + 1) * 4].repeat_interleave(
                torch.tensor(exchange.counts, device="cuda")
            )
            expanded = expert_values[:, None].expand(-1, source.shape[1]).contiguous()
            contributions = torch.zeros(
                4, exchange.recv_src_metadata.shape[0], source.shape[1], device="cuda", dtype=torch.bfloat16
            )
            ep_gather_chunk(expanded, 0, exchange.weights, exchange.recv_src_metadata, contributions)
            ep_compact_metadata(exchange.recv_src_metadata)
        else:
            contributions = values[rank * 4 : (rank + 1) * 4, None, None].expand(*exchange.inputs[0].shape).contiguous()
        output, ready = runner._combine(contributions, exchange, prefill=prefill)
        ready()
        failed = torch.tensor(int(not torch.equal(output, target[:, None].expand_as(output))), device="cuda")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        assert not failed.item(), f"EP lost per-expert BF16 rounding, rank={rank}, prefill={prefill}"
