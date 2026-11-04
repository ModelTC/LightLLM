from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest
import torch

from lightllm.core.contracts.compute import ComputeResult
from lightllm.core.contracts.device import DeviceCaps
from lightllm.core.contracts.state import FormatSpec
from lightllm.core.device.context import ExecutionContext, create_context
from lightllm.core.device.graph import GraphExecution
from lightllm.core.formats.weights.view import WeightView
from lightllm.core.registry import ProviderRegistry


def test_execution_factory_keeps_injected_platform_and_rejects_cuda_defaults(monkeypatch):
    runtime = SimpleNamespace(caps=DeviceCaps("cuda", 0, (9, 0), "other"))
    providers = ProviderRegistry()
    context = ExecutionContext(runtime, providers)
    monkeypatch.setenv("LIGHTLLM_EXECUTION_FACTORY", "test_platform_factory:create")
    monkeypatch.setitem(sys.modules, "test_platform_factory", SimpleNamespace(create=lambda **kwargs: context))
    assert create_context(0) is context
    assert create_context(0, device=runtime, providers=providers).device is runtime
    with pytest.raises(ValueError, match="explicit provider registry"):
        create_context(0, device=runtime)
    assert not runtime.caps.is_nvidia
    monkeypatch.delenv("LIGHTLLM_EXECUTION_FACTORY")
    with pytest.raises(ValueError, match="requires an execution factory"):
        create_context(0, config=SimpleNamespace(hardware_platform="other"))


def test_distributed_startup_uses_runtime_device_backend_and_options(monkeypatch):
    from lightllm.utils import dist_utils

    events = []
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        distributed_backend="gloo",
        set_device=lambda index: events.append(("device", index)),
        process_group_options=lambda: {"device_id": torch.device("cpu")},
    )
    monkeypatch.setattr(dist_utils, "set_environ", lambda name, value: monkeypatch.setenv(name, str(value)))
    monkeypatch.setattr(dist_utils, "_init_nccl_env", lambda: pytest.fail("CUDA initialization"))
    monkeypatch.setattr(
        dist_utils.dist, "init_process_group", lambda backend, **kwargs: events.append((backend, kwargs))
    )
    monkeypatch.setattr(dist_utils.dist, "all_reduce", lambda value: events.append(("reduce", value.device)))
    dist_utils.init_distributed_env(
        dict(world_size=2, rank_id=1, args=SimpleNamespace(nnodes=1), nccl_host="localhost", nccl_port=1), runtime
    )
    assert events[0] == ("device", 1)
    assert events[1][0] == "gloo" and events[1][1]["device_id"] == torch.device("cpu")
    assert events[2] == ("reduce", torch.device("cpu"))


def test_graph_waits_before_updating_captured_inputs_and_closes_auxiliary_work():
    events = []
    graph = SimpleNamespace(replay=lambda: events.append("replay"), reset=lambda: events.append("reset"))
    state = SimpleNamespace(copy_for_cuda_graph=lambda value: events.append(("copy", value)))
    binding = SimpleNamespace(
        wait_for_update=lambda: events.append("wait"),
        before_replay=lambda: events.append("before"),
        after_replay=lambda: events.append("after"),
        close=lambda: events.append("close"),
    )
    execution = GraphExecution(graph, (state,), (binding,))
    execution.replay(13)
    assert events == ["wait", ("copy", 13), "before", "replay", "after"]
    execution.reset()
    execution.reset()
    assert events[-2:] == ["close", "reset"]
    with pytest.raises(RuntimeError, match="closed"):
        execution.replay(14)


def test_dense_prefill_receives_current_unquantized_kv():
    from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

    layer = LlamaTransformerLayerInfer.__new__(LlamaTransformerLayerInfer)
    layer.tp_q_head_num_, layer.tp_k_head_num_, layer.head_dim_ = 2, 1, 4
    received = []

    def run(inputs, metadata):
        received.append(inputs)
        return inputs.q

    layer.bound_prefill_attention = SimpleNamespace(run=run, prepare_batch=lambda batch: None)
    q, kv = torch.randn(3, 8), torch.randn(3, 2, 4)
    result = layer._context_attention_kernel(q, kv, None, None)
    assert result.data_ptr() == q.data_ptr()
    assert received[0].current_k.data_ptr() == kv[:, :1].data_ptr()
    assert received[0].current_v.data_ptr() == kv[:, 1:].data_ptr()
    torch.testing.assert_close(received[0].current_k, kv[:, :1])


def test_native_weight_finalization_precedes_compute_binding():
    from lightllm.common.basemodel.weight_binding import finalize_weights, bind_linears
    from lightllm.core.compute.linear.backend import LinearBackend
    from lightllm.core.compute.linear.spec import LinearSpec

    events = []
    tensor = torch.randn(4, 3)
    view = WeightView(
        FormatSpec("plain", torch.float32, "output_input", ("weight",), 1),
        {"weight": tensor, "scale": None, "bias": None},
        finalized=False,
    )
    weight = SimpleNamespace()
    weight.native_linear_view = lambda: (LinearSpec(3, 4, torch.float32), weight.view, "mm")
    weight.view = view

    def finalize(runtime):
        events.append("finalize")
        weight.view = replace(view, finalized=True)

    weight.finalize_weight = finalize
    layer = SimpleNamespace(projection=weight)
    runtime = SimpleNamespace(device=torch.device("cpu"), caps=DeviceCaps("cpu"))
    providers = ProviderRegistry(linear=(lambda: LinearBackend("torch.linear"),))
    with pytest.raises(ValueError, match="finalized"):
        bind_linears((layer,), torch.empty, providers=providers, runtime=runtime)
    assert not hasattr(weight, "mm")
    finalize_weights((layer,), runtime, providers)
    bind_linears((layer,), torch.empty, providers=providers, runtime=runtime)
    x = torch.randn(2, 3)
    torch.testing.assert_close(weight.mm(x), x @ tensor.T)
    assert events == ["finalize"]
    assert weight.bound_mm.weight_view is weight.view


@pytest.mark.parametrize("distribution,reductions", [("partial", 1), ("replicated", 0)])
def test_fused_output_is_reduced_exactly_once(distribution, reductions):
    calls = []
    tensor = torch.ones(2, 3)
    result = ComputeResult(tensor, distribution)
    assert result.finish(lambda value: calls.append(value) or value) is tensor
    assert len(calls) == reductions
    if distribution == "replicated":
        with pytest.raises(ValueError, match="expected sequence_shard"):
            result.finish(lambda value: pytest.fail("duplicate reduction"), "sequence_shard")


def test_transfer_matches_encoding_and_memory_path_before_submission():
    from lightllm.core.contracts.transfer import TransferRequest
    from lightllm.core.registry import TransferProvider
    from lightllm.core.transfer import CopyPlan, Transfer, TransferCompletion

    fmt = FormatSpec("int8_per_token", torch.int8, "paged", ("k", "v", "scales"), 32)
    request = TransferRequest(fmt, fmt, "device", "host", "cpu_cache")
    source, destination = object(), object()
    calls = []
    driver = SimpleNamespace(
        submit=lambda plan, **kwargs: calls.append(plan) or TransferCompletion(source, destination)
    )
    providers = ProviderRegistry(
        transfers=(TransferProvider(lambda value, caps: value == request, lambda *args: driver),)
    )
    transfer = Transfer(SimpleNamespace(caps=DeviceCaps("cpu")), providers)
    plan = CopyPlan("pack", (), request)
    completion = transfer.submit(plan)
    assert completion.source_reusable is source and completion.destination_ready is destination
    for changed in (replace(request, destination=replace(fmt, encoding="int8_group8")), replace(request, path="pd")):
        with pytest.raises(ValueError, match="unsupported transfer path"):
            transfer.submit(replace(plan, request=changed))
    assert calls == [plan]


def test_host_lengths_are_ready_before_graph_buffer_copy():
    from lightllm.core.compute.attention.dense.backend import DenseBatchMetadata

    value = torch.zeros(1, dtype=torch.int32)
    captured = DenseBatchMetadata(
        1, 1, 8, 0, value, value, value, value, value, value, value, host_seq_lengths=torch.zeros(1, dtype=torch.int32)
    )
    host = torch.empty_like(value)
    incoming = replace(captured, host_seq_lengths=host, host_ready=SimpleNamespace(synchronize=lambda: host.fill_(7)))
    captured.copy_for_graph(incoming)
    assert captured.host_seq_lengths.item() == 7


def test_decode_warmup_uses_platform_length_requirement():
    from lightllm.core.runner.warmup import WarmupRunner

    requests = SimpleNamespace(req_to_token_indexs=torch.zeros(2, 32), HOLD_REQUEST_ID=1)
    runtime = SimpleNamespace(decode_warmup_length=lambda maximum: maximum)
    runner = WarmupRunner(
        requests, None, None, lambda tokens: {}, 16, overlap_prefill=None, overlap_decode=None, runtime=runtime
    )
    batch = runner.decode_input(3, 16)
    assert batch.total_token_num == 48 and batch.b_seq_len.tolist() == [16, 16, 16]


def test_plain_weight_loader_uses_worker_target_device(monkeypatch):
    from lightllm.common.quantization.no_quant import NoQuantization

    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setenv("LIGHTLLM_TORCH_DEVICE_TYPE", "cpu")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda *args, **kwargs: pytest.fail("CUDA weight allocation"))
    packed, split = NoQuantization().create_weight(in_dim=3, out_dims=[2, 4], dtype=torch.float32, device_id=0)
    assert packed.weight.device.type == "cpu"
    assert split[1].weight.untyped_storage().data_ptr() == packed.weight.untyped_storage().data_ptr()
