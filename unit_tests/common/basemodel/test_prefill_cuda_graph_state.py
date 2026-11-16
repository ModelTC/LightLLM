from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.hidden_collector as hidden_collector_module
from lightllm.common.basemodel.hidden_collector import LayerHiddenCollector
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.prefill_cuda_graph import PrefillCudaGraph, PrefillGraphExecution
from lightllm.platform import DeviceRuntime
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.execution import ExecutionScope


class _GraphInferState:
    def capture_manifest(self, *, segmented):
        from lightllm.common.attention.api.graph import CaptureManifest

        return CaptureManifest()

    def __init__(self, hidden_collector):
        self.hidden_collector = hidden_collector
        self.input_ids = torch.empty(4, dtype=torch.int64)
        self.att_state_manager = AttentionStateManager(request_capacity=2)
        self.attention = SimpleNamespace(retain=lambda *args: None, binding=None)
        self.validated_from = None
        self.replayed_with = None

    def prefill_input_copies(self, new_infer_state):
        self.validated_from = new_infer_state
        return []

    def prefill_replay(self, new_infer_state):
        self.replayed_with = new_infer_state


class _IdentityPreInfer:
    @staticmethod
    def _tpsp_allgather(input, infer_state):
        del infer_state
        return input


def test_prefill_output_reader_blocks_reuse_until_completion():
    reader = SimpleNamespace(synchronize=lambda: events.append("reader complete"))
    events = []
    runtime = DeviceRuntime("cpu")
    scope = ExecutionScope(runtime)
    state = _GraphInferState(hidden_collector=object())
    execution = PrefillGraphExecution(state, [], [], state.hidden_collector, runtime, scope)
    execution.record_output_use(reader)
    scope.synchronize()
    assert events == ["reader complete"]
    execution.invalid = True
    with pytest.raises(RuntimeError, match="closed or invalid"):
        execution.record_output_use(reader)


def _create_layer_collector(monkeypatch):
    monkeypatch.setattr(
        hidden_collector_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_draft_model_dir=["/models/draft"]),
    )
    monkeypatch.setattr(
        hidden_collector_module.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"target_layer_ids": [0]}, {}),
    )
    model = SimpleNamespace(layers_num=2, pre_infer=_IdentityPreInfer())
    return LayerHiddenCollector(model=model)


def test_replay_restores_captured_hidden_state_into_request_collector(monkeypatch):
    graph_collector = _create_layer_collector(monkeypatch)
    graph_collector.add(layer_index=0, hidden=torch.randn(2, 3))
    request_collector = graph_collector.new_instance()
    graph_infer_state = _GraphInferState(hidden_collector=graph_collector)
    request_infer_state = SimpleNamespace(
        input_ids=torch.empty(4, dtype=torch.int64),
        hidden_collector=request_collector,
        att_state_manager=graph_infer_state.att_state_manager,
        attention=SimpleNamespace(retain=lambda *args: None, binding=None),
    )
    graph_output = torch.randn(2, 3)
    prefill_graph = PrefillCudaGraph.__new__(PrefillCudaGraph)
    prefill_graph.execution = ExecutionScope(DeviceRuntime("cpu"))
    prefill_graph.graph = {
        4: PrefillGraphExecution(
            graph_infer_state, [], [graph_output], graph_collector, DeviceRuntime("cpu"), prefill_graph.execution
        )
    }

    outputs = prefill_graph._replay(input_tensors=[], infer_state=request_infer_state)

    assert len(outputs) == 1
    assert outputs[0] is graph_output
    assert request_infer_state.hidden_collector is request_collector
    assert request_collector.layer_hiddens is not graph_collector.layer_hiddens
    assert request_collector.layer_hiddens[0] is graph_collector.layer_hiddens[0]
    assert graph_infer_state.validated_from is request_infer_state
    assert graph_infer_state.replayed_with is request_infer_state


def test_first_replay_replaces_capture_collector_with_runtime_instance(monkeypatch):
    graph_collector = _create_layer_collector(monkeypatch)
    graph_collector.add(layer_index=0, hidden=torch.randn(2, 3))
    graph_infer_state = _GraphInferState(hidden_collector=graph_collector)
    graph_output = torch.randn(2, 3)
    prefill_graph = PrefillCudaGraph.__new__(PrefillCudaGraph)
    prefill_graph.execution = ExecutionScope(DeviceRuntime("cpu"))
    prefill_graph.graph = {
        4: PrefillGraphExecution(
            graph_infer_state, [], [graph_output], graph_collector, DeviceRuntime("cpu"), prefill_graph.execution
        )
    }

    prefill_graph._replay(input_tensors=[], infer_state=graph_infer_state)

    assert graph_infer_state.hidden_collector is not graph_collector
    assert graph_infer_state.hidden_collector.layer_hiddens[0] is graph_collector.layer_hiddens[0]


@pytest.mark.parametrize("between_segments", [False, True])
def test_failed_prefill_capture_releases_segments(between_segments):
    events = []

    class Runtime(DeviceRuntime):
        def create_graph(self):
            return SimpleNamespace(reset=lambda: events.append("reset"))

        @contextmanager
        def capture_graph(self, graph, **kwargs):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        def synchronize(self):
            events.append("synchronize")

    prefill_graph = PrefillCudaGraph.__new__(PrefillCudaGraph)
    prefill_graph.execution = ExecutionScope(DeviceRuntime("cpu"))
    prefill_graph.graph = {}
    prefill_graph.mempool = None
    prefill_graph.runtime = Runtime("cpu")
    state = InferStateInfo()
    state.input_ids = torch.empty(4, dtype=torch.int64)
    state.att_state_manager = AttentionStateManager(request_capacity=2)
    state.att_state_manager.finish_state_registration()
    state.b_req_idx = torch.zeros(1, dtype=torch.int32)
    persistent = torch.ones(2)
    state.att_state_manager.snapshot_capture_state = lambda *args: persistent.clone()
    state.att_state_manager.restore_capture_state = lambda saved: persistent.copy_(saved)

    def forward(inputs, state):
        persistent.add_(3)
        if between_segments:
            state.prefill_cuda_graph_get_current_capture_graph().__exit__(None, None, None)
        raise ValueError("failed segment")

    with pytest.raises(ValueError, match="failed segment"):
        prefill_graph._capture_prefill(forward, [], state)
    assert events == ["enter", "exit", "synchronize", "reset"]
    assert not prefill_graph.graph
    assert not state.prefill_cuda_graph_exe_list
    torch.testing.assert_close(persistent, torch.ones(2))


def test_prefill_capture_restores_state_before_live_replay():
    persistent = torch.ones(2)
    state = InferStateInfo()
    state.input_ids = torch.zeros(4, dtype=torch.int64)
    state.att_state_manager = AttentionStateManager(request_capacity=2)
    state.att_state_manager.snapshot_capture_state = lambda *args: persistent.clone()
    state.att_state_manager.restore_capture_state = lambda saved: persistent.copy_(saved)
    state.hidden_collector = SimpleNamespace(release_graph_tensor_ownership=lambda: (0, 0))
    runtime = DeviceRuntime("cpu")
    runtime.create_graph = lambda: SimpleNamespace(reset=lambda: None)

    @contextmanager
    def capture(*args, **kwargs):
        yield

    runtime.capture_graph = capture
    runner = PrefillCudaGraph.__new__(PrefillCudaGraph)
    runner.execution = ExecutionScope(DeviceRuntime("cpu"))
    runner.graph, runner.mempool, runner.runtime = {}, None, runtime

    def forward(inputs, state):
        persistent.add_(5)
        return []

    def replay(inputs, state):
        torch.testing.assert_close(persistent, torch.ones(2))
        persistent.add_(1)

    runner.replay = replay
    assert runner._capture_prefill(forward, [], state) == []
    torch.testing.assert_close(persistent, torch.full((2,), 2.0))
    runner.close()


@pytest.mark.parametrize("invalid", ["count", "shape", "dtype", "hidden", "overlap", "manager"])
def test_prefill_replay_validates_entire_manifest_before_writes(invalid):
    target, incoming = InferStateInfo(), InferStateInfo()
    target.att_state_manager = AttentionStateManager(request_capacity=2)
    incoming.att_state_manager = target.att_state_manager
    target.input_ids, incoming.input_ids = torch.zeros(4, dtype=torch.int64), torch.ones(4, dtype=torch.int64)
    target.position_ids = torch.zeros(4, dtype=torch.int64)
    incoming.position_ids = torch.ones(4, dtype=torch.int64)
    target.hidden_collector = object()
    first, second = torch.zeros(4), torch.zeros(4)
    sources = [torch.ones(4), torch.ones(4)]
    if invalid == "count":
        sources.pop()
    elif invalid == "shape":
        sources[-1] = torch.ones(5)
    elif invalid == "dtype":
        sources[-1] = sources[-1].double()
    elif invalid == "hidden":
        incoming.mtp_draft_input_hiddens = torch.ones(4, 2)
    elif invalid == "overlap":
        sources[-1] = first
    else:
        incoming.att_state_manager = AttentionStateManager(request_capacity=2)
    runner = PrefillCudaGraph.__new__(PrefillCudaGraph)
    runner.execution = ExecutionScope(DeviceRuntime("cpu"))
    execution = PrefillGraphExecution(target, [first, second], [], object(), DeviceRuntime("cpu"), runner.execution)
    runner.graph = {4: execution}
    with pytest.raises(ValueError, match="count|shape|dtype|presence|overwrite|manager"):
        runner._replay(sources, incoming)
    assert not first.count_nonzero() and not second.count_nonzero()
    assert not target.position_ids.count_nonzero() and not target.input_ids.count_nonzero()
    assert not execution.invalid


def test_prefill_manifest_allows_current_request_shape_to_change():
    target, incoming = InferStateInfo(), InferStateInfo()
    for state, requests in ((target, 2), (incoming, 3)):
        state.input_ids = torch.arange(4)
        state.position_ids = torch.arange(4)
        state.mem_index = torch.arange(4)
        state.b_req_idx = torch.arange(requests)
        state.b_q_start_loc = torch.arange(requests)
    copies = target.prefill_input_copies(incoming)
    assert len(copies) == 3
