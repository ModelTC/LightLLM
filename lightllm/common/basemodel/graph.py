from copy import copy
from threading import Lock

from lightllm.common.attention.api.execution import ExecutionFailure
from lightllm.common.attention.api.graph import TensorCopyPlan
from lightllm.common.execution import ExecutionScope
from lightllm.platform import get_device_runtime


class CapturedGraph:
    def __init__(self, graph, states=(), outputs=(), *, runtime=None, execution=None):
        self.graph = graph
        self.infer_states = tuple(copy(state) for state in states)
        self.outputs = tuple(outputs)
        self.runtime = runtime if runtime is not None else get_device_runtime(self.infer_states[0].input_ids.device)
        self.execution = execution
        self.generations = []
        self._lock = Lock()
        self._invalid = False
        self._copy_plan = None
        self.manifests = ()
        try:
            for state in self.infer_states:
                state.attention = state.attention.capture()
                manager = state.att_state_manager
                self.generations.append((manager, manager.storage_generation))
                for context in state.attention.contexts.values():
                    if self.execution is None:
                        self.execution = context.workspace.execution
                    if context.workspace.execution is not self.execution:
                        raise ValueError("captured contexts must share the model execution scope")
                    context.workspace.freeze()
                manager.register_graph(self)
            if self.execution is None:
                self.execution = ExecutionScope(self.runtime)
            self._copy_plan = TensorCopyPlan(
                [target for state in self.infer_states for target, _ in state.graph_input_copies(state)]
            )
            self.manifests = tuple(state.capture_manifest() for state in self.infer_states)
            for manifest, output in zip(self.manifests, self.outputs):
                manifest.add("output", "generated", output)
        except BaseException:
            self.runtime.synchronize()
            self.graph.reset()
            self.graph = None
            raise

    def replay(self, *states):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Graph already has an active submission")
        try:
            return self._replay(states)
        finally:
            self._lock.release()

    def _replay(self, states):
        if self.graph is None:
            raise RuntimeError("Graph execution is closed")
        if self._invalid:
            raise RuntimeError("Graph execution is invalid after a failed submission")
        if states and len(states) != len(self.infer_states):
            raise ValueError("Graph execution lane count changed")
        for manager, generation in self.generations:
            manager.check_execution_health()
            if manager.storage_generation != generation:
                raise RuntimeError("attention storage changed; rebind layers and recapture Graph")
        for state in self.infer_states:
            state.attention.check_binding()
        for manifest in self.manifests:
            manifest.check()
        if any(source.att_state_manager is not manager for source, (manager, _) in zip(states, self.generations)):
            raise ValueError("Graph input belongs to a different attention state manager")
        copies = (
            self._copy_plan.validate(
                [
                    pair
                    for target, source in zip(self.infer_states, states)
                    for pair in target.graph_input_copies(source)
                ]
            )
            if states
            else ()
        )
        submitted = False
        preparing = False
        try:
            with self.execution.join(retain=(self, *states)):
                for state in self.infer_states:
                    state.attention.check_binding()
                preparing = True
                for target, source in copies:
                    target.copy_(source, non_blocking=True)
                for target, source in zip(self.infer_states, states):
                    target.attention.plan_replay(source.attention)
                for manifest in self.manifests:
                    manifest.check(prepared=True)
                submitted = True
                for state in states:
                    state.attention.state_may_be_modified = True
                self.graph.replay()
        except BaseException as error:
            if preparing:
                self._invalid = True
            if submitted and self.infer_states:
                failure = ExecutionFailure(
                    self.infer_states[0].att_state_role,
                    tuple(state.attention.batch.b_req_idx for state in (states or self.infer_states)),
                    "graph submission",
                )
                for manager, _ in self.generations:
                    manager.invalidate_execution(failure)
                error.attention_failure = failure
            raise
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs

    def record_output_use(self, event):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot register an output reader during Graph submission")
        try:
            if self.graph is None or self._invalid:
                raise RuntimeError("Graph execution is closed or invalid")
            self.execution.record_reader(event)
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close Graph during an active submission")
        try:
            if self.graph is None:
                return
            with self.execution.join():
                self.runtime.synchronize()
                self.graph.reset()
                self.graph = None
                self.infer_states = self.outputs = ()
                self.generations.clear()
                self._copy_plan = None
                self.manifests = ()
        finally:
            self._lock.release()
