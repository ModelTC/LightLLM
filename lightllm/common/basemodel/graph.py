"""持有一次捕获的输入、provider 任务和完成事件；bucket 策略仍由模型决定。"""

from lightllm.platform import get_device_runtime


class GraphExecution:
    def __init__(self, graph, states=(), outputs=(), *, runtime=None):
        self.graph = graph
        self.states = tuple(states)
        self.outputs = tuple(outputs)
        self.runtime = runtime if runtime is not None else get_device_runtime(self.states[0].input_ids.device)
        self.bindings = []
        self.workspaces = []
        self.generations = []
        try:
            for state in self.states:
                manager = state.att_state_manager
                self.generations.append((manager, manager.storage_generation))
                for group_id, context in state.attention_contexts.items():
                    backend = state.attention_groups[group_id].backend_for(context.capacity.phase)
                    binding = backend.bind_graph(context)
                    if binding is not None:
                        self.bindings.append(binding)
                    if context.workspace not in self.workspaces:
                        self.workspaces.append(context.workspace)
                manager.register_graph(self)
            for workspace in self.workspaces:
                workspace.freeze()
        except BaseException:
            self.close()
            raise

    def replay(self, *states):
        if self.graph is None:
            raise RuntimeError("Graph execution is closed")
        if states and len(states) != len(self.states):
            raise ValueError("Graph execution lane count changed")
        for manager, generation in self.generations:
            if manager.storage_generation != generation:
                raise RuntimeError("attention storage changed; rebind layers and recapture Graph")
        for binding in self.bindings:
            binding.wait_for_update()
        if not states:
            for workspace in self.workspaces:
                workspace.begin()
        for target, source in zip(self.states, states):
            target.copy_for_cuda_graph(source)
        for binding in self.bindings:
            binding.before_replay()
        self.graph.replay()
        # 必须先提交辅助更新，再记录完成事件：Graph 本身可能正在等待更新流。
        events = []
        for binding in self.bindings:
            events.extend(binding.after_submit())
        for workspace in self.workspaces:
            workspace.finish(dependencies=events)
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs

    def close(self):
        if self.graph is None:
            return
        for binding in self.bindings:
            binding.wait_for_update()
        self.runtime.synchronize()
        for binding in self.bindings:
            binding.close()
        self.graph.reset()
        self.graph = None
        self.bindings.clear()
        self.workspaces.clear()
        self.states = self.outputs = ()
        self.generations.clear()
