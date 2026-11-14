from contextlib import ExitStack

"""持有一次捕获的输入、provider 任务和完成事件；bucket 策略仍由模型决定。"""

from lightllm.platform import get_device_runtime


class CapturedGraph:
    def __init__(self, graph, states=(), outputs=(), *, runtime=None):
        self.graph = graph
        self.infer_states = tuple(states)
        self.outputs = tuple(outputs)
        self.runtime = runtime if runtime is not None else get_device_runtime(self.infer_states[0].input_ids.device)
        self.hooks = []
        self.workspaces = []
        self.generations = []
        try:
            for state in self.infer_states:
                manager = state.att_state_manager
                self.generations.append((manager, manager.storage_generation))
                for group_id, context in state.attention.contexts.items():
                    backend = state.attention.groups[group_id][context.capacity.phase]
                    binding = backend.create_capture_hooks(context)
                    if binding is not None:
                        self.hooks.append(binding)
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
        if states and len(states) != len(self.infer_states):
            raise ValueError("Graph execution lane count changed")
        for manager, generation in self.generations:
            if manager.storage_generation != generation:
                raise RuntimeError("attention storage changed; rebind layers and recapture Graph")
        copies = [
            pair for target, source in zip(self.infer_states, states) for pair in target.graph_input_copies(source)
        ]
        for binding in self.hooks:
            binding.wait_for_update()
        events = []
        with ExitStack() as leases:
            for workspace in self.workspaces:
                leases.enter_context(workspace.lease(events))
            for target, source in copies:
                target.copy_(source, non_blocking=True)
            for target, source in zip(self.infer_states, states):
                target.attention.plan_replay(source.attention)
            for hooks in self.hooks:
                hooks.before_replay()
            self.graph.replay()
            for hooks in self.hooks:
                events.extend(hooks.after_submit())
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs

    def close(self):
        if self.graph is None:
            return
        for binding in self.hooks:
            binding.wait_for_update()
        self.runtime.synchronize()
        for binding in self.hooks:
            binding.close()
        self.graph.reset()
        self.graph = None
        self.hooks.clear()
        self.workspaces.clear()
        self.infer_states = self.outputs = ()
        self.generations.clear()
