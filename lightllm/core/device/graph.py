class GraphExecution:
    def __init__(self, graph, states=(), bindings=()):
        self.graph = graph
        self.states = tuple(states)
        self.bindings = tuple(bindings)

    def replay(self, *states):
        if self.graph is None:
            raise RuntimeError("Graph execution is closed")
        if states and len(states) != len(self.states):
            raise ValueError("Graph execution lane count changed")
        for binding in self.bindings:
            binding.wait_for_update()
        for target, source in zip(self.states, states):
            target.copy_for_cuda_graph(source)
        for binding in self.bindings:
            binding.before_replay()
        self.graph.replay()
        for binding in self.bindings:
            binding.after_replay()

    def reset(self):
        if self.graph is None:
            return
        for binding in self.bindings:
            binding.close()
        self.graph.reset()
        self.graph = None
        self.states = ()
        self.bindings = ()


def bind_graph(graph, states=(), runtime=None):
    bindings = []
    for state in states:
        runner = getattr(state, "execution_runner", None)
        if runner is not None:
            bindings.extend(runner.bind_graph(state.execution_metadata, runtime))
    return GraphExecution(graph, states, bindings)
