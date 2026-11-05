"""Runner-owned sequence of captured segments and dynamic batch operations."""

from contextlib import contextmanager
import sys

import torch

from lightllm.common.execution.device.cuda import CudaRuntime


class GraphProgram:
    def __init__(self, pool, runtime=CudaRuntime):
        self.pool, self.runtime = pool, runtime
        self.steps = []
        self._capture = None
        self._complete = False

    @property
    def is_capturing(self):
        return self._capture is not None

    def close(self):
        if self.is_capturing:
            raise RuntimeError("cannot close an active Graph capture")
        self._complete = False
        for step in self.steps:
            if not callable(step):
                step.reset()
        self.steps.clear()
        self.pool = None

    def _start_segment(self):
        graph = self.runtime.create_graph()
        capture = self.runtime.capture_graph(graph, pool=self.pool)
        capture.__enter__()
        self._capture = capture
        self.steps.append(graph)

    def _end_segment(self, exc_info=(None, None, None)):
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.__exit__(*exc_info)

    @contextmanager
    def capture(self):
        if self.steps or self.is_capturing:
            raise RuntimeError("a Graph program can only be captured once")
        try:
            self._start_segment()
            yield
        except BaseException:
            self._end_segment(sys.exc_info())
            raise
        else:
            self._end_segment()
            self._complete = True

    def _pause(self, inputs):
        if not self.is_capturing:
            raise RuntimeError("dynamic Graph steps require an active capture")
        views = tuple(self.runtime.borrow_tensor(value) if value is not None else None for value in inputs)
        self._end_segment()
        return views

    def defer(self, function, inputs):
        """Schedule an in-place operation using the request passed to replay."""
        views = self._pause(inputs)
        self.steps.append(lambda batch: function(views, batch))
        self._start_segment()

    def call(self, function, inputs, batch, output_specs=None):
        """Give a dynamic operation stable outputs consumed by later segments.

        Functions return a tuple of tensors. Explicit (shape, dtype, device)
        descriptors avoid probing operations that synchronize or mutate state.
        """
        views = self._pause(inputs)
        if output_specs is None:
            with self.runtime.capture_graph(self.runtime.create_graph()):
                outputs = function(views, batch)
                output_specs = tuple((value.shape, value.dtype, value.device) for value in outputs)
                del outputs
        self._start_segment()
        outputs = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype, device in output_specs)
        targets = tuple(self.runtime.borrow_tensor(value) for value in outputs)

        def execute(current_batch):
            values = function(views, current_batch)
            if len(values) != len(targets):
                raise ValueError("dynamic Graph output count changed")
            for target, value in zip(targets, values):
                if target.shape != value.shape or target.dtype != value.dtype or target.device != value.device:
                    raise ValueError("dynamic Graph output layout changed")
                target.copy_(value)

        self.steps.insert(len(self.steps) - 1, execute)
        return outputs

    def replay(self, batch):
        if not self._complete or self.is_capturing:
            raise RuntimeError("cannot replay an incomplete Graph program")
        for step in self.steps:
            if callable(step):
                step(batch)
            else:
                step.replay()
