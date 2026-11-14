from contextlib import ExitStack, contextmanager
from dataclasses import replace

import torch


class AttentionSession:
    def __init__(self):
        self.groups = ()
        self.contexts = {}
        self.batch = None
        self._active = False

    @contextmanager
    def prepare(self, batch, groups, capacity, *, workspace, capture=False, replay=False):
        if self._active:
            raise RuntimeError("attention session is already active")
        batch.validate()
        capacity.check(batch)
        self.groups = groups
        self.batch = batch
        self.contexts = {}
        with ExitStack() as leases:
            if not replay:
                if capture:
                    self.batch = replace(
                        batch,
                        max_kv_seq_len=capacity.max_kv_seq_len,
                        total_token_num=capacity.batch_size * capacity.max_kv_seq_len,
                    )
                for group_id, group in enumerate(groups):
                    scratch = workspace(group_id)
                    leases.enter_context(scratch.lease())
                    backend = group[batch.phase]
                    if capture:
                        scratch = scratch.borrow_for_capture()
                    scratch.reserve(backend.resource_requirements(capacity, capture=capture))
                    context = self._allocate(backend, capacity, capture=capture, workspace=scratch)
                    context.batch = self.batch
                    context.current_max_kv_seq_len = batch.max_kv_seq_len
                    backend.plan_host(context)
                    self.contexts[group_id] = context
            self._active = not replay
            try:
                yield self
            finally:
                self._active = False

    @staticmethod
    def _allocate(backend, capacity, *, capture, workspace):
        if capacity.phase is not backend.phase:
            raise ValueError("backend and context phases differ")
        context = backend.allocate_context(capacity, capture=capture, workspace=workspace)
        if context.capacity != capacity:
            raise ValueError("backend changed attention context capacity")
        if context.workspace is not workspace or context.capture != capture:
            raise ValueError("backend changed attention context workspace or capture mode")
        context.backend_identity = backend
        return context

    @staticmethod
    def _check_context(backend, context):
        if context.backend_identity is not backend or context.capacity.phase is not backend.phase:
            raise ValueError("context belongs to a different attention backend or phase")

    def _check_active(self):
        if not self._active:
            raise RuntimeError("attention execution requires an active session scope")

    def context(self, layer):
        self._check_active()
        context = self.contexts[layer.group_id]
        self._check_context(layer.backend, context)
        if context.batch is None:
            raise RuntimeError("prepare attention inputs before execution")
        return context

    def build_device_metadata(self):
        self._check_active()
        for group_id, context in self.contexts.items():
            backend = self.groups[group_id][context.capacity.phase]
            self._check_context(backend, context)
            backend.build_device_metadata(context)

    def forward(self, layer, inputs, *, out=None):
        context = self.context(layer)
        result = layer.backend.forward(layer, context, inputs, out=out)
        if out is None:
            return result
        attention = result[0] if isinstance(result, tuple) else result
        if attention.dtype != out.dtype or attention.device != out.device:
            raise ValueError("out must match the attention result dtype and device")
        if attention.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
            if out.is_contiguous():
                begin = attention.storage_offset()
                end = begin + sum((size - 1) * stride for size, stride in zip(attention.shape, attention.stride())) + 1
                valid = begin >= out.storage_offset() and end <= out.storage_offset() + out.numel()
            else:
                valid = (
                    attention.ndim == out.ndim
                    and attention.storage_offset() == out.storage_offset()
                    and all(size <= bound for size, bound in zip(attention.shape, out.shape))
                    and all(
                        size <= 1 or stride == out_stride
                        for size, stride, out_stride in zip(attention.shape, attention.stride(), out.stride())
                    )
                )
            if attention.numel() and not valid:
                raise ValueError("attention result must be contained in contiguous out or a strided out prefix")
            return result
        if attention.numel() != out.numel():
            raise ValueError("out must match the attention result element count")
        out.copy_(attention.reshape(out.shape))
        if isinstance(result, tuple):
            return type(result)(out, *result[1:]) if hasattr(result, "_fields") else (out, *result[1:])
        return out

    def replay_input_copies(self, source):
        if len(self.groups) != len(source.groups) or any(
            current is not incoming for current, incoming in zip(self.groups, source.groups)
        ):
            raise ValueError("captured attention belongs to a different model role or group")

        source.batch.validate()
        for context in self.contexts.values():
            context.capacity.check(source.batch)
        return self.batch.input_copies(source.batch)

    def plan_replay(self, source):
        for group_id, context in self.contexts.items():
            context.current_max_kv_seq_len = source.batch.max_kv_seq_len
            self.groups[group_id][context.capacity.phase].plan_host(context)
