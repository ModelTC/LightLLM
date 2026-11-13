from contextlib import ExitStack, contextmanager
from dataclasses import replace

import torch


class AttentionSession:
    def __init__(self):
        self.groups = ()
        self.contexts = {}
        self.batch = None

    @contextmanager
    def prepare(self, batch, groups, capacity, *, workspace, capture=False, replay=False):
        batch.validate()
        capacity.check(batch)
        self.groups = groups
        self.batch = batch
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
                    context = self.allocate(backend, capacity, capture=capture, workspace=scratch)
                    context.batch = self.batch
                    context.current_max_kv_seq_len = batch.max_kv_seq_len
                    backend.plan_host(context)
                    self.contexts[group_id] = context
            yield self

    @staticmethod
    def allocate(backend, capacity, *, capture, workspace):
        if capacity.phase is not backend.phase:
            raise ValueError("backend and context phases differ")
        context = backend.allocate_context(capacity, capture=capture, workspace=workspace)
        if context.capacity != capacity:
            raise ValueError("backend changed attention context capacity")
        context.backend_identity = backend
        return context

    @staticmethod
    def plan(backend, context, batch, *, copy_inputs=True):
        AttentionSession._check_context(backend, context)
        batch.validate()
        context.capacity.check(batch)
        context.current_max_kv_seq_len = batch.max_kv_seq_len
        if context.capture and context.batch is not None:
            if copy_inputs:
                context.batch.copy_inputs(batch)
        elif context.capture:
            context.batch = replace(
                batch,
                max_kv_seq_len=context.capacity.max_kv_seq_len,
                total_token_num=context.capacity.batch_size * context.capacity.max_kv_seq_len,
            )
        else:
            context.batch = batch
        backend.plan_host(context)

    @staticmethod
    def _check_context(backend, context):
        if context.backend_identity is not backend or context.capacity.phase is not backend.phase:
            raise ValueError("context belongs to a different attention backend or phase")

    def context(self, layer):
        context = self.contexts[layer.group_id]
        self._check_context(layer.backend, context)
        if context.batch is None:
            raise RuntimeError("prepare attention inputs before execution")
        return context

    @staticmethod
    def build(backend, context):
        AttentionSession._check_context(backend, context)
        if context.batch is None:
            raise RuntimeError("prepare attention inputs before update")
        backend.build_device_metadata(context)

    def build_device_metadata(self):
        for group_id, context in self.contexts.items():
            self.build(self.groups[group_id][context.capacity.phase], context)

    def forward(self, layer, inputs, *, out=None):
        context = self.context(layer)
        result = layer.backend.forward(layer, context, inputs, out=out)
        if out is None:
            return result
        attention = result[0] if isinstance(result, tuple) else result
        if attention.dtype != out.dtype or attention.device != out.device:
            raise ValueError("out must match the attention result dtype and device")
        if attention.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
            begin = attention.storage_offset()
            end = begin + sum((size - 1) * stride for size, stride in zip(attention.shape, attention.stride())) + 1
            out_end = (
                out.storage_offset() + sum((size - 1) * stride for size, stride in zip(out.shape, out.stride())) + 1
            )
            if attention.numel() and (begin < out.storage_offset() or end > out_end):
                raise ValueError("attention result view extends beyond out")
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
