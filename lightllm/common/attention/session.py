from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace

import torch

from .api.graph import tensors_overlap


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for item in fields(value):
            yield from _tensors(getattr(value, item.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


@dataclass
class _CapturedAttention:
    groups: tuple
    contexts: dict
    batch: object
    binding: object

    def check_binding(self):
        if self.binding is not None:
            self.binding.check()

    def replay_input_copies(self, source):
        self.check_binding()
        if self.binding is not source.binding:
            raise ValueError("captured attention belongs to a different model binding")
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


class AttentionSession:
    def __init__(self):
        self.groups = ()
        self.contexts = {}
        self.batch = None
        self.binding = None
        self._active = False
        self._entered = False
        self._pass_active = False
        self._failed = False
        self.state_may_be_modified = False
        self._retained = []

    @contextmanager
    def prepare(self, batch, groups, capacity, *, workspace, capture=False, replay=False, retain=(), binding=None):
        if self._entered:
            raise RuntimeError("attention session is already active")
        batch.validate()
        capacity.check(batch)
        if binding is not None:
            binding.check()
        self.binding = binding
        self.groups = groups
        self.batch = batch
        self.contexts = {}
        self._entered = True
        self._failed = self.state_may_be_modified = False
        self._retained = [batch, *retain]
        try:
            with ExitStack() as leases:
                if replay:
                    yield self
                    return
                if capture:
                    self.batch = replace(
                        batch,
                        max_kv_seq_len=capacity.max_kv_seq_len,
                        total_token_num=capacity.batch_size * capacity.max_kv_seq_len,
                    )
                scratches = [workspace(group_id) for group_id in range(len(groups))]
                for execution in dict.fromkeys(scratch.execution for scratch in scratches):
                    leases.enter_context(execution.join(retain=(self._retained,)))
                for group_id, (group, scratch) in enumerate(zip(groups, scratches)):
                    backend = group[batch.phase]
                    if capture:
                        scratch = scratch.borrow_for_capture()
                    scratch.reserve(backend.resource_requirements(capacity, capture=capture))
                    context = self._allocate(backend, capacity, capture=capture, workspace=scratch)
                    self._retained.append(context)
                    context.batch = self.batch
                    context.current_max_kv_seq_len = batch.max_kv_seq_len
                    backend.plan_host(context)
                    self.contexts[group_id] = context
                self._active = True
                yield self
        finally:
            self._active = self._entered = self._pass_active = False
            self._retained = []

    def retain(self, *values):
        self._check_active()
        self._retained.extend(values)

    @contextmanager
    def layer_pass(self):
        self._check_active()
        if self._pass_active:
            raise RuntimeError("attention layer pass is already active")
        self._pass_active = True
        try:
            self.build_device_metadata()
            self.state_may_be_modified = True
            yield
        except BaseException:
            self._failed = True
            raise
        finally:
            self._pass_active = False

    def capture(self):
        return _CapturedAttention(self.groups, dict(self.contexts), self.batch, self.binding)

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
        if self._failed:
            raise RuntimeError("attention pass failed; execution cannot continue")

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
        if not self._pass_active:
            raise RuntimeError("attention execution requires an active layer pass")
        description = layer.output_spec(layer.backend.spec, inputs) if out is not None else None
        destination = out
        if description is not None:
            shape, exemplar = description
            if out.dtype != exemplar.dtype or out.device != exemplar.device:
                raise ValueError("out must match the attention input dtype and device")
            if out.ndim == 2 and out.shape == (shape[0], shape[1] * shape[2]) and out.is_contiguous():
                destination = out.view(shape)
            elif out.ndim == len(shape) and all(size >= needed for size, needed in zip(out.shape, shape)):
                destination = out[tuple(slice(0, size) for size in shape)] if tuple(out.shape) != shape else out
            else:
                raise ValueError("out does not contain the logical attention output shape")
            strides = sorted((stride, size) for stride, size in zip(out.stride(), out.shape) if size > 1)
            extent = 1
            for stride, size in strides:
                if stride < extent:
                    raise ValueError("out must not contain overlapping elements")
                extent += (size - 1) * stride
            if any(tensors_overlap(out, tensor) for value in (inputs, layer.resources) for tensor in _tensors(value)):
                raise ValueError("out must not overlap live attention inputs or persistent resources")
        result = layer.backend.forward(layer, context, inputs, out=out)
        if out is None:
            return result
        attention = result[0] if isinstance(result, tuple) else result
        if description is not None:
            shape, _ = description
            allowed = (shape, (shape[0], shape[1] * shape[2]))
            if layer.backend.spec.family.is_recurrent:
                allowed += ((1, *shape),)
            if tuple(attention.shape) not in allowed:
                raise ValueError("attention result does not match its logical output shape")
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
        if attention.numel() != destination.numel():
            raise ValueError("out must match the attention result element count")
        destination.copy_(attention.reshape(destination.shape))
        out = destination if destination.numel() != out.numel() else out
        if isinstance(result, tuple):
            return type(result)(out, *result[1:]) if hasattr(result, "_fields") else (out, *result[1:])
        return out
