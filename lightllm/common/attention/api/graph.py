from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass

import torch


def tensor_input_copy(name, target, source, *, shared=False):
    if target is None and source is None:
        return []
    if not isinstance(target, torch.Tensor) or not isinstance(source, torch.Tensor):
        raise ValueError(f"captured input {name} changed presence")
    if (target.shape, target.dtype, target.device) != (source.shape, source.dtype, source.device):
        raise ValueError(f"captured input {name} changed shape, dtype or device")
    if target.data_ptr() == source.data_ptr():
        if target.stride() != source.stride():
            raise ValueError(f"captured input {name} changed strides at the captured address")
        return [(target, source)]
    if shared:
        raise ValueError(f"{name} storage changed; recapture attention")
    return [(target, source)]


def tensor_span(tensor):
    if tensor.numel() == 0:
        return tensor.data_ptr(), tensor.data_ptr()
    extent = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())) + 1
    return tensor.data_ptr(), tensor.data_ptr() + extent * tensor.element_size()


def tensors_overlap(left, right):
    if left.device != right.device:
        return False
    a, b = tensor_span(left)
    c, d = tensor_span(right)
    return a < b and c < d and a < d and c < b


@dataclass(frozen=True)
class _TensorView:
    address: int
    end: int
    device: object
    dtype: object
    shape: tuple
    strides: tuple
    contiguous: bool

    @classmethod
    def of(cls, tensor):
        address = tensor.data_ptr()
        shape, strides = tuple(tensor.shape), tensor.stride()
        extent = sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1 if tensor.numel() else 0
        return cls(
            address,
            address + extent * tensor.element_size(),
            tensor.device,
            tensor.dtype,
            shape,
            strides,
            tensor.is_contiguous(),
        )

    def overlaps(self, other):
        return (
            self.device == other.device
            and self.address < self.end
            and other.address < other.end
            and self.address < other.end
            and other.address < self.end
        )

    def check_destination(self):
        extent = 1
        for stride, size in sorted((stride, size) for stride, size in zip(self.strides, self.shape) if size > 1):
            if stride < extent:
                raise ValueError("captured input destination has overlapping elements")
            extent += (size - 1) * stride


@dataclass(frozen=True)
class CaptureEntry:
    name: str
    policy: str
    value: object
    view: _TensorView | None = None


def capture_entries(name, policy, value, *, objects=False, seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        yield CaptureEntry(name, policy, value, _TensorView.of(value))
        return
    if isinstance(value, Mapping):
        children = value.items()
    elif isinstance(value, (tuple, list)):
        children = enumerate(value)
    elif is_dataclass(value) and not isinstance(value, type):
        children = ((item.name, getattr(value, item.name)) for item in fields(value))
    elif objects and hasattr(value, "__dict__") and not isinstance(value, type) and not callable(value):
        children = vars(value).items()
    else:
        yield CaptureEntry(name, policy, value)
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    for key, item in children:
        if not isinstance(key, (str, int)):
            key = f"{type(key).__name__}@{id(key)}"
        yield from capture_entries(f"{name}.{key}", policy, item, objects=objects, seen=seen)


class CaptureManifest:
    def __init__(self):
        self.entries = []
        self._guards = []

    def add(self, name, policy, value, *, read=None):
        entries = tuple(capture_entries(name, policy, value))
        self.entries.extend(entries)
        if read is not None and (policy != "prepare" or any(entry.view is not None for entry in entries)):
            self._guards.append((name, policy, read, entries))

    def check(self, *, prepared=False):
        for name, policy, read, entries in self._guards:
            if prepared and policy not in ("prepare", "shared"):
                continue
            current = read()
            incoming = tuple(capture_entries(name, policy, current))
            if len(incoming) != len(entries):
                raise ValueError(f"captured inventory {name} changed; recapture Graph")
            for expected, actual in zip(entries, incoming):
                if expected.name != actual.name or expected.view != actual.view:
                    raise ValueError(f"captured address {expected.name} changed; recapture Graph")
                if expected.view is not None or not isinstance(expected.value, (str, int, float, bool, type(None))):
                    if expected.value is not actual.value:
                        raise ValueError(f"captured owner {expected.name} changed; recapture Graph")
                elif expected.value != actual.value:
                    raise ValueError(f"capture constant {expected.name} changed")


class TensorCopyPlan:
    def __init__(self, targets):
        self.targets = tuple(targets)
        self.views = tuple(_TensorView.of(target) for target in self.targets)
        unique = {}
        self.indices = []
        self.duplicates = []
        for index, view in enumerate(self.views):
            view.check_destination()
            if view in unique:
                self.duplicates.append((unique[view], index))
            else:
                unique[view] = index
                self.indices.append(index)
        self.overlaps = []
        self.covered_by = {i: [] for i in self.indices}
        for position, i in enumerate(self.indices):
            left = self.views[i]
            for j in self.indices[position + 1 :]:
                right = self.views[j]
                if left.overlaps(right):
                    self.overlaps.append((i, j))
                    if left.contiguous and right.contiguous:
                        if left.address <= right.address and right.end <= left.end:
                            self.covered_by[j].append(i)
                        elif right.address <= left.address and left.end <= right.end:
                            self.covered_by[i].append(j)
        self.write_order = sorted(self.indices, key=lambda i: self.views[i].end - self.views[i].address, reverse=True)
        self.intervals = {}
        for device in {view.device for view in self.views}:
            ordered = sorted(
                (self.views[i].address, self.views[i].end, i) for i in self.indices if self.views[i].device == device
            )
            starts, ends, maximum = [], [], 0
            for start, end, _ in ordered:
                starts.append(start)
                maximum = max(maximum, end)
                ends.append(maximum)
            self.intervals[device] = starts, ends, ordered

    def validate(self, copies):
        if len(copies) != len(self.targets):
            raise ValueError("captured input inventory changed")
        sources = []
        for (target, source), expected, view in zip(copies, self.targets, self.views):
            if (
                target is not expected
                or (
                    target.data_ptr(),
                    tuple(target.shape),
                    target.stride(),
                    target.dtype,
                    target.device,
                )
                != (view.address, view.shape, view.strides, view.dtype, view.device)
            ):
                raise ValueError("captured input destination changed; recapture Graph")
            incoming = _TensorView.of(source)
            if (incoming.shape, incoming.dtype, incoming.device) != (view.shape, view.dtype, view.device):
                raise ValueError("captured input changed shape, dtype or device")
            if incoming.address == view.address and incoming.strides != view.strides:
                raise ValueError("captured input changed strides at the captured address")
            sources.append(incoming)
        for first, duplicate in self.duplicates:
            if sources[first] != sources[duplicate]:
                raise ValueError("captured inputs have conflicting destination writes")
        unchanged = {i for i in self.indices if sources[i].address == self.views[i].address}

        def same_mapping(i, j):
            left, right = self.views[i], self.views[j]
            incoming, other = sources[i], sources[j]
            return (
                left.contiguous
                and right.contiguous
                and incoming.contiguous
                and other.contiguous
                and left.dtype == right.dtype
                and incoming.device == other.device
                and incoming.address - left.address == other.address - right.address
            )

        for i, j in self.overlaps:
            if i in unchanged and j in unchanged:
                continue
            if not same_mapping(i, j):
                raise ValueError("captured inputs have overlapping destinations")
        for i in self.indices:
            source = sources[i]
            if source.address == source.end:
                continue
            starts, ends, ordered = self.intervals[source.device]
            for position in range(bisect_right(ends, source.address), bisect_left(starts, source.end)):
                start, end, j = ordered[position]
                if start < end and end > source.address and not (i in unchanged and j in unchanged):
                    raise ValueError("captured input copy would overwrite another source or overlaps its source")
        selected = set()
        writes = []
        for i in self.write_order:
            if i in unchanged or any(j in selected and same_mapping(i, j) for j in self.covered_by[i]):
                continue
            selected.add(i)
            writes.append(copies[i])
        return writes


def validate_copy_plan(copies):
    return TensorCopyPlan([target for target, _ in copies]).validate(copies)
