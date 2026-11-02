from dataclasses import dataclass, replace

import torch

from ..types import TensorField
from .copy_kernel import copy_bounded_att_tensors


@dataclass(frozen=True)
class BatchedStateView:

    fields: tuple[tuple[str, TensorField], ...]
    cursor: torch.Tensor | None = None
    candidate_strides: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        names = [name for name, _ in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("state fields must have unique names")
        for _, field in self.fields:
            if (
                len(field.axes) != field.tensor.ndim
                or len(field.axes) != len(set(field.axes))
                or field.axes[0] != "slot"
                or field.tensor.shape[0] != self.fields[0][1].tensor.shape[0]
            ):
                raise ValueError("state fields require a common leading slot dimension")
        if self.cursor is not None and (
            not self.fields
            or self.cursor.shape != (self.fields[0][1].tensor.shape[0],)
            or set(dict(self.candidate_strides)) != set(names)
        ):
            raise ValueError("state candidates require a cursor and byte stride for each field")

    def at(self, index):
        return {name: TensorField(field.tensor[index], field.axes[1:]) for name, field in self.fields}

    def narrow_layers(self, start, count):
        return replace(
            self,
            fields=tuple(
                (name, replace(field, tensor=field.tensor.narrow(field.axes.index("layer"), start, count)))
                for name, field in self.fields
            ),
        )

    def offsets(self, indices, device):
        if self.cursor is None:
            return {}
        requests = torch.tensor(indices, dtype=torch.int64, pin_memory=True).to(self.cursor.device, non_blocking=True)
        accepted = self.cursor[requests].to(device=device, dtype=torch.int64, non_blocking=True)
        return {name: accepted * stride for name, stride in self.candidate_strides}

    def finish_restore(self, indices):
        if self.cursor is not None:
            for index in indices:
                self.cursor[index].fill_(0)


def _bind_regions(view, pages, layout, write, window=False):
    if pages.ndim < 2:
        raise ValueError("state pages require a leading page dimension")
    page_bytes = pages.view(torch.uint8).view(pages.shape[0], -1)
    fields = dict(view.fields)
    bound = []
    for region in layout.regions:
        if write and not region.write_enabled:
            continue
        field = fields[region.field]
        if set(field.axes[1:]) != set(region.axes):
            raise ValueError("state field axes differ from the page representation")
        if not 0 <= region.byte_start < region.byte_end <= page_bytes.shape[1]:
            raise ValueError("state region lies outside the page")
        local = field.tensor
        external = page_bytes[:, region.byte_start : region.byte_end].view(region.dtype)
        external = external.view(pages.shape[0], *region.shape)
        external = external.permute(0, *(1 + region.axes.index(axis) for axis in field.axes[1:]))
        if local.dtype != external.dtype:
            raise ValueError("local and external state dtypes differ")
        for mapping in region.axis_mappings:
            if mapping.local_circular or mapping.external_circular or (window and mapping.axis != "layer"):
                raise ValueError("state page mapping requires a non-circular native slice")
            axis = field.axes.index(mapping.axis)
            if (
                min(mapping.local_start, mapping.external_start) < 0
                or mapping.length <= 0
                or mapping.local_start + mapping.length > local.shape[axis]
                or mapping.external_start + mapping.length > external.shape[axis]
            ):
                raise ValueError("state page mapping lies outside the field")
            local = local.narrow(axis, mapping.local_start, mapping.length)
            external = external.narrow(axis, mapping.external_start, mapping.length)
        if window:
            if field.axes[1:] != ("layer", "window", "channel") or local.shape[1] != external.shape[1]:
                raise ValueError("window pages must preserve the logical layer mapping")
        elif local.shape[1:] != external.shape[1:]:
            raise ValueError("local and external state shapes differ")
        bound.append((region.field, local, external))
    return tuple(bound)


@dataclass(frozen=True)
class BoundedPageCopy:
    view: BatchedStateView
    regions: tuple
    write: bool

    @classmethod
    def bind(cls, view, pages, layout, write):
        return cls(view, _bind_regions(view, pages, layout, write), write)

    def run(self, slots, pages, ready=None):
        if len(slots) != len(pages) or (ready is not None and len(ready) != len(pages)):
            raise ValueError("state slot, page and ready counts differ")
        selected = [(slot, page) for i, (slot, page) in enumerate(zip(slots, pages)) if ready is None or not ready[i]]
        if not selected:
            return
        for _, local, external in self.regions:
            if any(not 0 <= slot < len(local) or not 0 <= page < len(external) for slot, page in selected):
                raise ValueError("state slot or page lies outside its allocation")
        indices = [slot for slot, _ in selected]
        offsets = (
            self.view.offsets(indices, self.view.cursor.device) if self.write and self.view.cursor is not None else {}
        )
        for name, local, external in self.regions:
            copies = [(local[slot], external[page]) for slot, page in selected]
            if not self.write:
                copies = [(dst, src) for src, dst in copies]
            copy_bounded_att_tensors(copies, source_byte_offsets=offsets.get(name))
        if not self.write:
            self.view.finish_restore(indices)

    def run_pd(self, *, req_idx, pd_page_index, state_token_len):
        if state_token_len < 0:
            raise ValueError("state recovery requires a nonnegative prefix length")
        self.run([req_idx], [pd_page_index])


def bind_window_page_copy(view, pages, layout, write, kernel):
    regions = _bind_regions(view, pages, layout, write, window=True)

    def run(*, req_idx, pd_page_index, state_token_len):
        if state_token_len < 0 or any(
            not 0 <= req_idx < len(local) or not 0 <= pd_page_index < len(external) for _, local, external in regions
        ):
            raise ValueError("window request, page or prefix length is outside its domain")
        for _, local, external in regions:
            kernel(
                local=local[req_idx],
                external=external[pd_page_index],
                state_token_len=state_token_len,
                write=write,
            )

    return run
