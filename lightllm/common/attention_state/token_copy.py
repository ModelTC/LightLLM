from dataclasses import dataclass
from math import prod, lcm
from typing import Mapping

import torch
import triton
import triton.language as tl

from .types import StateTransferLayout, TensorField
from .utils import merge_contiguous_tail_dims, view_max_copy_dtype


@dataclass(frozen=True)
class TransferBatch:
    token_indices: torch.Tensor
    page_indices: torch.Tensor
    page_readies: torch.Tensor | None = None

    @classmethod
    def prepare(cls, token_indices, page_indices, page_readies=None, page_count=None):
        if page_count is not None and any(not 0 <= index < page_count for index in page_indices):
            raise ValueError("page index outside page pool")
        if token_indices.ndim != 1 or token_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("transfer token indices must be a one-dimensional integer tensor")
        if page_readies is not None and len(page_readies) != len(page_indices):
            raise ValueError("page and ready counts differ")
        device = token_indices.device

        def upload(values, dtype):
            host = torch.tensor(values, dtype=dtype, pin_memory=device.type == "cuda")
            return host.to(device, non_blocking=True)

        return cls(
            token_indices,
            upload(page_indices, torch.int64),
            None if page_readies is None else upload(page_readies, torch.bool),
        )


@dataclass(frozen=True)
class TokenPageCopyPlan:

    copies: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    page_tokens: int
    write: bool
    token_block_size: int = 1
    pad_tail: bool = False

    @classmethod
    def for_tokens(cls, source: Mapping[str, TensorField], destination: Mapping[str, TensorField]):
        if source is None or destination is None or source.keys() != destination.keys():
            raise ValueError("token copy requires matching native fields")
        copies = []
        block_size = 1
        for name, field in source.items():
            other = destination[name]
            if field.token_block_size != other.token_block_size:
                raise ValueError(f"incompatible token block geometry: {name}")
            block_size = lcm(block_size, field.token_block_size)
            axes = ("layer", "token", *(axis for axis in field.axes if axis not in ("layer", "token")))
            if set(axes) != set(other.axes) or field.tensor.dtype != other.tensor.dtype:
                raise ValueError(f"incompatible token copy field {name}")
            src = field.tensor.permute(*(field.axes.index(axis) for axis in axes))
            dst = other.tensor.permute(*(other.axes.index(axis) for axis in axes))
            if (src.shape[0], *src.shape[2:]) != (dst.shape[0], *dst.shape[2:]):
                raise ValueError(f"incompatible token copy shape {name}")
            if src.ndim == 2:
                src, dst = src.unsqueeze(-1), dst.unsqueeze(-1)
            dst = dst.transpose(0, 1).unsqueeze(2)
            pair = merge_contiguous_tail_dims([(src, dst)], start_dim=2)
            copies.extend(view_max_copy_dtype(pair))
        return cls(tuple(copies), 1, True, block_size)

    @classmethod
    def bind(
        cls,
        fields: Mapping[str, TensorField],
        layout: StateTransferLayout,
        pages: torch.Tensor,
        write: bool,
        pad_tail: bool = False,
    ):
        if pages.ndim < 2 or pages.stride(-1) != 1 or not layout.tokens_per_page or layout.tokens_per_page < 0:
            raise ValueError("token copy requires contiguous pages and positive page capacity")
        page_bytes = pages.view(pages.shape[0], prod(pages.shape[1:])).view(torch.uint8)
        if not 0 <= layout.byte_start < layout.byte_end <= page_bytes.shape[1]:
            raise ValueError("page layout outside page pool")
        block_size = lcm(*(field.token_block_size for field in fields.values()))
        if layout.tokens_per_page % block_size:
            raise ValueError("page capacity must contain complete token blocks")
        copies = []
        for region in layout.regions:
            if not layout.byte_start <= region.byte_start < region.byte_end <= layout.byte_end:
                raise ValueError(f"field outside page layout: {region.field}")
            if write and not region.write_enabled:
                continue
            field = fields[region.field]
            if (
                field.tensor is None
                or len(field.axes) != field.tensor.ndim
                or len(set(field.axes)) != len(field.axes)
                or set(field.axes) != set(region.axes)
                or len(region.axes) != len(region.shape)
                or len(set(region.axes)) != len(region.axes)
                or not {"layer", "token"} <= set(field.axes)
            ):
                raise ValueError(f"invalid token field axes: {region.field}")
            if (
                field.tensor.dtype != region.dtype
                or not 0 <= region.byte_start < region.byte_end <= page_bytes.shape[1]
                or region.byte_end - region.byte_start != prod(region.shape) * region.dtype.itemsize
                or region.shape[region.axes.index("token")] != layout.tokens_per_page
            ):
                raise ValueError(f"invalid token field representation: {region.field}")
            if field.tensor.is_cuda and pages.device.type == "cpu" and not pages.is_pinned():
                raise ValueError("CUDA transfers require pinned or registered host pages")
            axes = ("layer", "token", *(axis for axis in field.axes if axis not in ("layer", "token")))
            local = field.tensor.permute(*(field.axes.index(axis) for axis in axes))
            external = page_bytes[:, region.byte_start : region.byte_end].view(region.dtype)
            external = external.view(pages.shape[0], *region.shape)
            external = external.permute(0, *(region.axes.index(axis) + 1 for axis in axes))
            local_slices, external_slices = [slice(None)] * local.ndim, [slice(None)] * external.ndim
            mapped = set()
            for mapping in region.axis_mappings:
                if (
                    mapping.axis == "token"
                    or mapping.axis in mapped
                    or mapping.local_circular
                    or mapping.external_circular
                ):
                    raise ValueError("token page copies require unique non-circular field mappings")
                mapped.add(mapping.axis)
                axis = axes.index(mapping.axis)
                if (
                    min(mapping.local_start, mapping.external_start) < 0
                    or mapping.length <= 0
                    or mapping.local_start + mapping.length > local.shape[axis]
                    or mapping.external_start + mapping.length > external.shape[axis + 1]
                ):
                    raise ValueError(f"field mapping outside storage: {region.field}/{mapping.axis}")
                local_slices[axis] = slice(mapping.local_start, mapping.local_start + mapping.length)
                external_slices[axis + 1] = slice(mapping.external_start, mapping.external_start + mapping.length)
            local, external = local[tuple(local_slices)], external[tuple(external_slices)]
            if (local.shape[0], *local.shape[2:]) != (external.shape[1], *external.shape[3:]):
                raise ValueError(f"local and external field shapes differ: {region.field}")
            if local.ndim == 2:
                local, external = local.unsqueeze(-1), external.unsqueeze(-1)
            pair = merge_contiguous_tail_dims([(local, external)], start_dim=2)
            copies.extend(view_max_copy_dtype(pair))
        return cls(tuple(copies), layout.tokens_per_page, write, block_size, pad_tail)

    def validate_batch(self, batch):
        indices = (batch.token_indices, batch.page_indices)
        if any(index.ndim != 1 or index.dtype not in (torch.int32, torch.int64) for index in indices):
            raise ValueError("copy indices must be one-dimensional integer tensors")
        domain = batch.token_indices.device.type
        if batch.page_indices.device.type != domain or any(local.device.type != domain for local, _ in self.copies):
            raise ValueError("copy indices and native storage must use the same memory domain")
        count = batch.token_indices.numel()
        if count % self.token_block_size and not self.pad_tail:
            raise ValueError("token copies require complete token blocks")
        if batch.page_indices.numel() != triton.cdiv(count, self.page_tokens):
            raise ValueError("KV token and page counts differ")

    def run(self, batch: TransferBatch):
        self.validate_batch(batch)
        index_count = batch.token_indices.numel()
        count = triton.cdiv(index_count, self.token_block_size) * self.token_block_size
        if not count:
            return
        for local, external in self.copies:
            if batch.token_indices.device.type == "cpu":
                if local.device.type != "cpu" or external.device.type != "cpu":
                    raise ValueError("device copies require device token indices")
                integer = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[local.element_size()]
                src, dst = local.view(integer), external.view(integer)
                positions = torch.arange(count)
                slots = batch.token_indices.long()
                if count > index_count:
                    slots = torch.cat((slots, slots[-1:] + torch.arange(1, count - index_count + 1)))
                page_rows, offsets = positions // self.page_tokens, positions % self.page_tokens
                if self.write and batch.page_readies is not None:
                    keep = ~batch.page_readies[page_rows]
                    slots, page_rows, offsets = slots[keep], page_rows[keep], offsets[keep]
                pages = batch.page_indices[page_rows]
                if self.write:
                    dst[pages, :, offsets] = src[:, slots].transpose(0, 1)
                else:
                    src[:, slots] = dst[pages, :, offsets].transpose(0, 1)
                continue
            elements = local.shape[0] * count * prod(local.shape[2:])
            _copy_token_pages[(24,)](
                local,
                external,
                batch.token_indices,
                batch.page_indices,
                batch.page_readies,
                count,
                index_count,
                elements,
                batch.token_indices.stride(0),
                batch.page_indices.stride(0),
                local.shape[2:],
                local.stride(),
                external.stride(),
                self.page_tokens,
                local.element_size(),
                self.write,
                batch.page_readies is not None,
                count > index_count,
                BLOCK=1024,
                num_warps=8,
                num_stages=1,
            )


@triton.jit
def _copy_token_pages(
    LOCAL,
    EXTERNAL,
    TOKENS,
    PAGES,
    READIES,
    count,
    index_count,
    elements,
    TOKEN_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TAIL: tl.constexpr,
    LS: tl.constexpr,
    ES: tl.constexpr,
    PAGE: tl.constexpr,
    BYTES: tl.constexpr,
    WRITE: tl.constexpr,
    HAS_READIES: tl.constexpr,
    PAD_TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if BYTES == 1:
        LOCAL, EXTERNAL = LOCAL.to(tl.pointer_type(tl.uint8)), EXTERNAL.to(tl.pointer_type(tl.uint8))
    elif BYTES == 2:
        LOCAL, EXTERNAL = LOCAL.to(tl.pointer_type(tl.uint16)), EXTERNAL.to(tl.pointer_type(tl.uint16))
    elif BYTES == 4:
        LOCAL, EXTERNAL = LOCAL.to(tl.pointer_type(tl.uint32)), EXTERNAL.to(tl.pointer_type(tl.uint32))
    else:
        LOCAL, EXTERNAL = LOCAL.to(tl.pointer_type(tl.uint64)), EXTERNAL.to(tl.pointer_type(tl.uint64))
    count = tl.cast(count, tl.int64)
    for start in range(tl.program_id(0).to(tl.int64) * BLOCK, elements, tl.num_programs(0) * BLOCK):
        index = start + tl.arange(0, BLOCK)
        valid = index < elements
        remaining, lo, eo = index, tl.full((BLOCK,), 0, tl.int64), tl.full((BLOCK,), 0, tl.int64)
        for axis in tl.static_range(len(TAIL) - 1, -1, -1):
            coord = remaining % TAIL[axis]
            remaining //= TAIL[axis]
            lo += coord * LS[axis + 2]
            eo += coord * ES[axis + 3]
        token, layer = remaining % count, remaining // count
        if WRITE and HAS_READIES:
            valid &= ~tl.load(READIES + token // PAGE, valid, other=1)
        if PAD_TAIL:
            index = tl.minimum(token, index_count - 1)
            slot = tl.load(TOKENS + index * TOKEN_STRIDE, valid, other=0).to(tl.int64)
            slot += tl.maximum(token - index_count + 1, 0)
        else:
            slot = tl.load(TOKENS + token * TOKEN_STRIDE, valid, other=0).to(tl.int64)
        page = tl.load(PAGES + (token // PAGE) * PAGE_STRIDE, valid, other=0).to(tl.int64)
        lo += layer * LS[0] + slot * LS[1]
        eo += page * ES[0] + layer * ES[1] + (token % PAGE) * ES[2]
        if WRITE:
            tl.store(EXTERNAL + eo, tl.load(LOCAL + lo, valid, other=0), valid)
        else:
            tl.store(LOCAL + lo, tl.load(EXTERNAL + eo, valid, other=0), valid)
