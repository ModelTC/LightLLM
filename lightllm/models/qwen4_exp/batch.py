from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_same_req_mark


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _minimum(a, b):
    return tl.minimum(a, b)


@triton.jit
def _decode_metadata(
    Requests,
    Lengths,
    Starts,
    Active,
    KeyLengths,
    FirstBlocks,
    QueryRows,
    Visible,
    TokenMask,
    BATCH: tl.constexpr,
    HOLD: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    valid = row < BATCH
    request = tl.load(Requests + row, valid, -1)
    previous = tl.load(Requests + row - 1, (row > 0) & valid, -1)
    following = tl.load(Requests + row + 1, row + 1 < BATCH, -1)
    first = (row == 0) | (request != previous) | (request == HOLD)
    last = (row == BATCH - 1) | (request != following) | (request == HOLD)
    start = tl.associative_scan(tl.where(first, row, 0), 0, _maximum)
    end = tl.associative_scan(tl.where(last, row, BATCH), 0, _minimum, reverse=True)
    length = tl.load(Lengths + row, valid, 0)
    last_length = tl.load(Lengths + end, valid & (end < BATCH), 0)
    active = last & (request != HOLD)
    tl.store(Starts + row, start, valid)
    tl.store(Starts + BATCH, BATCH)
    tl.store(Active + row, tl.where(active, request, -1), valid)
    tl.store(KeyLengths + row, tl.where(active, length, 0), valid)
    tl.store(FirstBlocks + row, tl.where(active, (length - (row - start + 1)) // RATIO, 0), valid)
    tl.store(QueryRows + row, request, valid)
    tl.store(Visible + row, tl.where(request != HOLD, last_length - end + row, 0), valid)
    tl.store(TokenMask + row, request != HOLD, valid)


@triton.jit
def _batch_metadata(
    Requests,
    Lengths,
    Starts,
    Active,
    KeyLengths,
    FirstBlocks,
    QueryRows,
    Visible,
    TokenMask,
    HOLD: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(Requests + row)
    start = tl.load(Starts + row)
    end = tl.load(Starts + row + 1)
    count = end - start
    length = tl.load(Lengths + row)
    active = (request != HOLD) & (count > 0)
    tl.store(Active + row, tl.where(active, request, -1))
    tl.store(KeyLengths + row, tl.where(active, length, 0))
    tl.store(FirstBlocks + row, tl.where(active, (length - count) // RATIO, 0))
    for first in range(0, tl.cdiv(count, BLOCK)):
        offset = first * BLOCK + tl.arange(0, BLOCK)
        mask = offset < count
        tl.store(QueryRows + start + offset, request, mask)
        tl.store(Visible + start + offset, tl.where(active, length - count + offset + 1, 0), mask)
        tl.store(TokenMask + start + offset, active, mask)


@dataclass(frozen=True)
class PackedBatchMetadata:
    starts: torch.Tensor
    active_rows: torch.Tensor
    key_lengths: torch.Tensor
    first_blocks: torch.Tensor
    query_rows: torch.Tensor
    visible_lengths: torch.Tensor
    token_mask: torch.Tensor

    @classmethod
    @torch.no_grad()
    def prepare(
        cls,
        requests,
        lengths,
        query_lengths,
        token_count: int,
        hold_request: int,
        compress_ratio: int,
        segment_capacity=None,
        decode=False,
    ):
        if token_count < 0 or compress_ratio < 1:
            raise ValueError("Token count must be nonnegative and compression ratio positive")
        for tensor in (requests, lengths, query_lengths):
            if (
                not tensor.is_cuda
                or tensor.device != requests.device
                or tensor.dtype != torch.int32
                or tensor.ndim != 1
                or not tensor.is_contiguous()
                or tensor.shape != requests.shape
            ):
                raise ValueError("Batch metadata requires matching contiguous int32 CUDA vectors")
        batch = requests.numel()
        capacity = batch if segment_capacity is None else segment_capacity
        if capacity < batch:
            raise ValueError("Metadata segment capacity cannot be smaller than the input batch")
        if not batch and token_count:
            raise ValueError("An empty batch cannot contain input tokens")

        if decode:
            if token_count != batch or capacity != batch:
                raise ValueError("Decode metadata requires one token per row and no extra segment capacity")
            starts = torch.empty(batch + 1, dtype=torch.int32, device=requests.device)
            active, key_lengths, first_blocks, query_rows, visible = (torch.empty_like(requests) for _ in range(5))
            token_mask = torch.empty(batch, dtype=torch.bool, device=requests.device)
            if batch:
                _decode_metadata[(1,)](
                    requests,
                    lengths,
                    starts,
                    active,
                    key_lengths,
                    first_blocks,
                    query_rows,
                    visible,
                    token_mask,
                    batch,
                    hold_request,
                    compress_ratio,
                    triton.next_power_of_2(batch),
                )
            else:
                starts.zero_()
            return cls(starts, active, key_lengths, first_blocks, query_rows, visible, token_mask)

        starts = torch.full((capacity + 1,), token_count, dtype=torch.int32, device=requests.device)
        starts[:1].zero_()
        if batch:
            marks = gen_same_req_mark(requests, hold_req_idx=hold_request)
            counts = torch.where(marks > 0, query_lengths + marks - 1, 0)
            torch.cumsum(counts, dim=0, dtype=torch.int32, out=starts[1 : batch + 1])
        active = torch.full((capacity,), -1, dtype=torch.int32, device=requests.device)
        key_lengths = torch.zeros_like(active)
        first_blocks = torch.zeros_like(active)
        query_rows = torch.empty(token_count, dtype=torch.int32, device=requests.device)
        visible = torch.empty_like(query_rows)
        token_mask = torch.empty(token_count, dtype=torch.bool, device=requests.device)
        if batch:
            _batch_metadata[(batch,)](
                requests,
                lengths,
                starts,
                active,
                key_lengths,
                first_blocks,
                query_rows,
                visible,
                token_mask,
                hold_request,
                compress_ratio,
                256,
            )
        return cls(starts, active, key_lengths, first_blocks, query_rows, visible, token_mask)

    def copy_from(self, other: "PackedBatchMetadata"):
        for name in self.__dataclass_fields__:
            target, source = getattr(self, name), getattr(other, name)
            if target.shape != source.shape or target.device != source.device or target.dtype != source.dtype:
                raise ValueError(f"Incompatible graph batch metadata: {name}")
        for name in self.__dataclass_fields__:
            getattr(self, name).copy_(getattr(other, name), non_blocking=True)
