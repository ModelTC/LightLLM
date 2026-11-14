from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .state import PLEBatch
from .checkpoint import ngram_eos_token_id
from lightllm.utils.dist_utils import get_current_device_id


@triton.jit
def _gather_history(
    Rows,
    Starts,
    Lengths,
    Ngram,
    Convolution,
    OutNgram,
    OutConvolution,
    NGRAM: tl.constexpr,
    WIDTH: tl.constexpr,
    EOS: tl.constexpr,
    RESET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    request = tl.load(Rows + row).to(tl.int64)
    active = request >= 0
    if RESET:
        count = tl.load(Starts + row + 1) - tl.load(Starts + row)
        active = active & (count != tl.load(Lengths + row))
    col = part * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Convolution + request * WIDTH + col, active & (col < WIDTH), 0)
    tl.store(OutConvolution + row.to(tl.int64) * WIDTH + col, values, col < WIDTH)
    if part == 0 and NGRAM > 0:
        offset = tl.arange(0, triton.next_power_of_2(NGRAM))
        tokens = tl.load(Ngram + request * NGRAM + offset, active & (offset < NGRAM), EOS)
        tl.store(OutNgram + row.to(tl.int64) * NGRAM + offset, tokens, offset < NGRAM)


@triton.jit
def _scatter_history(Source, Destinations, Target, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    destination = tl.load(Destinations + row).to(tl.int64)
    if destination >= 0:
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(Source + row.to(tl.int64) * WIDTH + col, col < WIDTH, other=0)
        tl.store(Target + destination * WIDTH + col, values, col < WIDTH)


@dataclass(frozen=True)
class LexicalSnapshot:
    ngram: torch.Tensor
    convolution: torch.Tensor


class LexicalRequestCache:
    def __init__(self, layer_weight, requests: int):
        if requests < 1:
            raise ValueError("Lexical cache requires at least one request slot")
        self.layer_weight = layer_weight
        config = layer_weight.network_config_
        self.eos_token_id = ngram_eos_token_id(config)

        device = torch.device("cuda", get_current_device_id())
        self.ngram = torch.full(
            (requests, config["ngram_size"] - 1), self.eos_token_id, dtype=torch.int64, device=device
        )
        self.convolution = torch.zeros(
            (
                requests,
                (config["ple_conv_kernel_size"] - 1) * config["ngram_size"],
                config["hc_count"] * config["hidden_size"],
            ),
            dtype=layer_weight.data_type_,
            device=device,
        )

    def reset(self, request: int):
        self.ngram[request].fill_(self.eos_token_id)
        self.convolution[request].zero_()

    @torch.no_grad()
    def prepare(self, metadata, reset_empty_prefix=False) -> PLEBatch:
        rows = metadata.active_rows
        if rows.is_cuda:
            ngram = torch.empty((rows.numel(), self.ngram.shape[1]), dtype=self.ngram.dtype, device=rows.device)
            convolution = torch.empty(
                (rows.numel(), *self.convolution.shape[1:]), dtype=self.convolution.dtype, device=rows.device
            )
            width = self.convolution.shape[1] * self.convolution.shape[2]
            if rows.numel():
                _gather_history[(rows.numel(), max(1, triton.cdiv(width, 512)))](
                    rows,
                    metadata.starts,
                    metadata.key_lengths,
                    self.ngram,
                    self.convolution,
                    ngram,
                    convolution,
                    self.ngram.shape[1],
                    width,
                    self.eos_token_id,
                    reset_empty_prefix,
                    512,
                )
            return PLEBatch(metadata.starts, ngram, convolution, metadata.token_mask)
        indices = rows.clamp_min(0).to(torch.int64)
        ngram = self.ngram.index_select(0, indices)
        convolution = self.convolution.index_select(0, indices)
        empty = rows < 0
        if reset_empty_prefix:
            counts = metadata.starts[1:] - metadata.starts[:-1]
            empty = empty | (metadata.key_lengths == counts)
        ngram.masked_fill_(empty[:, None], self.eos_token_id)
        convolution.masked_fill_(empty[:, None, None], 0)
        return PLEBatch(metadata.starts, ngram, convolution, metadata.token_mask)

    @torch.no_grad()
    def commit(self, metadata, batch: PLEBatch, accepted_lengths=None):
        ngram, convolution = batch.histories_after(self.layer_weight, accepted_lengths)
        for values, storage in ((ngram, self.ngram), (convolution, self.convolution)):
            width = storage[0].numel()
            if metadata.active_rows.numel() and width:
                _scatter_history[(metadata.active_rows.numel(), triton.cdiv(width, 256))](
                    values, metadata.active_rows, storage, width, 256
                )

    def snapshot(self, request: int) -> LexicalSnapshot:
        return LexicalSnapshot(self.ngram[request].clone(), self.convolution[request].clone())

    def restore(self, request: int, snapshot: LexicalSnapshot):
        for target, source in (
            (self.ngram[request], snapshot.ngram),
            (self.convolution[request], snapshot.convolution),
        ):
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError("Lexical snapshot has incompatible shape or dtype")
        self.ngram[request].copy_(snapshot.ngram, non_blocking=True)
        self.convolution[request].copy_(snapshot.convolution, non_blocking=True)
