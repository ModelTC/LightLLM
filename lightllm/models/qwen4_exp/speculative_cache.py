import torch
import triton
import triton.language as tl


@triton.jit
def _stage(
    History,
    Inputs,
    Starts,
    Requests,
    Pending,
    CONTEXT: tl.constexpr,
    WIDTH: tl.constexpr,
    VERIFY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    segment = tl.program_id(0)
    request = tl.load(Requests + segment).to(tl.int64)
    if request >= 0:
        start = tl.load(Starts + segment)
        count = tl.load(Starts + segment + 1) - start
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        position, column = offset // WIDTH, offset % WIDTH
        old = tl.load(History + segment.to(tl.int64) * CONTEXT * WIDTH + offset, position < CONTEXT, other=0)
        new = tl.load(
            Inputs + (start.to(tl.int64) + position - CONTEXT) * WIDTH + column,
            (position >= CONTEXT) & (position < CONTEXT + count),
            other=0,
        )
        tl.store(
            Pending + request * (CONTEXT + VERIFY) * WIDTH + offset,
            tl.where(position < CONTEXT, old, new),
            (position < CONTEXT + count) & (position < CONTEXT + VERIFY),
        )


@triton.jit
def _commit(
    Pending,
    History,
    Requests,
    Starts,
    Accepted,
    HOLD: tl.constexpr,
    CONTEXT: tl.constexpr,
    WIDTH: tl.constexpr,
    VERIFY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    start = tl.load(Starts + group)
    request = tl.load(Requests + start).to(tl.int64)
    if request != HOLD:
        count = tl.load(Accepted + group)
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(
            Pending + request * (CONTEXT + VERIFY) * WIDTH + count * WIDTH + offset, offset < CONTEXT * WIDTH, other=0
        )
        tl.store(History + request * CONTEXT * WIDTH + offset, values, offset < CONTEXT * WIDTH)


class LexicalSpeculativeCache:
    def __init__(self, accepted_cache, verify_width):
        if verify_width < 1:
            raise ValueError("Speculative lexical cache requires positive verify width")
        self.verify_width = verify_width
        self.fields = []
        for history in (accepted_cache.ngram, accepted_cache.convolution):
            requests, context = history.shape[:2]
            width = history.shape[2] if history.ndim == 3 else 1
            pending = torch.empty((requests, context + verify_width, width), dtype=history.dtype, device=history.device)
            self.fields.append((history, pending, context, width))

    def stage(self, metadata, batch):
        if batch.input_tokens is None or batch.conv_inputs is None:
            raise RuntimeError("PLE candidates must be computed before staging")
        for (history, pending, context, width), prefix, inputs in zip(
            self.fields, (batch.ngram_history, batch.conv_history), (batch.input_tokens, batch.conv_inputs)
        ):
            if metadata.active_rows.numel():
                _stage[(metadata.active_rows.numel(), triton.cdiv((context + self.verify_width) * width, 256))](
                    prefix,
                    inputs,
                    metadata.starts,
                    metadata.active_rows,
                    pending,
                    context,
                    width,
                    self.verify_width,
                    256,
                )

    def commit(self, requests, starts, accepted_lengths, hold_request):
        for history, pending, context, width in self.fields:
            if starts.numel() and context:
                _commit[(starts.numel(), triton.cdiv(context * width, 256))](
                    pending,
                    history,
                    requests,
                    starts,
                    accepted_lengths,
                    hold_request,
                    context,
                    width,
                    self.verify_width,
                    256,
                )
