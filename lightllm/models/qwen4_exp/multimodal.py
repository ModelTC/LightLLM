import torch
import triton
import triton.language as tl


@triton.jit
def _canonical_image_ids(
    Tokens,
    Starts,
    Lengths,
    Output,
    COUNT: tl.constexpr,
    IMAGES: tl.constexpr,
    IMAGE_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Tokens + offsets, offsets < COUNT, other=0).to(tl.int64)
    canonical = values
    for image in range(IMAGES):
        start = tl.load(Starts + image).to(tl.int64)
        length = tl.load(Lengths + image).to(tl.int64)
        canonical = tl.where((values >= start) & (values < start + length), IMAGE_TOKEN, canonical)
    tl.store(Output + offsets, canonical, offsets < COUNT)


def canonical_image_ids(tokens, starts, lengths, image_token_id):
    if image_token_id is None:
        raise ValueError("Qwen4 image hashing requires the checkpoint image token ID")
    for value in (tokens, starts, lengths):
        if value.ndim != 1 or not value.is_cuda or not value.is_contiguous() or value.device != tokens.device:
            raise ValueError("Image token mapping requires contiguous CUDA vectors on one device")
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Image token mapping requires integer IDs and ranges")
    if starts.shape != lengths.shape:
        raise ValueError("Image cache starts and lengths must have matching shapes")
    if not starts.numel() or not tokens.numel():
        return tokens
    output = torch.empty_like(tokens)
    _canonical_image_ids[(triton.cdiv(tokens.numel(), 256),)](
        tokens, starts, lengths, output, tokens.numel(), starts.numel(), image_token_id, 256
    )
    return output
