import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["grid_w", "numel"])
def _pack_rgb_patches(pixels, lookup, patches, grid_w, numel, PATCH_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    patch_index = offset // (3 * PATCH_SIZE * PATCH_SIZE)
    channel = offset // (PATCH_SIZE * PATCH_SIZE) % 3
    row = patch_index // grid_w * PATCH_SIZE + offset // PATCH_SIZE % PATCH_SIZE
    col = patch_index % grid_w * PATCH_SIZE + offset % PATCH_SIZE
    pixel_index = (row * (grid_w * PATCH_SIZE) + col) * 3 + channel
    valid = offset < numel
    pixel = tl.load(pixels + pixel_index, mask=valid, other=0).to(tl.int32)
    value = tl.load(lookup + pixel)
    tl.store(patches + offset, value, mask=valid)


def pack_rgb_patches(pixels: torch.Tensor, lookup: torch.Tensor, grid_w: int, patch_size: int) -> torch.Tensor:
    """Pack resized HWC RGB bytes into contiguous BF16 patches on the GPU."""
    numel = pixels.numel()
    patches = torch.empty(
        (numel // (3 * patch_size ** 2), 3, patch_size, patch_size), dtype=torch.bfloat16, device=pixels.device
    )
    block = 1024
    _pack_rgb_patches[(triton.cdiv(numel, block),)](
        pixels, lookup, patches.view(torch.int16), grid_w, numel, patch_size, block
    )
    return patches
