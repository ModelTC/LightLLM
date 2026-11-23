import triton
import triton.language as tl
from packaging.version import Version


USE_LEGACY_BLOCK_POINTERS = tl.constexpr(Version(triton.__version__).release < (3, 9))


@triton.jit
def make_block_ptr(base, shape, strides, offsets, block_shape: tl.constexpr, order: tl.constexpr):
    # Keep the existing lowering on stable Triton; 3.9 removed block pointers.
    # The mask slot is unused here, but Triton cannot return None from a jit function.
    if USE_LEGACY_BLOCK_POINTERS:
        return tl.make_block_ptr(base, shape, strides, offsets, block_shape, order), 0
    else:
        tl.static_assert(len(block_shape) == 1 or len(block_shape) == 2)
        rows = offsets[0] + tl.arange(0, block_shape[0])
        if len(block_shape) == 1:
            return base + rows * strides[0], (rows >= 0) & (rows < shape[0])
        else:
            cols = offsets[1] + tl.arange(0, block_shape[1])
            pointers = base + rows[:, None] * strides[0] + cols[None, :] * strides[1]
            mask = ((rows >= 0) & (rows < shape[0]))[:, None] & ((cols >= 0) & (cols < shape[1]))[None, :]
            return pointers, mask


@triton.jit
def load_block(pointer, boundary_check: tl.constexpr):
    if USE_LEGACY_BLOCK_POINTERS:
        return tl.load(pointer[0], boundary_check=boundary_check)
    else:
        tl.static_assert(len(boundary_check) == len(pointer[0].shape))
        return tl.load(pointer[0], mask=pointer[1], other=0)


@triton.jit
def load_block_zero_padded(pointer, boundary_check: tl.constexpr):
    if USE_LEGACY_BLOCK_POINTERS:
        return tl.load(pointer[0], boundary_check=boundary_check, padding_option="zero")
    else:
        tl.static_assert(len(boundary_check) == len(pointer[0].shape))
        return tl.load(pointer[0], mask=pointer[1], other=0)


@triton.jit
def store_block(pointer, value, boundary_check: tl.constexpr):
    if USE_LEGACY_BLOCK_POINTERS:
        tl.store(pointer[0], value, boundary_check=boundary_check)
    else:
        tl.static_assert(len(boundary_check) == len(pointer[0].shape))
        tl.store(pointer[0], value, mask=pointer[1])
