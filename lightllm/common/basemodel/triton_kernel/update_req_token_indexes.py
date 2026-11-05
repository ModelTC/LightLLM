import triton
import triton.language as tl


@triton.jit
def _update_req_token_indexes(Table, Packet, STRIDE: tl.constexpr, BLOCK: tl.constexpr, PAGE_SIZE: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    req = tl.load(Packet + 4 * row)
    start = tl.load(Packet + 4 * row + 1)
    size = tl.load(Packet + 4 * row + 2)
    # Metadata offsets use token slots; the payload follows all four-column headers.
    offset = 4 * tl.num_programs(0) + tl.load(Packet + 4 * row + 3) // PAGE_SIZE
    pos = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Packet + offset + pos // PAGE_SIZE, pos < size, other=0) + pos % PAGE_SIZE
    tl.store(Table + req * STRIDE + start + pos, value, pos < size)


def update_req_token_indexes(table, packet, count, max_size, page_size=1):
    _update_req_token_indexes[(count, triton.cdiv(max_size, 256))](
        table, packet, table.stride(0), 256, page_size, num_warps=4
    )
