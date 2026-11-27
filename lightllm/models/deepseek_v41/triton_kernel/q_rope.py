import torch
import triton
import triton.language as tl

from lightllm.utils.custom_kernel_utis import get_pdl_launch_kwargs


@triton.jit
def _apply_attention_query_rope_kernel(
    query_input,
    query_output,
    rope_coefficients,
    position_ids,
    input_token_stride,
    output_token_stride,
    position_stride,
    NUM_QUERY_HEADS: tl.constexpr,
    HEAD_TILES_PER_TOKEN: tl.constexpr,
    HEADS_PER_TILE: tl.constexpr,
    COPY_NON_ROPE_DIMS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    QUERY_HEAD_DIM: tl.constexpr = 512
    ROPE_DIM: tl.constexpr = 64
    NON_ROPE_DIM: tl.constexpr = QUERY_HEAD_DIM - ROPE_DIM
    NUM_ROPE_PAIRS: tl.constexpr = ROPE_DIM // 2

    tile_index = tl.program_id(0)
    token_index = (tile_index // HEAD_TILES_PER_TOKEN).to(tl.int64)
    head_indices = (tile_index % HEAD_TILES_PER_TOKEN) * HEADS_PER_TILE + tl.arange(0, HEADS_PER_TILE)
    valid_head_mask = head_indices < NUM_QUERY_HEADS
    input_head_ptrs = query_input + token_index * input_token_stride + head_indices[:, None] * QUERY_HEAD_DIM
    output_head_ptrs = query_output + token_index * output_token_stride + head_indices[:, None] * QUERY_HEAD_DIM

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if COPY_NON_ROPE_DIMS:
        head_columns = tl.arange(0, QUERY_HEAD_DIM)
        non_rope_mask = valid_head_mask[:, None] & (head_columns[None, :] < NON_ROPE_DIM)
        non_rope_values = tl.load(input_head_ptrs + head_columns[None, :], mask=non_rope_mask)
        tl.store(output_head_ptrs + head_columns[None, :], non_rope_values, mask=non_rope_mask)

    rope_columns = tl.arange(0, ROPE_DIM)
    rope_values = tl.load(
        input_head_ptrs + NON_ROPE_DIM + rope_columns[None, :], mask=valid_head_mask[:, None], other=0
    ).to(tl.float32)
    query_real, query_imag = tl.split(tl.reshape(rope_values, (HEADS_PER_TILE, NUM_ROPE_PAIRS, 2)))
    token_position = tl.load(position_ids + token_index * position_stride)
    position_coefficients = tl.load(rope_coefficients + token_position * ROPE_DIM + rope_columns)
    cos, sin = tl.split(tl.reshape(position_coefficients, (NUM_ROPE_PAIRS, 2)))
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    rotated_rope_values = tl.reshape(
        tl.join(
            query_real * cos[None, :] - query_imag * sin[None, :],
            query_real * sin[None, :] + query_imag * cos[None, :],
        ),
        (HEADS_PER_TILE, ROPE_DIM),
    )
    tl.store(
        output_head_ptrs + NON_ROPE_DIM + rope_columns[None, :], rotated_rope_values, mask=valid_head_mask[:, None]
    )


@torch.no_grad()
def apply_attention_query_rope(query_input, query_output, freqs_cis, position_ids):
    """Rotate the last 64 dimensions of BF16 Q with shape [tokens, heads, 512].

    The head dimensions are contiguous and freqs_cis is contiguous complex64
    with shape [max_position, 32]. The input and output may use different token
    strides, or share the same layout for in-place rotation.
    """
    num_tokens, num_heads, _ = query_input.shape
    num_query_rows = num_tokens * num_heads
    if num_query_rows == 0:
        return
    copy_non_rope_dims = query_input.data_ptr() != query_output.data_ptr()
    if copy_non_rope_dims:
        heads_per_tile = triton.next_power_of_2(min(num_heads, 2 if num_query_rows <= 256 else 8))
        num_warps = max(1, heads_per_tile // 2)
    else:
        # In-place rotation can share each token's frequencies across more heads.
        heads_per_tile = triton.next_power_of_2(min(num_heads, 32))
        num_warps = 1 if heads_per_tile <= 8 else 4
    head_tiles_per_token = triton.cdiv(num_heads, heads_per_tile)
    _apply_attention_query_rope_kernel[(num_tokens * head_tiles_per_token,)](
        query_input,
        query_output,
        torch.view_as_real(freqs_cis),
        position_ids,
        query_input.stride(0),
        query_output.stride(0),
        position_ids.stride(0),
        NUM_QUERY_HEADS=num_heads,
        HEAD_TILES_PER_TOKEN=head_tiles_per_token,
        HEADS_PER_TILE=heads_per_tile,
        COPY_NON_ROPE_DIMS=copy_non_rope_dims,
        num_warps=num_warps,
        num_stages=1,
        **get_pdl_launch_kwargs(query_input.device),
    )
