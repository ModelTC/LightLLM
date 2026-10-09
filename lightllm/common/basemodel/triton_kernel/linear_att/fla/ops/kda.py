# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang

"""Packed chunkwise KDA prefill with per-channel decay gates.

q/k/g: [1, total_tokens, head_num, head_dim]; v: [1, total_tokens, head_num, value_head_dim].
For one head/chunk, chunk_size=64; Q/K: [chunk_size, head_dim]; V/E: [chunk_size, value_head_dim].
S: [head_dim, value_head_dim].
G is the chunk-local log2 prefix sum of the decay gate; P = exp2(G).
The forward pass consists of:
    1. Activate the gate and compute G independently for each chunk.
    2. Build the strictly lower-triangular update coupling Akk and causal Aqk.
    3. Compute R = (I + Akk)^-1, U = R @ (beta * V), W = R @ (beta * K * P),
       and Kg = K * exp2(G_last - G), independently for each chunk.
    4. Recur across chunks: E = U - W @ S_in;
       S_out = P_last[:, None] * S_in + Kg.T @ E. Save S_in for each chunk.
    5. Compute O = scale * (Q * P) @ S_in + Aqk @ E in parallel across chunks.
Here beta is broadcast over key/value channels. E is stored as v_new.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from lightllm.common.triton_utils.autotuner import autotune

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .cumsum import chunk_local_cumsum
from .index import prepare_chunk_indices
from .l2norm import l2norm_fwd
from triton.language import exp2, log
from .solve_tril import solve_tril


FLA_CHUNK_SIZE = 64
RCP_LN2 = 1.4426950216293335


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    return 1 if n < 1 else 1 << (n - 1).bit_length()


def kda_safe_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    gate_bias: torch.Tensor,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """GLM-5 bounded KDA decay in fp32.

    ``raw_gate`` is ``[..., heads, key_dim]``; ``a_log`` is per-head and
    ``gate_bias`` is per head/key coordinate.
    """

    head_num = a_log.numel()
    head_dim = gate_bias.numel() // head_num
    gate = raw_gate.float().view(*raw_gate.shape[:-1], head_num, head_dim)
    amplitude = a_log.float().reshape(*((1,) * (gate.ndim - 2)), head_num, 1).exp()
    bias = gate_bias.float().reshape(*((1,) * (gate.ndim - 2)), head_num, head_dim)
    return lower_bound * torch.sigmoid(amplitude * (gate + bias))


@triton.jit
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    q,
    k,
    g,
    beta,
    Akk,
    Aqk,
    scale,
    cu_seqlens,
    chunk_indices,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    subchunk_size: tl.constexpr,
    head_block_size: tl.constexpr,
    subchunk_num: tl.constexpr,
):
    """Build Akk/Aqk on off-diagonal [subchunk_size, subchunk_size] tiles below the chunk diagonal.

    One program handles (global chunk, row/column subtile, head). With g=G:
        Akk[i,j] = beta_i * sum_d(k_i[d] * k_j[d] * exp2(G_i[d] - G_j[d]))
        Aqk[i,j] = scale * sum_d(q_i[d] * k_j[d] * exp2(G_i[d] - G_j[d]))
    The row subtile is strictly after the column subtile, so every pair has i>j.
    Accumulate [subchunk_size, head_block_size] @ [head_block_size, subchunk_size] over head_dim.
    All chunks/heads/tiles are independent.
    """
    global_chunk_id, subtile_id, head_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row_subtile_id, col_subtile_id = subtile_id // subchunk_num, subtile_id % subchunk_num
    if row_subtile_id <= col_subtile_id:
        return

    seq_id = tl.load(chunk_indices + global_chunk_id * 2).to(tl.int32)
    chunk_id_in_seq = tl.load(chunk_indices + global_chunk_id * 2 + 1).to(tl.int32)
    seq_start = tl.load(cu_seqlens + seq_id).to(tl.int32)
    seq_end = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    chunk_start = chunk_id_in_seq * chunk_size
    row_start = chunk_start + row_subtile_id * subchunk_size
    col_start = chunk_start + col_subtile_id * subchunk_size
    if row_start >= seq_len:
        return

    q += (seq_start * head_num + head_id) * head_dim
    k += (seq_start * head_num + head_id) * head_dim
    g += (seq_start * head_num + head_id) * head_dim
    Akk += (seq_start * head_num + head_id) * chunk_size
    Aqk += (seq_start * head_num + head_id) * chunk_size

    p_beta = tl.make_block_ptr(
        base=beta + seq_start * head_num + head_id,
        shape=(seq_len,),
        strides=(head_num,),
        offsets=(row_start,),
        block_shape=(subchunk_size,),
        order=(0,),
    )
    beta_tile = tl.load(p_beta, boundary_check=(0,))

    kk_tile = tl.zeros([subchunk_size, subchunk_size], dtype=tl.float32)
    qk_tile = tl.zeros([subchunk_size, subchunk_size], dtype=tl.float32)
    for key_block_id in range(tl.cdiv(head_dim, head_block_size)):
        p_q = tl.make_block_ptr(
            base=q,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(row_start, key_block_id * head_block_size),
            block_shape=(subchunk_size, head_block_size),
            order=(1, 0),
        )
        p_k = tl.make_block_ptr(
            base=k,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(row_start, key_block_id * head_block_size),
            block_shape=(subchunk_size, head_block_size),
            order=(1, 0),
        )
        p_g = tl.make_block_ptr(
            base=g,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(row_start, key_block_id * head_block_size),
            block_shape=(subchunk_size, head_block_size),
            order=(1, 0),
        )
        p_k_col = tl.make_block_ptr(
            base=k,
            shape=(head_dim, seq_len),
            strides=(1, head_num * head_dim),
            offsets=(key_block_id * head_block_size, col_start),
            block_shape=(head_block_size, subchunk_size),
            order=(0, 1),
        )
        p_g_cols = tl.make_block_ptr(
            base=g,
            shape=(head_dim, seq_len),
            strides=(1, head_num * head_dim),
            offsets=(key_block_id * head_block_size, col_start),
            block_shape=(head_block_size, subchunk_size),
            order=(0, 1),
        )

        channel_offsets = key_block_id * head_block_size + tl.arange(0, head_block_size)
        valid_channels = channel_offsets < head_dim
        # Use the first row's G as a shared anchor: the two decay factors
        # multiply to exp2(G_i - G_j) without separately forming exp2(-G_j).
        # [head_block_size,]
        g_anchor = tl.load(g + row_start * head_num * head_dim + channel_offsets, mask=valid_channels, other=0)
        # [subchunk_size, head_block_size]
        g_rows = tl.load(p_g, boundary_check=(0, 1))
        gated_k_rows = tl.load(p_k, boundary_check=(0, 1)) * exp2(g_rows - g_anchor[None, :])
        # [head_block_size, subchunk_size]
        g_cols = tl.load(p_g_cols, boundary_check=(0, 1))
        k_cols = tl.load(p_k_col, boundary_check=(0, 1))
        # [head_block_size, subchunk_size]
        gated_k_cols = k_cols * exp2(g_anchor[:, None] - g_cols)
        kk_tile += tl.dot(gated_k_rows, gated_k_cols)

        q_rows = tl.load(p_q, boundary_check=(0, 1))
        gated_q_rows = q_rows * exp2(g_rows - g_anchor[None, :]) * scale
        qk_tile += tl.dot(gated_q_rows, gated_k_cols)

    kk_tile *= beta_tile[:, None]

    p_Akk = tl.make_block_ptr(
        base=Akk,
        shape=(seq_len, chunk_size),
        strides=(head_num * chunk_size, 1),
        offsets=(row_start, col_subtile_id * subchunk_size),
        block_shape=(subchunk_size, subchunk_size),
        order=(1, 0),
    )
    tl.store(p_Akk, kk_tile.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p_Aqk = tl.make_block_ptr(
        base=Aqk,
        shape=(seq_len, chunk_size),
        strides=(head_num * chunk_size, 1),
        offsets=(row_start, col_subtile_id * subchunk_size),
        block_shape=(subchunk_size, subchunk_size),
        order=(1, 0),
    )
    tl.store(p_Aqk, qk_tile.to(Aqk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    q,
    k,
    g,
    beta,
    Akk,
    Aqk,
    scale,
    cu_seqlens,
    chunk_indices,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    subchunk_size: tl.constexpr,
    head_block_size: tl.constexpr,
):
    """Complete Akk/Aqk inside each diagonal [subchunk_size, subchunk_size] subtile of a chunk.

    One program handles (global chunk, diagonal subtile, head), keeping Q/K/G
    tiles of shape [subchunk_size, head_block_size]. Each loop iteration fixes column j and reduces over head_dim
    for all subchunk_size rows. Akk uses i>j because E_i reads the state before its own write;
    Aqk uses i>=j because o_i reads the state after that write. Only Akk has beta_i.
    """
    global_chunk_id, subtile_id, head_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_id = tl.load(chunk_indices + global_chunk_id * 2).to(tl.int32)
    chunk_id_in_seq = tl.load(chunk_indices + global_chunk_id * 2 + 1).to(tl.int32)
    seq_start = tl.load(cu_seqlens + seq_id).to(tl.int32)
    seq_end = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    chunk_start = chunk_id_in_seq * chunk_size
    row_start = chunk_start + subtile_id * subchunk_size
    if row_start >= seq_len:
        return

    row_offsets = tl.arange(0, subchunk_size)
    channel_offsets = tl.arange(0, head_block_size)
    valid_channels = channel_offsets < head_dim
    valid_rows = (row_start + row_offsets) < seq_len
    output_offsets = (
        (seq_start + row_start + row_offsets) * head_num * chunk_size
        + head_id * chunk_size
        + subtile_id * subchunk_size
    )

    p_q = tl.make_block_ptr(
        base=q + (seq_start * head_num + head_id) * head_dim,
        shape=(seq_len, head_dim),
        strides=(head_num * head_dim, 1),
        offsets=(row_start, 0),
        block_shape=(subchunk_size, head_block_size),
        order=(1, 0),
    )
    p_k = tl.make_block_ptr(
        base=k + (seq_start * head_num + head_id) * head_dim,
        shape=(seq_len, head_dim),
        strides=(head_num * head_dim, 1),
        offsets=(row_start, 0),
        block_shape=(subchunk_size, head_block_size),
        order=(1, 0),
    )
    p_g = tl.make_block_ptr(
        base=g + (seq_start * head_num + head_id) * head_dim,
        shape=(seq_len, head_dim),
        strides=(head_num * head_dim, 1),
        offsets=(row_start, 0),
        block_shape=(subchunk_size, head_block_size),
        order=(1, 0),
    )
    q_rows = tl.load(p_q, boundary_check=(0, 1))
    beta_k_rows = tl.load(p_k, boundary_check=(0, 1))
    g_rows = tl.load(p_g, boundary_check=(0, 1))

    p_beta = beta + (seq_start + row_start + row_offsets) * head_num + head_id
    beta_k_rows = beta_k_rows * tl.load(p_beta, mask=valid_rows, other=0)[:, None]

    p_k_col = k + (seq_start + row_start) * head_num * head_dim + head_id * head_dim + channel_offsets
    p_g_col = g + (seq_start + row_start) * head_num * head_dim + head_id * head_dim + channel_offsets

    for j in range(0, min(subchunk_size, seq_len - row_start)):
        k_col = tl.load(p_k_col, mask=valid_channels, other=0).to(tl.float32)
        g_col = tl.load(p_g_col, mask=valid_channels, other=0).to(tl.float32)
        decayed_k_col = k_col[None, :] * exp2(g_rows - g_col[None, :])
        kk_col = tl.sum(beta_k_rows * decayed_k_col, 1)
        kk_col = tl.where(row_offsets > j, kk_col, 0.0)
        qk_col = tl.sum(q_rows * decayed_k_col, 1)
        qk_col = tl.where(row_offsets >= j, qk_col * scale, 0.0)
        tl.store(Akk + output_offsets + j, kk_col, mask=valid_rows)
        tl.store(Aqk + output_offsets + j, qk_col, mask=valid_rows)
        p_k_col += head_num * head_dim
        p_g_col += head_num * head_dim


def _get_kda_kkt_sub_inter_configs():
    return [
        {"head_block_size": head_block_size, "num_warps": num_warps, "num_stages": num_stages}
        for head_block_size in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ]


def _get_kda_kkt_sub_intra_configs():
    return [{"num_warps": num_warps} for num_warps in [1, 2, 4, 8]]


def _get_kda_kkt_static_key(k, gk, beta, Akk):
    return {
        "head_num": k.shape[2],
        "head_dim": k.shape[3],
        "chunk_size": Akk.shape[-1],
        "dtype": str(k.dtype).removeprefix("torch."),
        "gate_dtype": str(gk.dtype).removeprefix("torch."),
        "beta_dtype": str(beta.dtype).removeprefix("torch."),
        "out_dtype": str(Akk.dtype).removeprefix("torch."),
    }


@autotune(
    kernel_name="chunk_kda_scaled_dot_kkt_sub_inter:v1",
    configs_gen_func=_get_kda_kkt_sub_inter_configs,
    static_key_func=_get_kda_kkt_static_key,
    run_key_func=lambda k: k.shape[1],
)
def _chunk_kda_scaled_dot_kkt_sub_inter(q, k, gk, beta, Akk, Aqk, scale, cu_seqlens, chunk_indices, run_config=None):
    """Tune and launch off-diagonal Akk/Aqk tiles, overwriting only those output tiles."""
    head_num, head_dim = k.shape[-2:]
    chunk_size = Akk.shape[-1]
    subchunk_size = min(16, chunk_size)
    subchunk_num = cdiv(chunk_size, subchunk_size)
    chunk_num = len(chunk_indices)
    if run_config is None:
        run_config = {"head_block_size": 64, "num_warps": 4, "num_stages": 2}

    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[(chunk_num, subchunk_num * subchunk_num, head_num)](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Akk=Akk,
        Aqk=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_num=head_num,
        head_dim=head_dim,
        chunk_size=chunk_size,
        subchunk_size=subchunk_size,
        head_block_size=run_config.get("head_block_size", 64),
        subchunk_num=subchunk_num,
        num_warps=run_config.get("num_warps", 4),
        num_stages=run_config.get("num_stages", 2),
    )


@autotune(
    kernel_name="chunk_kda_scaled_dot_kkt_sub_intra:v1",
    configs_gen_func=_get_kda_kkt_sub_intra_configs,
    static_key_func=_get_kda_kkt_static_key,
    run_key_func=lambda k: k.shape[1],
)
def _chunk_kda_scaled_dot_kkt_sub_intra(q, k, gk, beta, Akk, Aqk, scale, cu_seqlens, chunk_indices, run_config=None):
    """Tune and launch diagonal Akk/Aqk subtiles independently of off-diagonal tiles."""
    head_num, head_dim = k.shape[-2:]
    chunk_size = Akk.shape[-1]
    subchunk_size = min(16, chunk_size)
    chunk_num = len(chunk_indices)
    if run_config is None:
        run_config = {"num_warps": 4}

    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra[(chunk_num, cdiv(chunk_size, subchunk_size), head_num)](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Akk=Akk,
        Aqk=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_num=head_num,
        head_dim=head_dim,
        chunk_size=chunk_size,
        subchunk_size=subchunk_size,
        head_block_size=max(next_power_of_2(head_dim), 16),
        num_warps=run_config.get("num_warps", 4),
    )


def chunk_kda_scaled_dot_kkt_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the update coupling Akk and the output weights Aqk for every chunk.

    q/k/gk: [1, total_tokens, head_num, head_dim], where gk is the chunk-local log2 prefix sum G.
    beta: [1, total_tokens, head_num]. Both returned tensors have shape [1, total_tokens, head_num, chunk_size]; each
    token row stores weights for the chunk_size positions in its own chunk:
        Akk[i,j] = beta_i * sum_d(k_i[d] * k_j[d] * exp2(G_i[d] - G_j[d])), i>j.
        Aqk[i,j] = scale * sum_d(q_i[d] * k_j[d] * exp2(G_i[d] - G_j[d])), i>=j.
    Entries outside the respective causal masks are zero. solve_tril computes R=(I+Akk)^-1;
    Aqk is retained for O = scale * (Q * exp2(G)) @ S_in + Aqk @ E.
    The two subtile kernels have separate LightLLM autotune configurations.
    """
    _, total_tokens, head_num, head_dim = k.shape
    assert head_dim <= 256
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    Akk = torch.zeros(1, total_tokens, head_num, chunk_size, device=k.device, dtype=output_dtype)
    Aqk = torch.zeros(1, total_tokens, head_num, chunk_size, device=k.device, dtype=output_dtype)
    _chunk_kda_scaled_dot_kkt_sub_inter(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Akk=Akk,
        Aqk=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    _chunk_kda_scaled_dot_kkt_sub_intra(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Akk=Akk,
        Aqk=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return Akk, Aqk


@triton.jit
def recompute_w_u_fwd_kernel(
    k,
    kg,
    v,
    beta,
    w,
    u,
    R,
    gk,
    cu_seqlens,
    chunk_indices,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    value_head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    head_block_size: tl.constexpr,
    value_block_size: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Precompute U/W/Kg from R=(I+Akk)^-1 without reading recurrent state.

    For one chunk/head, P=exp2(gk); R: [chunk_size, chunk_size];
    K/P: [chunk_size, head_dim]; V: [chunk_size, value_head_dim]:
        U = R @ (beta[:, None] * V)            # [chunk_size, value_head_dim]
        W = R @ (beta[:, None] * K * P)        # [chunk_size, head_dim]
        Kg = K * exp2(G_last - G)              # [chunk_size, head_dim], writes decayed to chunk end
    Later E = U - W @ S_in and S_out = P_last[:, None] * S_in + Kg.T @ E.
    One program handles (global chunk, head), looping over blocks of key/value channels.
    Q * P is computed directly in the output kernel.
    """
    global_chunk_id, head_id = tl.program_id(0), tl.program_id(1)
    seq_id = tl.load(chunk_indices + global_chunk_id * 2).to(tl.int32)
    chunk_id_in_seq = tl.load(chunk_indices + global_chunk_id * 2 + 1).to(tl.int32)
    seq_start = tl.load(cu_seqlens + seq_id).to(tl.int32)
    seq_end = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    chunk_start = chunk_id_in_seq * chunk_size
    last_idx = min(chunk_start + chunk_size, seq_len) - 1
    p_beta = tl.make_block_ptr(
        base=beta + seq_start * head_num + head_id,
        shape=(seq_len,),
        strides=(head_num,),
        offsets=(chunk_start,),
        block_shape=(chunk_size,),
        order=(0,),
    )
    beta_tile = tl.load(p_beta, boundary_check=(0,))

    p_r = tl.make_block_ptr(
        base=R + (seq_start * head_num + head_id) * chunk_size,
        shape=(seq_len, chunk_size),
        strides=(head_num * chunk_size, 1),
        offsets=(chunk_start, 0),
        block_shape=(chunk_size, chunk_size),
        order=(1, 0),
    )
    r_tile = tl.load(p_r, boundary_check=(0, 1))

    for value_block_id in range(tl.cdiv(value_head_dim, value_block_size)):
        p_v = tl.make_block_ptr(
            base=v + (seq_start * head_num + head_id) * value_head_dim,
            shape=(seq_len, value_head_dim),
            strides=(head_num * value_head_dim, 1),
            offsets=(chunk_start, value_block_id * value_block_size),
            block_shape=(chunk_size, value_block_size),
            order=(1, 0),
        )
        p_u = tl.make_block_ptr(
            base=u + (seq_start * head_num + head_id) * value_head_dim,
            shape=(seq_len, value_head_dim),
            strides=(head_num * value_head_dim, 1),
            offsets=(chunk_start, value_block_id * value_block_size),
            block_shape=(chunk_size, value_block_size),
            order=(1, 0),
        )
        v_tile = tl.load(p_v, boundary_check=(0, 1))
        weighted_v = (v_tile * beta_tile[:, None]).to(v_tile.dtype)
        # U: [chunk_size, chunk_size] @ [chunk_size, value_block_size] -> [chunk_size, value_block_size].
        u_tile = tl.dot(r_tile, weighted_v, input_precision=DOT_PRECISION)
        tl.store(p_u, u_tile.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for key_block_id in range(tl.cdiv(head_dim, head_block_size)):
        p_w = tl.make_block_ptr(
            base=w + (seq_start * head_num + head_id) * head_dim,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(chunk_start, key_block_id * head_block_size),
            block_shape=(chunk_size, head_block_size),
            order=(1, 0),
        )
        p_k = tl.make_block_ptr(
            base=k + (seq_start * head_num + head_id) * head_dim,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(chunk_start, key_block_id * head_block_size),
            block_shape=(chunk_size, head_block_size),
            order=(1, 0),
        )
        k_tile = tl.load(p_k, boundary_check=(0, 1))
        weighted_k = k_tile * beta_tile[:, None]

        p_gk = tl.make_block_ptr(
            base=gk + (seq_start * head_num + head_id) * head_dim,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(chunk_start, key_block_id * head_block_size),
            block_shape=(chunk_size, head_block_size),
            order=(1, 0),
        )
        g_tile = tl.load(p_gk, boundary_check=(0, 1))
        weighted_k *= exp2(g_tile)

        # Kg carries each write forward to the last valid token of this chunk.
        channel_offsets = key_block_id * head_block_size + tl.arange(0, head_block_size)
        valid_channels = channel_offsets < head_dim
        g_last = tl.load(
            gk + ((seq_start + last_idx) * head_num + head_id) * head_dim + channel_offsets,
            mask=valid_channels,
            other=0.0,
        )
        kg_tile = k_tile * exp2(g_last - g_tile)
        p_kg = tl.make_block_ptr(
            base=kg + (seq_start * head_num + head_id) * head_dim,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(chunk_start, key_block_id * head_block_size),
            block_shape=(chunk_size, head_block_size),
            order=(1, 0),
        )
        tl.store(p_kg, kg_tile.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        # W maps the chunk-entry state to corrections: [chunk_size, chunk_size] @ [chunk_size, head_block_size].
        w_tile = tl.dot(r_tile, weighted_k.to(k_tile.dtype))
        tl.store(p_w, w_tile.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def _get_kda_w_u_configs():
    return [{"num_warps": num_warps, "num_stages": num_stages} for num_warps in [2, 4, 8] for num_stages in [2, 3, 4]]


def _get_kda_w_u_static_key(k, v, beta, R, gk):
    return {
        "head_num": k.shape[2],
        "head_dim": k.shape[3],
        "value_head_dim": v.shape[-1],
        "chunk_size": R.shape[-1],
        "dtype": str(k.dtype).removeprefix("torch."),
        "v_dtype": str(v.dtype).removeprefix("torch."),
        "r_dtype": str(R.dtype).removeprefix("torch."),
        "gate_dtype": str(gk.dtype).removeprefix("torch."),
        "beta_dtype": str(beta.dtype).removeprefix("torch."),
    }


@autotune(
    kernel_name="kda_recompute_w_u_fwd:v1",
    configs_gen_func=_get_kda_w_u_configs,
    static_key_func=_get_kda_w_u_static_key,
    run_key_func=lambda k: k.shape[1],
)
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    R: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    run_config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return W/U/Kg for the packed chunk recurrence.

    R: [1, total_tokens, head_num, chunk_size] stores (I+Akk)^-1; gk: [1, total_tokens, head_num, head_dim] stores G.
    W/Kg: [1, total_tokens, head_num, head_dim]; U: [1, total_tokens, head_num, value_head_dim].
    These factors depend only on the current chunk, so grid (chunk_num, head_num) computes all chunks
    independently before the state scan. LightLLM tunes num_warps/num_stages;
    the channel tiles head_block_size=value_block_size=64 stay fixed.
    """
    head_num, head_dim = k.shape[-2:]
    value_head_dim = v.shape[-1]
    chunk_size = R.shape[-1]
    head_block_size = 64
    value_block_size = 64
    if run_config is None:
        run_config = {"num_warps": 4, "num_stages": 2}

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_num = len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    recompute_w_u_fwd_kernel[(chunk_num, head_num)](
        k=k,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        R=R,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_num=head_num,
        head_dim=head_dim,
        value_head_dim=value_head_dim,
        chunk_size=chunk_size,
        head_block_size=head_block_size,
        value_block_size=value_block_size,
        DOT_PRECISION="ieee",
        num_warps=run_config.get("num_warps", 4),
        num_stages=run_config.get("num_stages", 2),
    )
    return w, u, kg


@triton.jit
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    Aqk,
    cu_seqlens,
    chunk_indices,
    scale,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    value_head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    head_block_size: tl.constexpr,
    value_block_size: tl.constexpr,
):
    """Compute O = scale * (Q * exp2(G)) @ S_in + Aqk @ E for each chunk.

    h holds the state before each chunk; v holds E (v_new); Aqk holds the output weights.
    One program handles (value channel block, global chunk, head), producing [chunk_size, value_block_size].
    The history term reduces [chunk_size, head_block_size] @ [head_block_size, value_block_size] over head_dim.
    The local term is [chunk_size, chunk_size] @ [chunk_size, value_block_size].
    All chunks can run independently after h/E are saved.
    """
    value_block_id, global_chunk_id, head_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_id = tl.load(chunk_indices + global_chunk_id * 2).to(tl.int32)
    chunk_id_in_seq = tl.load(chunk_indices + global_chunk_id * 2 + 1).to(tl.int32)
    seq_start = tl.load(cu_seqlens + seq_id).to(tl.int32)
    seq_end = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    chunk_start = chunk_id_in_seq * chunk_size

    causal_mask = tl.arange(0, chunk_size)[:, None] >= tl.arange(0, chunk_size)[None, :]

    output_tile = tl.zeros([chunk_size, value_block_size], dtype=tl.float32)
    for key_block_id in range(tl.cdiv(head_dim, head_block_size)):
        p_q = tl.make_block_ptr(
            base=q + (seq_start * head_num + head_id) * head_dim,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(chunk_start, key_block_id * head_block_size),
            block_shape=(chunk_size, head_block_size),
            order=(1, 0),
        )
        p_g = tl.make_block_ptr(
            base=g + (seq_start * head_num + head_id) * head_dim,
            shape=(seq_len, head_dim),
            strides=(head_num * head_dim, 1),
            offsets=(chunk_start, key_block_id * head_block_size),
            block_shape=(chunk_size, head_block_size),
            order=(1, 0),
        )
        p_h = tl.make_block_ptr(
            base=h + (global_chunk_id * head_num + head_id) * head_dim * value_head_dim,
            shape=(head_dim, value_head_dim),
            strides=(value_head_dim, 1),
            offsets=(key_block_id * head_block_size, value_block_id * value_block_size),
            block_shape=(head_block_size, value_block_size),
            order=(1, 0),
        )

        # [chunk_size, head_block_size]
        q_tile = tl.load(p_q, boundary_check=(0, 1))
        q_tile = (q_tile * scale).to(q_tile.dtype)
        # [chunk_size, head_block_size]
        g_tile = tl.load(p_g, boundary_check=(0, 1))
        # [chunk_size, head_block_size]
        gated_q_tile = (q_tile * exp2(g_tile)).to(q_tile.dtype)
        # Chunk-entry state tile S_in: [head_block_size, value_block_size].
        state_tile = tl.load(p_h, boundary_check=(0, 1))
        # Historical contribution: scale * (Q * exp2(G)) @ S_in, [chunk_size, value_block_size].
        output_tile += tl.dot(gated_q_tile, state_tile.to(gated_q_tile.dtype))
    p_e = tl.make_block_ptr(
        base=v + (seq_start * head_num + head_id) * value_head_dim,
        shape=(seq_len, value_head_dim),
        strides=(head_num * value_head_dim, 1),
        offsets=(chunk_start, value_block_id * value_block_size),
        block_shape=(chunk_size, value_block_size),
        order=(1, 0),
    )
    p_o = tl.make_block_ptr(
        base=o + (seq_start * head_num + head_id) * value_head_dim,
        shape=(seq_len, value_head_dim),
        strides=(head_num * value_head_dim, 1),
        offsets=(chunk_start, value_block_id * value_block_size),
        block_shape=(chunk_size, value_block_size),
        order=(1, 0),
    )
    p_Aqk = tl.make_block_ptr(
        base=Aqk + (seq_start * head_num + head_id) * chunk_size,
        shape=(seq_len, chunk_size),
        strides=(head_num * chunk_size, 1),
        offsets=(chunk_start, 0),
        block_shape=(chunk_size, chunk_size),
        order=(1, 0),
    )
    # E: [chunk_size, value_block_size], already includes beta and earlier-token corrections.
    residual_tile = tl.load(p_e, boundary_check=(0, 1))
    # Aqk: [chunk_size, chunk_size], causal including the diagonal; scale is already included.
    qk_tile = tl.load(p_Aqk, boundary_check=(0, 1))
    qk_tile = tl.where(causal_mask, qk_tile, 0.0).to(residual_tile.dtype)
    output_tile += tl.dot(qk_tile, residual_tile, allow_tf32=False)  # Current-chunk contribution Aqk @ E.
    tl.store(p_o, output_tile.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _get_kda_output_configs():
    return [
        {
            "head_block_size": head_block_size,
            "value_block_size": value_block_size,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        for head_block_size in [32, 64]
        for value_block_size in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ]


def _get_kda_output_static_key(q, v, g, Aqk, h, o, chunk_size):
    return {
        "head_num": q.shape[2],
        "head_dim": q.shape[3],
        "value_head_dim": v.shape[-1],
        "chunk_size": chunk_size,
        "dtype": str(q.dtype).removeprefix("torch."),
        "v_dtype": str(v.dtype).removeprefix("torch."),
        "gate_dtype": str(g.dtype).removeprefix("torch."),
        "aqk_dtype": str(Aqk.dtype).removeprefix("torch."),
        "state_dtype": str(h.dtype).removeprefix("torch."),
        "out_dtype": str(o.dtype).removeprefix("torch."),
    }


@autotune(
    kernel_name="kda_chunk_gla_fwd_o_gk:v1",
    configs_gen_func=_get_kda_output_configs,
    static_key_func=_get_kda_output_static_key,
    run_key_func=lambda q: q.shape[1],
)
def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    Aqk: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    run_config: dict | None = None,
):
    """Write all chunk outputs from saved entry states h and corrections v=E.

    q/g: [1, total_tokens, head_num, head_dim]; v/o: [1, total_tokens, head_num, value_head_dim].
    Aqk: [1, total_tokens, head_num, chunk_size].
    h: [1, chunk_num, head_num, head_dim, value_head_dim], where chunk_num is the total chunk count across all requests.
    The caller supplies the output buffer o. Each launch fully overwrites it,
    allowing repeated tuning runs without changing h or E. LightLLM selects
    head_block_size/value_block_size and launch parameters.
    Grid: (ceil(value_head_dim/value_block_size), chunk_num, head_num).
    """
    head_num, head_dim = q.shape[-2:]
    value_head_dim = v.shape[-1]

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_num = len(chunk_indices)

    if run_config is None:
        run_config = {"head_block_size": 64, "value_block_size": 64, "num_warps": 4, "num_stages": 2}
    head_block_size = run_config.get("head_block_size", 64)
    value_block_size = run_config.get("value_block_size", 64)

    grid = (cdiv(value_head_dim, value_block_size), chunk_num, head_num)
    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        Aqk=Aqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        head_num=head_num,
        head_dim=head_dim,
        value_head_dim=value_head_dim,
        chunk_size=chunk_size,
        head_block_size=head_block_size,
        value_block_size=value_block_size,
        num_warps=run_config.get("num_warps", 4),
        num_stages=run_config.get("num_stages", 2),
    )
    return o


@triton.heuristics({"HAS_BIAS": lambda args: args["g_bias"] is not None})
@triton.jit
def kda_gate_cumsum_fwd_kernel(
    g,
    A_log,
    y,
    g_bias,
    cu_seqlens,
    chunk_indices,
    # Element strides for input/output [total_tokens, head_num, head_dim]: token, head, channel.
    stride_g_token: tl.constexpr,
    stride_g_head: tl.constexpr,
    stride_g_dim: tl.constexpr,
    stride_y_token: tl.constexpr,
    stride_y_head: tl.constexpr,
    stride_y_dim: tl.constexpr,
    cumsum_scale,
    beta,
    threshold,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    head_block_size: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Fuse raw-gate activation with G_i = sum_{r<=i} ell_r / ln(2) in each chunk.

    g/y: [total_tokens, head_num, head_dim]; A_log: [head_num].
    Activate ell with the bounded sigmoid or negative softplus, then multiply
    a [chunk_size, chunk_size] lower-triangular matrix of ones by ell: [chunk_size, head_block_size]. The resulting
    log2 prefixes reset at every sequence/chunk boundary. beta here is the
    scalar softplus parameter, not the per-token KDA update strength.
    """
    # One program handles one [chunk_size, head_block_size] tile for one request/head.
    head_block_id, global_chunk_id, head_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # chunk_indices[global_chunk_id] = (seq_id, chunk_id_in_seq).
    seq_id = tl.load(chunk_indices + global_chunk_id * 2).to(tl.int32)
    chunk_id_in_seq = tl.load(chunk_indices + global_chunk_id * 2 + 1).to(tl.int32)
    seq_start = tl.load(cu_seqlens + seq_id).to(tl.int32)
    seq_end = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    chunk_start = chunk_id_in_seq * chunk_size
    head_dim_start = head_block_id * head_block_size

    # Fix the request/head, then view [total_tokens, head_num, head_dim] as a [seq_len, head_dim] matrix.
    # Moving one token/channel advances by stride_*_token/stride_*_dim elements.
    g_seq_head = g + seq_start * stride_g_token + head_id * stride_g_head
    y_seq_head = y + seq_start * stride_y_token + head_id * stride_y_head
    p_g = tl.make_block_ptr(
        base=g_seq_head,
        shape=(seq_len, head_dim),
        strides=(stride_g_token, stride_g_dim),
        offsets=(chunk_start, head_dim_start),
        block_shape=(chunk_size, head_block_size),
        order=(1, 0),
    )
    p_y = tl.make_block_ptr(
        base=y_seq_head,
        shape=(seq_len, head_dim),
        strides=(stride_y_token, stride_y_dim),
        offsets=(chunk_start, head_dim_start),
        block_shape=(chunk_size, head_block_size),
        order=(1, 0),
    )

    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if HAS_BIAS:
        head_dim_indices = head_dim_start + tl.arange(0, head_block_size)
        b_bias = tl.load(
            g_bias + head_id * head_dim + head_dim_indices, mask=head_dim_indices < head_dim, other=0.0
        ).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    b_a = tl.load(A_log + head_id).to(tl.float32)
    b_a = tl.exp(b_a) if SAFE_GATE else -tl.exp(b_a)
    if SAFE_GATE:
        # log_gate = lower_bound * sigmoid(exp(A_log) * (raw_g + bias)).
        # For lower_bound < 0, log_gate is in [lower_bound, 0]; decay = exp(log_gate).
        b_gate = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        b_g_scaled = b_g * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_g,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = b_a * b_softplus

    # Out-of-bounds rows (load returns 0, but softplus/bias can still make
    # b_gate non-zero) participate in the dot product. They only contribute to
    # out-of-bounds output rows, which are masked away by `boundary_check` on
    # the store, so visible output matches unfused gate + chunk-local cumsum.
    o_t = tl.arange(0, chunk_size)
    m_cumsum = tl.where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)
    b_y = tl.dot(m_cumsum, b_gate, allow_tf32=False) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def _get_kda_gate_cumsum_configs():
    return [
        {"head_block_size": head_block_size, "num_warps": num_warps}
        for head_block_size in [32, 64]
        for num_warps in [2, 4, 8]
    ]


def _get_kda_gate_cumsum_static_key(raw_g, g_bias, chunk_size, output_dtype, safe_gate):
    return {
        "head_num": raw_g.shape[1],
        "head_dim": raw_g.shape[2],
        "chunk_size": chunk_size,
        "SAFE_GATE": safe_gate,
        "HAS_BIAS": g_bias is not None,
        "dtype": str(raw_g.dtype).removeprefix("torch."),
        "out_dtype": str(output_dtype or raw_g.dtype).removeprefix("torch."),
    }


@autotune(
    kernel_name="fused_kda_gate_chunk_cumsum:v1",
    configs_gen_func=_get_kda_gate_cumsum_configs,
    static_key_func=_get_kda_gate_cumsum_static_key,
    run_key_func=lambda raw_g: raw_g.shape[0],  # Total packed token count.
)
def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    cu_seqlens: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    run_config: dict | None = None,
) -> torch.Tensor:
    """Activate packed decay gates and return chunk-local log2 prefix sums in [total_tokens, head_num, head_dim].

    raw_g: [total_tokens, head_num, head_dim], packed tokens, local heads, and key channels.
        Input/output addressing uses each tensor's strides, measured in elements.
    A_log: [head_num]; g_bias: [head_num * head_dim] or [head_num, head_dim], or None to skip the bias.
    cu_seqlens: [request_num + 1], required token boundaries for request_num packed requests.
    run_config: optional LightLLM autotune config with head_block_size and num_warps.

    Returns G_i = sum_{r=chunk_start..i} ell_r / ln(2), where ell is the
    activated log decay. Subsequent kernels use P_i=exp2(G_i) for entry-state
    decay and exp2(G_i-G_j) for writes propagated from token j to token i.
    """
    assert raw_g.ndim == 3, "raw_g must have packed shape [total_tokens, head_num, head_dim]"
    assert cu_seqlens is not None, "cu_seqlens is required for packed KDA prefill"
    head_num, head_dim = raw_g.shape[1:]
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_num = len(chunk_indices)

    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    y = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)

    if run_config is None:
        run_config = {"head_block_size": 32, "num_warps": 4}
    head_block_size = run_config.get("head_block_size", 32)
    num_warps = run_config.get("num_warps", 4)

    grid = (cdiv(head_dim, head_block_size), chunk_num, head_num)
    kda_gate_cumsum_fwd_kernel[grid](
        g=raw_g,
        A_log=A_log,
        y=y,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        stride_g_token=raw_g.stride(0),
        stride_g_head=raw_g.stride(1),
        stride_g_dim=raw_g.stride(2),
        stride_y_token=y.stride(0),
        stride_y_head=y.stride(1),
        stride_y_dim=y.stride(2),
        # RCP_LN2 folds in the natural-log -> log2 conversion so downstream
        # exp2-based kernels reproduce exp(g). Keep this in sync with the
        # `use_exp2=True` path in `_chunk_kda_fwd_with_cumulative_g`.
        cumsum_scale=RCP_LN2,
        beta=beta,
        threshold=threshold,
        SAFE_GATE=safe_gate,
        LOWER_BOUND=lower_bound,
        head_dim=head_dim,
        chunk_size=chunk_size,
        head_block_size=head_block_size,
        num_warps=num_warps,
    )
    return y


def _chunk_kda_fwd_with_cumulative_g(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    """Compute KDA state updates and outputs from gate prefix sums.

    q/k/g: [1, total_tokens, head_num, head_dim]; v: [1, total_tokens, head_num, value_head_dim].
    beta: [1, total_tokens, head_num].

    Execution order and parallelism:
        1. Prepare Akk, Aqk, R, U, W, and Kg using only each chunk's inputs.
           Each of these stages can process different chunks in parallel.
        2. Update the state in chunk order within each request: one chunk's output state
           becomes the next chunk's input state. Save the state before processing each
           chunk, S_in, in h. Compute E = U - W @ S_in and save it in v_new;
           E contains the value correction used in each token's state update.
           Different requests, heads, and blocks of value channels can run in parallel.
        3. Once h and v_new are saved, compute all chunk outputs in parallel using these tensors.

    h: [1, chunk_num, head_num, head_dim, value_head_dim]; v_new: [1, total_tokens, head_num, value_head_dim].
    The output kernel writes into v, overwriting its original values. The returned o shares v's memory.
    """
    # 1. Build Akk and Aqk for all chunks: [1, total_tokens, head_num, chunk_size], where chunk_size is the chunk size.
    # Akk weights earlier writes in each token's value correction; Aqk weights writes in the chunk output.
    # Keep Aqk in fp32 until the output matrix multiplication. Only Akk is passed to solve_tril.
    Akk, Aqk = chunk_kda_scaled_dot_kkt_fwd(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    # 2. Compute R=(I+Akk)^-1, then use matrix multiplication to obtain U/W for all tokens in each chunk.
    R = solve_tril(
        A=Akk,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    del Akk
    # 3. Prepare U/W/Kg for the state update without reading the input state S_in.
    # U=R@(beta*V), W=R@(beta*K*exp2(G)), Kg=K*exp2(G_last-G).
    w, u, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        R=R,
        gk=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    del R
    # 4. Update state in chunk order within each request. Requests, heads, and blocks of value channels run in parallel.
    # Save the state before each chunk, S_in, in h; compute E=U-W@S_in and write it to v_new.
    # S_out=exp2(G_last)[:, None]*S_in + Kg.T@E becomes the next chunk's S_in.
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        use_exp2=True,
    )
    del w, u, kg
    # 5. Read each chunk's saved h=S_in and v_new=E to compute outputs in parallel:
    # O=scale*(Q*exp2(G))@S_in + Aqk@E combines the input state and the writes inside the chunk.
    # Passing o=v overwrites the original values, which are no longer needed at this stage.
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        Aqk=Aqk,
        h=h,
        o=v,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    del Aqk, v_new, h
    return o, final_state


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
):
    """Convert activated log decays g=ell to chunk-local log2 prefixes, then run KDA.

    g: [1, total_tokens, head_num, head_dim] contains per-token natural-log decays, not raw gate logits.
    chunk_local_cumsum followed by RCP_LN2 supplies G to the shared chunk path.
    """
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # KDA evaluates cumulative gate decays with exp2. Convert from natural-log
    # space so exp(x) is preserved as exp2(x / ln(2)).
    g = g * RCP_LN2
    return _chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
):
    """Activate packed raw gates and build G in one kernel before the shared KDA path.

    raw_g: [1, total_tokens, head_num, head_dim]; cu_seqlens separates requests. The gate kernel uses
    the [total_tokens, head_num, head_dim] view and returns fp32 chunk-local log2 prefixes of the same shape.
    """
    assert (
        raw_g.ndim == 4 and raw_g.shape[0] == 1
    ), "KDA prefill expects packed gates shaped [1, total_tokens, head_num, head_dim]"
    chunk_size = FLA_CHUNK_SIZE
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # The gate kernel uses [total_tokens, head_num, head_dim]; downstream FLA ops add a leading batch dimension.
    # Removing/restoring the leading dimension only creates tensor views.
    g = fused_kda_gate_chunk_cumsum(
        raw_g.squeeze(0),
        A_log=A_log,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    ).unsqueeze(0)
    return _chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    assert q.ndim == 4 and q.shape[0] == 1, "KDA prefill expects packed q shaped [1, total_tokens, head_num, head_dim]"
    assert cu_seqlens is not None, "cu_seqlens is required for packed KDA prefill"
    if scale is None:
        head_dim = k.shape[-1]
        scale = head_dim ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    o, final_state = chunk_kda_fwd(
        q=q,
        k=k,
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state.contiguous(),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state


def chunk_kda_with_fused_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    **kwargs,
):
    """KDA prefill with 64-token chunks and fused gate activation.

    Shapes:
        q/k/raw_g: [1, total_tokens, head_num, head_dim]; v: [1, total_tokens, head_num, value_head_dim].
        beta: [1, total_tokens, head_num].
        A_log: [head_num]; g_bias: [head_num * head_dim] or [head_num, head_dim].
        initial_state: [request_num, head_num, head_dim, value_head_dim].
        cu_seqlens: [request_num + 1]; chunk_indices: [chunk_num, 2].

    Per-token definition (one head; q/k/v/delta are column vectors; S: [head_dim, value_head_dim]):
        ell_t is the activated log decay from the gate kernel.
        S_decay = diag(exp(ell_t)) @ S_prev
        delta_t = beta_t * (v_t - S_decay.T @ k_t)  # Compute the value correction.
        S_t = S_decay + outer(k_t, delta_t)        # Write the correction into the state.
        o_t = scale * (S_t.T @ q_t)               # Read the updated state with q_t.

        beta is supplied after sigmoid. If use_qk_l2norm_in_kernel=True, normalize q/k first
        with x / sqrt(sum(x * x) + 1e-6). The default scale is head_dim ** -0.5.

    Chunk equations (one head, chunk_size=64; Q/K/V hold token vectors in rows):
        Q/K/P: [chunk_size, head_dim]; V/E/O: [chunk_size, value_head_dim].
        Akk/Aqk: [chunk_size, chunk_size]; S_in: [head_dim, value_head_dim].
        S_in is the state before the chunk; P=exp2(G); E_i=delta_i.T.

        Expanding the decayed state before token i's write gives:
            S_decay_i = diag(P_i) @ S_in
                      + sum_{j<i} (k_j * exp2(G_i - G_j)) @ E_j
        Substitute this into E_i = beta_i * (v_i.T - k_i.T @ S_decay_i):
            E_i = beta_i * v_i.T - beta_i * (k_i * P_i).T @ S_in
                  - sum_{j<i} Akk_ij * E_j
            Akk_ij = beta_i * sum_d(k_i[d] * k_j[d] * exp2(G_i[d] - G_j[d])), j < i.
        Akk is zero elsewhere. The subtractions remove predictions from S_in and earlier writes.
        This recurrence can be evaluated directly in token order: E_0, E_1, ...; it does not require R.

        Reading the updated state with q_i gives the output weights:
            Aqk_ij = scale * sum_d(q_i[d] * k_j[d] * exp2(G_i[d] - G_j[d])), j <= i.
        Aqk is zero elsewhere. It includes the current token's write, unlike Akk.
            O = scale * (Q * P) @ S_in + Aqk @ E

    Matrix solution of the E recurrence:
        Move the sum to the left and stack token rows:
            (I + Akk) @ E = diag(beta) @ V - (diag(beta) @ (K * P)) @ S_in
        Akk is strictly lower triangular, so I + Akk is invertible. Define R: [chunk_size, chunk_size]:
            R = (I + Akk)^-1
            E = R @ diag(beta) @ V - (R @ diag(beta) @ (K * P)) @ S_in
        Define U = R @ diag(beta) @ V and W = R @ diag(beta) @ (K * P), giving:
            E = U - W @ S_in
        This solves the same recurrence with matrix multiplication instead of token-by-token substitution.
        R/U/W depend only on chunk inputs and can be prepared before S_in is available.

    Kernel execution:
        1. fused_kda_gate_chunk_cumsum: activate raw_g and compute G within each chunk.
        2. chunk_kda_scaled_dot_kkt_fwd: build Akk and Aqk.
        3. solve_tril: compute R = (I + Akk)^-1.
        4. recompute_w_u_fwd: compute U/W above and Kg = K * exp2(G_last - G).
        5. chunk_gated_delta_rule_fwd_h: save S_in in h and compute v_new = E = U - W @ S_in.
               S_out = P_last[:, None] * S_in + Kg.T @ E
           Within each request, process chunks in order and pass S_out to the next chunk.
           Start from initial_state, or zeros if None. Different requests, heads, and
           blocks of value channels can run in parallel.
        6. chunk_gla_fwd_o_gk: compute O from h and E using the output equation above.
           Outputs overwrite the contiguous v buffer passed to the output kernel.

        Stages 1-4 and 6 process chunks in parallel; only stage 5 passes state between chunks.

    Returns:
        Output: [1, total_tokens, head_num, value_head_dim], v.dtype.
        Final state: [request_num, head_num, head_dim, value_head_dim], float32.
        The final state is None when output_final_state=False.
    """
    assert q.ndim == 4 and q.shape[0] == 1, "KDA prefill expects packed q shaped [1, total_tokens, head_num, head_dim]"
    assert cu_seqlens is not None, "cu_seqlens is required for packed KDA prefill"
    if scale is None:
        head_dim = k.shape[-1]
        scale = head_dim ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    o, final_state = chunk_kda_with_fused_gate_fwd(
        q=q,
        k=k,
        v=v.contiguous(),
        raw_g=raw_g.contiguous(),
        beta=beta.contiguous(),
        A_log=A_log,
        g_bias=g_bias,
        scale=scale,
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )
    return o, final_state
