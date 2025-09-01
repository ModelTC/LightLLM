import torch

import triton
import triton.language as tl
import itertools
from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _rotary_kernel(
    Q,
    K,
    Cos,
    Sin,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_cosbs,
    stride_cosd,
    stride_sinbs,
    stride_sind,
    max_total_len,
    HEAD_PARALLEL_NUM: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    head_start_index = tl.program_id(0)
    seq_block_index = tl.program_id(1)
    seq_start_index = seq_block_index * BLOCK_SEQ
    seq_end_index = (seq_block_index + 1) * BLOCK_SEQ
    seq_end_index = tl.where(seq_end_index < max_total_len, seq_end_index, max_total_len)

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2) * 2
    dim_range1 = dim_range0 + 1
    cos_range = tl.arange(0, BLOCK_DMODEL // 2)
    for seq_index in tl.range(seq_start_index, seq_end_index):

        off_dimcos_sin = seq_index * stride_cosbs + cos_range * stride_cosd
        cos = tl.load(Cos + off_dimcos_sin)
        sin = tl.load(Sin + off_dimcos_sin)

        if HEAD_PARALLEL_NUM == 1:
            for q_head_index in tl.static_range(0, HEAD_Q, step=1):
                off_q0 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range0 * stride_qd
                off_q1 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range1 * stride_qd
                q0 = tl.load(Q + off_q0)
                q1 = tl.load(Q + off_q1)
                out_q0 = q0 * cos - q1 * sin
                out_q1 = q0 * sin + q1 * cos
                tl.store(Q + off_q0, out_q0)
                tl.store(Q + off_q1, out_q1)

            for k_head_index in tl.static_range(0, HEAD_K, step=1):
                off_k0 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range0 * stride_kd
                off_k1 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range1 * stride_kd

                k0 = tl.load(K + off_k0)
                k1 = tl.load(K + off_k1)

                out_k0 = k0 * cos - k1 * sin
                out_k1 = k0 * sin + k1 * cos

                tl.store(K + off_k0, out_k0)
                tl.store(K + off_k1, out_k1)
        else:
            for q_head_index in tl.range(head_start_index, HEAD_Q, step=HEAD_PARALLEL_NUM, num_stages=NUM_STAGE):
                off_q0 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range0 * stride_qd
                off_q1 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range1 * stride_qd
                q0 = tl.load(Q + off_q0)
                q1 = tl.load(Q + off_q1)
                out_q0 = q0 * cos - q1 * sin
                out_q1 = q0 * sin + q1 * cos
                tl.store(Q + off_q0, out_q0)
                tl.store(Q + off_q1, out_q1)

            for k_head_index in tl.range(head_start_index, HEAD_K, step=HEAD_PARALLEL_NUM, num_stages=NUM_STAGE):
                off_k0 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range0 * stride_kd
                off_k1 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range1 * stride_kd

                k0 = tl.load(K + off_k0)
                k1 = tl.load(K + off_k1)

                out_k0 = k0 * cos - k1 * sin
                out_k1 = k0 * sin + k1 * cos

                tl.store(K + off_k0, out_k0)
                tl.store(K + off_k1, out_k1)
    return


def get_test_configs():
    configs = []
    result = itertools.product([1, 2, 4, 8, 16, 32], [1, 2, 4, 8], [1, 2, 3, 4, 5], [1, 2, 4, 8, 16])
    for BLOCK_SEQ, num_warps, num_stages, HEAD_PARALLEL_NUM in result:
        t_config = {
            "BLOCK_SEQ": BLOCK_SEQ,
            "HEAD_PARALLEL_NUM": HEAD_PARALLEL_NUM,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        configs.append(t_config)
    return configs


def get_static_key(q, k):
    head_num_q, head_num_k, head_dim = q.shape[1], k.shape[1], q.shape[2]
    return {
        "Q_HEAD_NUM": head_num_q,
        "K_HEAD_NUM": head_num_k,
        "HEAD_DIM": head_dim,
        "dtype": str(q.dtype),
    }


@autotune(
    kernel_name="rotary_emb_fwd:v1",
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=lambda q: q.shape[0],
    mutates_args=["q", "k"],
)
@torch.no_grad()
def rotary_emb_fwd(q, k, cos, sin, run_config=None):
    total_len = q.shape[0]
    head_num_q, head_num_k = q.shape[1], k.shape[1]
    head_dim = q.shape[2]
    assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
    assert k.shape[0] == cos.shape[0] and k.shape[0] == sin.shape[0], f"k shape {k.shape} cos shape {cos.shape}"
    assert triton.next_power_of_2(head_dim) == head_dim

    from .rotary_emb_config import DeepseekV3RotaryKernelConfig

    if not run_config:
        run_config = DeepseekV3RotaryKernelConfig.try_to_get_best_config(
            M=total_len,
            Q_HEAD_NUM=head_num_q,
            K_HEAD_NUM=head_num_k,
            HEAD_DIM=head_dim,
            dtype=str(q.dtype),
        )

    BLOCK_SEQ = run_config["BLOCK_SEQ"]
    HEAD_PARALLEL_NUM = run_config["HEAD_PARALLEL_NUM"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    grid = (
        HEAD_PARALLEL_NUM,
        triton.cdiv(total_len, BLOCK_SEQ),
    )
    _rotary_kernel[grid](
        Q=q,
        K=k,
        Cos=cos,
        Sin=sin,
        stride_qbs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_cosbs=cos.stride(0),
        stride_cosd=cos.stride(1),
        stride_sinbs=sin.stride(0),
        stride_sind=sin.stride(1),
        max_total_len=total_len,
        HEAD_Q=head_num_q,
        HEAD_PARALLEL_NUM=HEAD_PARALLEL_NUM,
        HEAD_K=head_num_k,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        NUM_STAGE=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return
