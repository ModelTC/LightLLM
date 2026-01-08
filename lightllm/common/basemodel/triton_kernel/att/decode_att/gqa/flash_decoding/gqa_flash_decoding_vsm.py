import torch
import triton
import triton.language as tl
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.device_utils import calcu_kernel_best_vsm_count
from frozendict import frozendict
from functools import lru_cache
from typing import Dict


class GQAVSMDecodeAttentionKernelConfig(KernelConfigs):
    kernel_name: str = "gqa_decode_attentnion_vsm"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        batch_size: int,
        avg_seq_len_in_batch: int,
        q_head_num: int,
        q_head_dim: int,
        kv_head_num: int,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "q_head_num": q_head_num,
            "q_head_dim": q_head_dim,
            "kv_head_num": kv_head_num,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            batch_size_config: dict = finded_config[
                min(
                    finded_config.keys(),
                    key=lambda x: abs(int(x) - avg_seq_len_in_batch),
                )
            ]
            config = batch_size_config[min(batch_size_config.keys(), key=lambda x: abs(int(x) - batch_size))]

            return config
        else:
            config = {
                "BLOCK_N": 64,
                "BLOCK_Q_HEAD": 16,
                "stage1_num_warps": 4,
                "stage1_num_stages": 2,
                "stage2_num_warps": 4,
                "stage2_num_stages": 1,
            }
        return config

    @classmethod
    def save_config(
        cls,
        q_head_num: int,
        q_head_dim: int,
        kv_head_num: int,
        out_dtype: str,
        config_json: Dict[int, Dict[int, Dict]],
    ):
        key_params = {
            "q_head_num": q_head_num,
            "q_head_dim": q_head_dim,
            "kv_head_num": kv_head_num,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)


@triton.jit
def _fwd_kernel_calcu_index_and_block_seq(
    b_seq_len,
    mid_o_decode_att_block_seq,
    mid_o_batch_start_index,
    vsm_count,
    batch_size,
    BLOCK_N: tl.constexpr,
):
    b_seq_len = tl.load(b_seq_len + tl.arange(0, 2048), mask=tl.arange(0, 2048) < batch_size, other=0)
    total_token_num = tl.sum(b_seq_len)

    block_seq = tl.cdiv(total_token_num, vsm_count * 4)
    block_seq = tl.cast(block_seq, tl.int64)
    block_seq = tl.cdiv(block_seq, BLOCK_N) * BLOCK_N

    block_seq_len = tl.cdiv(b_seq_len, block_seq)
    cumsum_seq_len = tl.cumsum(block_seq_len)
    batch_start_index = cumsum_seq_len - block_seq_len
    tl.store(
        mid_o_batch_start_index + tl.arange(0, 2048),
        batch_start_index,
        mask=tl.arange(0, 2048) < batch_size,
    )
    tl.store(mid_o_decode_att_block_seq, block_seq)


@triton.jit
def _kernel_gqa_token_decode_attention_flash_decoding_vsm_stage1(
    block_size,
    q,
    k,
    v,
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    mid_o,
    mid_o_logexpsum,
    softmax_scale,
    num_sm,
    gqa_group_size,
    q_head_num,
    kv_head_num,
    batch_size,
    stride_q_bs,
    stride_q_h,
    stride_q_d,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_v_bs,
    stride_v_h,
    stride_v_d,
    stride_req_to_token_bs,
    stride_req_to_token_seq,
    stride_mid_o_h,
    stride_mid_o_seq,
    stride_mid_o_d,
    stride_mid_o_logexpsum_h,
    stride_mid_o_logexpsum_seq,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    r"""
    shape:
        q: [batch_size, q_head_num, q_head_dim]
        k: [total_token_num, kv_head_num, kv_head_dim]
        v: [total_token_num, kv_head_num, kv_head_dim]
        req_to_token_indexs: [batch_size, max_seq_len]
        b_req_idx: [batch_size]
        b_seq_len: [batch_size]
        mid_o: [q_head_num, total_seq_block_num, q_head_dim]
        mid_o_logexpsum: [q_head_num, total_seq_block_num]
    """
    sm_id = tl.program_id(0).to(tl.int64)
    block_size = tl.load(block_size)

    out_batch_start_index = tl.cast(0, tl.int64)
    q_head_off = tl.arange(0, Q_HEAD_NUM)
    d_off = tl.arange(0, BLOCK_DMODEL)

    for cur_batch in range(0, batch_size):
        cur_req_idx = tl.load(b_req_idx + cur_batch)
        cur_seq_len = tl.load(b_seq_len + cur_batch)

        cur_num_of_blocks = tl.cdiv(cur_seq_len, block_size)
        cur_num_of_kv_head_pairs = cur_num_of_blocks * kv_head_num

        # loop_sm_id = sm_id
        while sm_id < cur_num_of_kv_head_pairs:
            cur_block_idx = sm_id % cur_num_of_blocks
            cur_kv_head_idx = sm_id // cur_num_of_blocks
            # cur_block_idx = sm_id // kv_head_num
            # cur_kv_head_idx = sm_id % kv_head_num

            cur_q_range = cur_kv_head_idx * gqa_group_size + q_head_off
            cur_q_mask = q_head_off < gqa_group_size

            cur_kv_start = cur_block_idx * block_size

            q_off = cur_batch * stride_q_bs + cur_q_range[:, None] * stride_q_h + d_off[None, :]
            q_tensor = tl.load(
                q + q_off,
                mask=cur_q_mask[:, None],
                other=0.0,
            )  # shape: [Q_HEAD_NUM, BLOCK_DMODEL]

            sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
            max_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
            accumu = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)

            cur_total_chunk = tl.cdiv(tl.minimum(cur_kv_start + block_size, cur_seq_len) - cur_kv_start, BLOCK_N)

            for chunk_idx in tl.range(0, cur_total_chunk, 1, num_stages=NUM_STAGES):
                cur_chunk_start = cur_kv_start + chunk_idx * BLOCK_N
                cur_chunk_range = cur_chunk_start + tl.arange(0, BLOCK_N)
                cur_chunk_mask = cur_chunk_range < cur_seq_len
                cur_kv_loc = tl.load(
                    req_to_token_indexs
                    + cur_req_idx * stride_req_to_token_bs
                    + cur_chunk_range * stride_req_to_token_seq,
                    mask=cur_chunk_mask,
                    other=0.0,
                ).to(tl.int64)

                k_off = (
                    cur_kv_loc[None, :] * stride_k_bs + cur_kv_head_idx * stride_k_h + d_off[:, None]
                )  # shape: [BLOCK_DMODEL, BLOCK_N]
                v_off = cur_kv_loc[:, None] * stride_v_bs + cur_kv_head_idx * stride_v_h + d_off[None, :]
                k_tensor = tl.load(k + k_off, mask=cur_chunk_mask[None, :], other=0.0)

                att_tensor = tl.dot(q_tensor, k_tensor)  # shape: [Q_HEAD_NUM, BLOCK_N]
                att_tensor *= softmax_scale
                att_tensor = tl.where(cur_chunk_mask[None, :], att_tensor, float("-inf"))

                cur_max = tl.max(att_tensor, axis=1)
                new_max = tl.maximum(cur_max, max_exp)

                exp_logic = tl.exp(att_tensor - new_max[:, None])
                log_scale = tl.exp(max_exp - new_max)
                accumu *= log_scale[:, None]
                v_tensor = tl.load(v + v_off, mask=cur_chunk_mask[:, None], other=0.0)  # shape: [BLOCK_N, BLOCK_DMODEL]
                accumu += tl.dot(exp_logic.to(v_tensor.dtype), v_tensor)

                sum_exp = sum_exp * log_scale + tl.sum(exp_logic, axis=1)
                max_exp = new_max
            off_mid_o = (
                cur_q_range[:, None] * stride_mid_o_h
                + (out_batch_start_index + cur_block_idx) * stride_mid_o_seq
                + d_off[None, :]
            )
            tl.store(mid_o + off_mid_o, accumu, mask=cur_q_mask[:, None])
            off_mid_o_logexpsum = (
                cur_q_range * stride_mid_o_logexpsum_h
                + (out_batch_start_index + cur_block_idx) * stride_mid_o_logexpsum_seq
            )
            max_exp = max_exp + tl.log(sum_exp)
            tl.store(
                mid_o_logexpsum + off_mid_o_logexpsum,
                max_exp,
                mask=cur_q_mask,
            )
            sm_id += num_sm
        sm_id -= cur_num_of_kv_head_pairs
        out_batch_start_index += cur_num_of_blocks


def gqa_token_decode_attention_flash_decoding_vsm_stage1(
    block_size,
    q,
    k,
    v,
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    mid_o,
    mid_o_logexpsum,
    softmax_scale,
    num_vsm,
    gqa_group_size,
    q_head_num,
    kv_head_num,
    batch_size,
    **run_config
):
    grid = (num_vsm,)

    assert num_vsm * 4 + batch_size <= mid_o.shape[1]
    _kernel_gqa_token_decode_attention_flash_decoding_vsm_stage1[grid](
        block_size,
        q,
        k,
        v,
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        mid_o,
        mid_o_logexpsum,
        softmax_scale,
        num_vsm,
        gqa_group_size,
        q_head_num,
        kv_head_num,
        batch_size,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *req_to_token_indexs.stride(),
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        BLOCK_N=run_config["BLOCK_N"],
        Q_HEAD_NUM=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_DMODEL=q.shape[-1],
        NUM_STAGES=run_config["stage1_num_stages"],
        num_stages=run_config["stage1_num_stages"],
        num_warps=run_config["stage1_num_warps"],
    )


@triton.jit
def _kernel_gqa_token_decode_attention_flash_decoding_vsm_stage2(
    mid_o_decode_att_block_seq,
    mid_o_batch_start_index,
    mid_o,
    mid_o_logexpsum,
    b_seq_len,
    out,
    stride_mid_o_h,
    stride_mid_o_seq,
    stride_mid_o_d,
    stride_mid_o_logexpsum_h,
    stride_mid_o_logexpsum_seq,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    BLOCK_DMODEL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    cur_head = tl.program_id(0)
    cur_batch = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(b_seq_len + cur_batch)
    cur_batch_start_index = tl.load(mid_o_batch_start_index + cur_batch)
    block_size = tl.load(mid_o_decode_att_block_seq)
    block_n_size = tl.cdiv(cur_batch_seq_len, block_size)
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    off_mo = cur_head * stride_mid_o_h + cur_batch_start_index * stride_mid_o_seq + off_d
    off_ml = cur_head * stride_mid_o_logexpsum_h + cur_batch_start_index * stride_mid_o_logexpsum_seq

    for block_seq_n in tl.range(0, block_n_size, 1, num_stages=NUM_STAGES):
        mo_tensor = tl.load(mid_o + off_mo + block_seq_n * stride_mid_o_seq)
        ml_tensor = tl.load(mid_o_logexpsum + off_ml + block_seq_n)
        new_max_logic = tl.maximum(ml_tensor, max_logic)

        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(ml_tensor - new_max_logic)
        acc += exp_logic * mo_tensor
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(out + cur_batch * stride_o_bs + cur_head * stride_o_h + off_d, acc / sum_exp)


def gqa_token_decode_attention_flash_decoding_vsm_stage2(
    mid_o_decode_att_block_seq, mid_o_batch_start_index, mid_o, mid_o_logexpsum, b_seq_len, out, **run_config
):
    num_warps = run_config["stage2_num_warps"]
    num_stages = run_config["stage2_num_stages"]

    batch, q_head_num = mid_o_batch_start_index.shape[0], mid_o.shape[0]
    grid = (q_head_num, batch)

    _kernel_gqa_token_decode_attention_flash_decoding_vsm_stage2[grid](
        mid_o_decode_att_block_seq,
        mid_o_batch_start_index,
        mid_o,
        mid_o_logexpsum,
        b_seq_len,
        out,
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        *out.stride(),
        BLOCK_DMODEL=mid_o.shape[-1],
        NUM_STAGES=run_config["stage2_num_stages"],
        num_warps=num_warps,
        num_stages=num_stages,
    )


def emstimate_stage1_vsm(
    q, k, v, req_to_token_indexs, b_req_idx, b_seq_len, mid_o, mid_o_logexpsum, softmax_scale, **run_config
):
    num_sm = 1
    q_head_num = q.shape[1]
    kv_head_num = k.shape[1]
    gqa_group_size = triton.cdiv(q_head_num, kv_head_num)
    q_head_num = q_head_num
    batch_size = b_req_idx.shape[0]
    kernel = _kernel_gqa_token_decode_attention_flash_decoding_vsm_stage1.warmup(
        torch.empty([1], dtype=torch.int64, device="cuda"),
        q,
        k,
        v,
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        mid_o,
        mid_o_logexpsum,
        softmax_scale,
        num_sm,
        gqa_group_size,
        q_head_num,
        kv_head_num,
        batch_size,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *req_to_token_indexs.stride(),
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        Q_HEAD_NUM=max(run_config["BLOCK_Q_HEAD"], triton.next_power_of_2(q_head_num)),
        BLOCK_DMODEL=q.shape[-1],
        BLOCK_N=run_config["BLOCK_N"],
        NUM_STAGES=run_config["stage1_num_stages"],
        grid=(1,),
    )
    kernel._init_handles()
    num_vsm = calcu_kernel_best_vsm_count(kernel, num_warps=run_config["stage1_num_warps"])
    return num_vsm


def gqa_token_decode_attention_flash_decoding_vsm(
    q, k, v, infer_state, out=None, alloc_tensor_func=torch.empty, **run_config
):
    batch_size, q_head_num, q_head_dim = q.shape
    kv_head_num = k.shape[1]
    gqa_group_size = q_head_num // kv_head_num
    sm_scale = 1.0 / (q_head_dim ** 0.5)

    if not run_config:
        if torch.cuda.is_current_stream_capturing():
            avg_seq_len_in_batch = infer_state.max_kv_seq_len
        else:
            avg_seq_len_in_batch = infer_state.total_token_num // batch_size

        run_config = GQAVSMDecodeAttentionKernelConfig.try_to_get_best_config(
            batch_size=batch_size,
            avg_seq_len_in_batch=avg_seq_len_in_batch,
            q_head_num=q_head_num,
            q_head_dim=q_head_dim,
            kv_head_num=kv_head_num,
            out_dtype=q.dtype,
        )

    if out is None:
        out = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device)

    num_vsm = emstimate_stage1_vsm(
        q,
        k,
        v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        torch.empty(
            [q_head_num, 0, q_head_dim],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.empty([q_head_num, 0], dtype=torch.float32, device="cuda"),
        sm_scale,
        **run_config,
    )

    if not hasattr(infer_state, "decode_att_block_seq"):
        assert batch_size <= 2048
        decode_att_block_seq = torch.empty(
            [
                1,
            ],
            dtype=torch.int64,
            device="cuda",
        )
        mid_o_batch_start_index = torch.empty(
            [
                batch_size,
            ],
            dtype=torch.int64,
            device="cuda",
        )
        _fwd_kernel_calcu_index_and_block_seq[(1,)](
            infer_state.b_seq_len,
            decode_att_block_seq,
            mid_o_batch_start_index,
            num_vsm,
            batch_size,
            BLOCK_N=run_config["BLOCK_N"],
            num_warps=4,
        )

        infer_state.decode_att_block_seq = decode_att_block_seq
        infer_state.mid_o_batch_start_index = mid_o_batch_start_index

    mid_o = torch.empty(
        [
            q_head_num,
            num_vsm * 4 + batch_size,
            q_head_dim,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    mid_o_logexpsum = torch.empty(
        [q_head_num, num_vsm * 4 + batch_size],
        dtype=torch.float32,
        device="cuda",
    )
    gqa_token_decode_attention_flash_decoding_vsm_stage1(
        infer_state.decode_att_block_seq,
        q,
        k,
        v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        mid_o,
        mid_o_logexpsum,
        sm_scale,
        num_vsm,
        gqa_group_size,
        q_head_num,
        kv_head_num,
        batch_size,
        **run_config,
    )

    gqa_token_decode_attention_flash_decoding_vsm_stage2(
        infer_state.decode_att_block_seq,
        infer_state.mid_o_batch_start_index,
        mid_o,
        mid_o_logexpsum,
        infer_state.b_seq_len,
        out,
        **run_config,
    )
    return out
