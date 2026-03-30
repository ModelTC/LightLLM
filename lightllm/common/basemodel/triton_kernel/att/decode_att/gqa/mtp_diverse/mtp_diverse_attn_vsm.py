"""
MTP Diverse Attention - VSM Version (Optimized with Group Sharing)

优化版本：真正利用 b_mark_shared_group 减少计算量

核心优化：
1. 只有组内最后一个请求（b_mark_shared_group != 0）触发计算
2. 动态计算 block_seq 和 batch_start_index，只考虑需要计算的 batch
3. stage2 跳过 b_mark_shared_group=0 的 batch，避免无效计算

注意：当前版本未使用 BLOCK_BATCH 同时处理组内多个 Q，
      但已经通过跳过 b_mark_shared_group=0 的 batch 减少了计算量。
"""
import torch
import triton
import triton.language as tl
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.device_utils import calcu_kernel_best_vsm_count
from frozendict import frozendict
from functools import lru_cache
from typing import Dict


class MTPDiverseVSMKernelConfig(KernelConfigs):
    kernel_name: str = "mtp_diverse_vsm"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls, batch_size: int, max_kv_len: int, q_head_num: int, q_head_dim: int,
        kv_head_num: int, out_dtype: str
    ) -> dict:
        key_params = frozendict({
            "q_head_num": q_head_num, "q_head_dim": q_head_dim,
            "kv_head_num": kv_head_num, "out_dtype": str(out_dtype),
        })
        finded_config = cls.get_the_config(key_params)
        if finded_config:
            batch_size_config = finded_config[min(finded_config.keys(), key=lambda x: abs(int(x) - max_kv_len))]
            return batch_size_config[min(batch_size_config.keys(), key=lambda x: abs(int(x) - batch_size))]
        return {"BLOCK_N": 64, "BLOCK_HEAD": 16, "BLOCK_BATCH": 16, "stage1_num_warps": 4, "stage1_num_stages": 2, "stage2_num_warps": 4, "stage2_num_stages": 2}

    @classmethod
    def save_config(cls, q_head_num: int, q_head_dim: int, kv_head_num: int, out_dtype: str, config_json: Dict):
        return cls.store_config(frozendict({"q_head_num": q_head_num, "q_head_dim": q_head_dim, "kv_head_num": kv_head_num, "out_dtype": str(out_dtype)}), config_json)


@triton.jit
def _fwd_kernel_calcu_index_and_block_seq(
    b_seq_len, b_mark_shared_group, mid_o_decode_att_block_seq, mid_o_batch_start_index,
    vsm_count, batch_size, BLOCK_N: tl.constexpr, MAX_BATCH_SIZE: tl.constexpr,
):
    """
    计算动态 block_seq 和 batch_start_index
    注意：block_seq 基于所有需要计算的 token 总数（只考虑 b_mark_shared_group > 0 的 batch）
    batch_start_index 也只对 b_mark_shared_group > 0 的 batch 有意义
    """
    b_seq_len_arr = tl.load(b_seq_len + tl.arange(0, MAX_BATCH_SIZE), mask=tl.arange(0, MAX_BATCH_SIZE) < batch_size, other=0)
    b_mark_arr = tl.load(b_mark_shared_group + tl.arange(0, MAX_BATCH_SIZE), mask=tl.arange(0, MAX_BATCH_SIZE) < batch_size, other=0)

    # 计算 block_seq：只考虑需要计算的 batch
    need_compute = b_mark_arr > 0
    total_compute_token_num = tl.sum(b_seq_len_arr * need_compute)
    block_seq = tl.cdiv(total_compute_token_num, vsm_count * 4)
    block_seq = tl.cast(block_seq, tl.int64)
    block_seq = tl.cdiv(block_seq, BLOCK_N) * BLOCK_N

    # 只为 need_compute的batch 计算block_start_index
    # 每个 batch 的 block 数量 = cdiv(seq_len, block_seq)
    block_seq_len = tl.cdiv(b_seq_len_arr, block_seq)

    # 只有 need_compute 的 batch 才计入 cumsum
    block_seq_len_for_cumsum = block_seq_len * need_compute
    cumsum_seq_len = tl.cumsum(block_seq_len_for_cumsum)
    batch_start_index = cumsum_seq_len - block_seq_len

    tl.store(mid_o_batch_start_index + tl.arange(0, MAX_BATCH_SIZE), batch_start_index, mask=tl.arange(0, MAX_BATCH_SIZE) < batch_size)
    tl.store(mid_o_decode_att_block_seq, block_seq)


@triton.jit
def _kernel_mtp_diverse_vsm_stage1(
    block_size, q, k, v, req_to_token_indexs, b_req_idx, b_seq_len, b_mark_shared_group,
    mid_o, mid_o_logexpsum, softmax_scale, num_sm, gqa_group_size, q_head_num, kv_head_num, batch_size,
    stride_q_bs, stride_q_h, stride_q_d, stride_k_bs, stride_k_h, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_d, stride_req_to_token_bs, stride_req_to_token_seq,
    stride_mid_o_h, stride_mid_o_seq, stride_mid_o_d, stride_mid_o_logexpsum_h, stride_mid_o_logexpsum_seq,
    Q_HEAD_NUM: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, NUM_STAGES: tl.constexpr, BLOCK_BATCH: tl.constexpr,
):
    """
    MTP Diverse VSM Stage1 - 只有组内最后一个请求计算
    简化版本：每个 batch 独立计算，逻辑与普通版本一致
    """
    sm_id = tl.program_id(0).to(tl.int64)
    block_size_val = tl.load(block_size)
    out_batch_start_index = tl.cast(0, tl.int64)
    q_head_off = tl.arange(0, Q_HEAD_NUM)
    d_off = tl.arange(0, BLOCK_DMODEL)

    for cur_batch in range(0, batch_size):
        cur_req_idx = tl.load(b_req_idx + cur_batch)
        cur_seq_len = tl.load(b_seq_len + cur_batch)
        cur_mark = tl.load(b_mark_shared_group + cur_batch)
        cur_num_of_blocks = tl.cdiv(cur_seq_len, block_size_val)

        # 只有组内最后一个请求才计算
        if cur_mark > 0:
            cur_num_of_kv_head_pairs = cur_num_of_blocks * kv_head_num
            while sm_id < cur_num_of_kv_head_pairs:
                cur_block_idx = sm_id % cur_num_of_blocks
                cur_kv_head_idx = sm_id // cur_num_of_blocks
                cur_q_range = cur_kv_head_idx * gqa_group_size + q_head_off
                cur_q_mask = q_head_off < gqa_group_size
                cur_kv_start = cur_block_idx * block_size_val

                # 加载 Q: [Q_HEAD_NUM, BLOCK_DMODEL]
                q_off = cur_batch * stride_q_bs + cur_q_range[:, None] * stride_q_h + d_off[None, :]
                q_tensor = tl.load(q + q_off, mask=cur_q_mask[:, None], other=0.0)

                sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
                max_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
                accumu = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)

                cur_total_chunk = tl.cdiv(tl.minimum(cur_kv_start + block_size_val, cur_seq_len) - cur_kv_start, BLOCK_N)

                for chunk_idx in tl.range(0, cur_total_chunk, 1, num_stages=NUM_STAGES):
                    cur_chunk_start = cur_kv_start + chunk_idx * BLOCK_N
                    cur_chunk_range = cur_chunk_start + tl.arange(0, BLOCK_N)
                    cur_chunk_mask = cur_chunk_range < cur_seq_len

                    cur_kv_loc = tl.load(
                        req_to_token_indexs + cur_req_idx * stride_req_to_token_bs + cur_chunk_range * stride_req_to_token_seq,
                        mask=cur_chunk_mask, other=0.0).to(tl.int64)

                    k_off = cur_kv_loc[None, :] * stride_k_bs + cur_kv_head_idx * stride_k_h + d_off[:, None]
                    v_off = cur_kv_loc[:, None] * stride_v_bs + cur_kv_head_idx * stride_v_h + d_off[None, :]
                    k_tensor = tl.load(k + k_off, mask=cur_chunk_mask[None, :], other=0.0)
                    v_tensor = tl.load(v + v_off, mask=cur_chunk_mask[:, None], other=0.0)

                    att_tensor = tl.dot(q_tensor, k_tensor)
                    att_tensor *= softmax_scale
                    att_tensor = tl.where(cur_chunk_mask[None, :], att_tensor, float("-inf"))

                    cur_max = tl.max(att_tensor, axis=1)
                    new_max = tl.maximum(cur_max, max_exp)
                    exp_logic = tl.exp(att_tensor - new_max[:, None])
                    log_scale = tl.exp(max_exp - new_max)
                    accumu = accumu * log_scale[:, None] + tl.dot(exp_logic.to(v_tensor.dtype), v_tensor)
                    sum_exp = sum_exp * log_scale + tl.sum(exp_logic, axis=1)
                    max_exp = new_max

                # Store 只存储有效的 head
                off_mid_o = cur_q_range[:, None] * stride_mid_o_h + (out_batch_start_index + cur_block_idx) * stride_mid_o_seq + d_off[None, :]
                mid_o_val = accumu / sum_exp[:, None]
                tl.store(mid_o + off_mid_o, mid_o_val, mask=cur_q_mask[:, None])
                off_mid_o_logexpsum = cur_q_range * stride_mid_o_logexpsum_h + (out_batch_start_index + cur_block_idx) * stride_mid_o_logexpsum_seq
                tl.store(mid_o_logexpsum + off_mid_o_logexpsum, max_exp + tl.log(sum_exp), mask=cur_q_mask)
                sm_id += num_sm
            sm_id -= cur_num_of_kv_head_pairs
            # 只有 compute batch 才增加 out_batch_start_index
            out_batch_start_index += cur_num_of_blocks


@triton.jit
def _kernel_mtp_diverse_vsm_stage2(
    mid_o_decode_att_block_seq, mid_o_batch_start_index, mid_o, mid_o_logexpsum, b_seq_len, b_mark_shared_group, out,
    stride_mid_o_h, stride_mid_o_seq, stride_mid_o_d, stride_mid_o_logexpsum_h, stride_mid_o_logexpsum_seq,
    stride_o_bs, stride_o_h, stride_o_d, BLOCK_DMODEL: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    cur_head = tl.program_id(0)
    cur_batch = tl.program_id(1)

    # Only compute output for the last request in each group (b_mark_shared_group > 0)
    cur_mark = tl.load(b_mark_shared_group + cur_batch)
    if cur_mark == 0:
        return

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


def mtp_diverse_vsm_stage1(block_size, q, k, v, req_to_token_indexs, b_req_idx, b_seq_len, b_mark_shared_group,
                           mid_o, mid_o_logexpsum, softmax_scale, num_vsm, gqa_group_size, q_head_num, kv_head_num, batch_size, **run_config):
    _kernel_mtp_diverse_vsm_stage1[(num_vsm,)](
        block_size, q, k, v, req_to_token_indexs, b_req_idx, b_seq_len, b_mark_shared_group,
        mid_o, mid_o_logexpsum, softmax_scale, num_vsm, gqa_group_size, q_head_num, kv_head_num, batch_size,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o_logexpsum.stride(0), mid_o_logexpsum.stride(1),
        BLOCK_N=run_config["BLOCK_N"], Q_HEAD_NUM=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_DMODEL=q.shape[-1], NUM_STAGES=run_config["stage1_num_stages"],
        num_stages=run_config["stage1_num_stages"], num_warps=run_config["stage1_num_warps"],
        BLOCK_BATCH=run_config.get("BLOCK_BATCH", 16),
    )


def mtp_diverse_vsm_stage2(mid_o_decode_att_block_seq, mid_o_batch_start_index, mid_o, mid_o_logexpsum, b_seq_len, b_mark_shared_group, out, **run_config):
    batch, q_head_num = mid_o_batch_start_index.shape[0], mid_o.shape[0]
    _kernel_mtp_diverse_vsm_stage2[(q_head_num, batch)](
        mid_o_decode_att_block_seq, mid_o_batch_start_index, mid_o, mid_o_logexpsum, b_seq_len, b_mark_shared_group, out,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o_logexpsum.stride(0), mid_o_logexpsum.stride(1),
        out.stride(0), out.stride(1), out.stride(2), BLOCK_DMODEL=mid_o.shape[-1],
        NUM_STAGES=run_config["stage2_num_stages"], num_warps=run_config["stage2_num_warps"], num_stages=run_config["stage2_num_stages"],
    )


def estimate_stage1_vsm(q, k, v, req_to_token_indexs, b_req_idx, b_seq_len, b_mark_shared_group, mid_o, mid_o_logexpsum, softmax_scale, **run_config):
    num_sm = 1
    q_head_num, kv_head_num = q.shape[1], k.shape[1]
    gqa_group_size = triton.cdiv(q_head_num, kv_head_num)
    batch_size = b_req_idx.shape[0]
    kernel = _kernel_mtp_diverse_vsm_stage1.warmup(
        torch.empty([1], dtype=torch.int64, device="cuda"), q, k, v, req_to_token_indexs, b_req_idx, b_seq_len, b_mark_shared_group,
        mid_o, mid_o_logexpsum, softmax_scale, num_sm, gqa_group_size, q_head_num, kv_head_num, batch_size,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o_logexpsum.stride(0), mid_o_logexpsum.stride(1),
        Q_HEAD_NUM=max(run_config.get("BLOCK_HEAD", 16), triton.next_power_of_2(q_head_num)),
        BLOCK_DMODEL=q.shape[-1], BLOCK_N=run_config["BLOCK_N"], NUM_STAGES=run_config["stage1_num_stages"],
        BLOCK_BATCH=run_config.get("BLOCK_BATCH", 16), grid=(1,),
    )
    kernel._init_handles()
    return calcu_kernel_best_vsm_count(kernel, num_warps=run_config["stage1_num_warps"])


def token_decode_attention_mtp_diverse_vsm_single_token(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor, b_seq_len: torch.Tensor, b_mark_shared_group: torch.Tensor,
    out=None, alloc_tensor_func=torch.empty, **run_config
):
    batch_size = b_seq_len.shape[0]
    q_head_num, q_head_dim = q.shape[1], q.shape[2]
    kv_head_num = k.shape[1]
    gqa_group_size = q_head_num // kv_head_num
    sm_scale = 1.0 / (q_head_dim ** 0.5)

    if not run_config:
        run_config = MTPDiverseVSMKernelConfig.try_to_get_best_config(
            batch_size, b_seq_len.max().item(), q_head_num, q_head_dim, kv_head_num, str(q.dtype))

    if out is None:
        out = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device)

    num_vsm = estimate_stage1_vsm(
        q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group,
        torch.empty([q_head_num, 0, q_head_dim], dtype=torch.float32, device="cuda"),
        torch.empty([q_head_num, 0], dtype=torch.float32, device="cuda"), sm_scale, **run_config)

    if not hasattr(b_seq_len, "mtp_decode_att_block_seq"):
        decode_att_block_seq = torch.empty([1], dtype=torch.int64, device="cuda")
        mid_o_batch_start_index = torch.empty([batch_size], dtype=torch.int64, device="cuda")
        _fwd_kernel_calcu_index_and_block_seq[(1,)](
            b_seq_len, b_mark_shared_group, decode_att_block_seq, mid_o_batch_start_index,
            num_vsm, batch_size, BLOCK_N=run_config["BLOCK_N"], MAX_BATCH_SIZE=triton.next_power_of_2(batch_size), num_warps=4)
        b_seq_len.mtp_decode_att_block_seq = decode_att_block_seq
        b_seq_len.mtp_batch_start_index = mid_o_batch_start_index

    mid_o = alloc_tensor_func([q_head_num, num_vsm * 4 + batch_size, q_head_dim], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = alloc_tensor_func([q_head_num, num_vsm * 4 + batch_size], dtype=torch.float32, device="cuda")

    mtp_diverse_vsm_stage1(
        b_seq_len.mtp_decode_att_block_seq, q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group,
        mid_o, mid_o_logexpsum, sm_scale, num_vsm, gqa_group_size, q_head_num, kv_head_num, batch_size, **run_config)
    mtp_diverse_vsm_stage2(
        b_seq_len.mtp_decode_att_block_seq, b_seq_len.mtp_batch_start_index, mid_o, mid_o_logexpsum, b_seq_len, b_mark_shared_group, out, **run_config)
    return out
