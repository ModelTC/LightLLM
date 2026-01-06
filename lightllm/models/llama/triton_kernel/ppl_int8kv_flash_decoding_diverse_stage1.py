import torch
import triton
import triton.language as tl
from typing import Optional
from lightllm.common.kernel_config import KernelConfigs
from frozendict import frozendict
from functools import lru_cache
from typing import Dict


class GQADiverseDecodeStage1KernelConfig(KernelConfigs):
    kernel_name: str = "_fwd_kernel_flash_decode_diverse_stage1:v1"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        batch_size: int,
        avg_seq_len_in_batch: int,
        gqa_group_size: int,
        max_batch_group_size: int,
        q_head_dim: int,
        block_seq: int,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "gqa_group_size": gqa_group_size,
            "max_batch_group_size": max_batch_group_size,
            "q_head_dim": q_head_dim,
            "block_seq": block_seq,
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
                "BLOCK_N": 16,
                "num_warps": 2,
                "num_stages": 2,
            }
        return config

    @classmethod
    def save_config(
        cls,
        gqa_group_size: int,
        max_batch_group_size: int,
        q_head_dim: int,
        block_seq: int,
        out_dtype: str,
        config_json: Dict[int, Dict[int, Dict]],
    ):
        key_params = {
            "gqa_group_size": gqa_group_size,
            "max_batch_group_size": max_batch_group_size,
            "q_head_dim": q_head_dim,
            "block_seq": block_seq,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)


@triton.jit
def _fwd_kernel_flash_decode_diverse_stage1(
    Q,
    stride_qbs,
    stride_qh,
    stride_qd,
    K,
    K_scale,
    stride_kbs,
    stride_kh,
    stride_kd,
    V,
    V_scale,
    stride_vbs,
    stride_vh,
    stride_vd,
    sm_scale,
    Req_to_tokens,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    B_req_idx,
    b_shared_seq_len,
    b_mark_shared_group,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    KV_QUANT_GROUP_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    shared_batch_group_size = tl.load(b_mark_shared_group + cur_batch)
    if shared_batch_group_size == 0:
        return
    cur_batch_end = cur_batch + 1
    cur_batch = cur_batch - (shared_batch_group_size - 1)
    cur_kv_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    cur_q_head_range = cur_kv_head * gqa_group_size + tl.arange(0, BLOCK_HEAD)
    q_head_end_index = (cur_kv_head + 1) * gqa_group_size
    cur_q_head_range = tl.where(cur_q_head_range < q_head_end_index, cur_q_head_range, cur_kv_head * gqa_group_size)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    cur_batch_seq_len = tl.load(b_shared_seq_len + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    offs_batch = cur_batch + tl.arange(0, BLOCK_BATCH)
    offs_batch = tl.where(offs_batch < cur_batch_end, offs_batch, cur_batch)

    off_q = offs_batch[:, None, None] * stride_qbs + cur_q_head_range[None, :, None] * stride_qh + offs_d[None, None, :]

    block_n_size = tl.cdiv(
        tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index),
        BLOCK_N,
    )

    if block_n_size == 0:
        return

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    Q_BATCH_HEAD_NUM: tl.constexpr = BLOCK_BATCH * BLOCK_HEAD
    q = tl.load(Q + off_q).reshape(Q_BATCH_HEAD_NUM, BLOCK_HEADDIM)

    sum_exp = tl.zeros([Q_BATCH_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.zeros([Q_BATCH_HEAD_NUM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Q_BATCH_HEAD_NUM, BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        n_mask = offs_n_new < cur_batch_end_index
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=n_mask,
            other=0,
        ).to(tl.int64)
        off_k = k_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        off_k_scale = off_k // KV_QUANT_GROUP_SIZE
        k = tl.load(K + off_k, mask=n_mask[None, :], other=0)
        k_scale = tl.load(K_scale + off_k_scale, mask=n_mask[None, :], other=0.0)
        k = k * k_scale
        att_value = tl.dot(q, k.to(q.dtype))
        att_value *= sm_scale
        att_value = tl.where(n_mask[None, :], att_value, float("-inf"))
        v = tl.load(
            V + off_k.T,
            mask=n_mask[:, None],
            other=0,
        )
        v_scale = tl.load(
            V_scale + off_k_scale.T,
            mask=n_mask[:, None],
            other=0.0,
        )
        v = v * v_scale

        cur_max_logic = tl.max(att_value, axis=1)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic[:, None])
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(q.dtype), v.to(q.dtype))

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic

    off_mid_o = (
        offs_batch[:, None, None] * stride_mid_ob
        + cur_q_head_range[None, :, None] * stride_mid_oh
        + seq_start_block * stride_mid_os
        + offs_d[None, None, :]
    )
    off_mid_o_logexpsum = (
        offs_batch[:, None] * stride_mid_o_eb + cur_q_head_range[None, :] * stride_mid_o_eh + seq_start_block
    )
    tl.store(
        Mid_O + off_mid_o,
        (acc / sum_exp[:, None]).reshape(BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM),
    )
    tl.store(
        Mid_O_LogExpSum + off_mid_o_logexpsum,
        (max_logic + tl.log(sum_exp)).reshape(BLOCK_BATCH, BLOCK_HEAD),
    )
    return


@torch.no_grad()
def flash_decode_stage1(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    b_shared_seq_len: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    max_len_in_batch: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    block_seq: int,
    max_batch_group_size: int,
    run_config: Optional[dict] = None,
):
    """
    该kernel是为多样性生成定制的gqa算子,其中 b_mark_shared_group 是一个shape 为 (batch_size,)的tensor,
    其内容标记那些请求是共享前缀的请求组。举列说明:
    b_shared_seq_len : [10, 10, 10, 11, 11, 11, 11]
    b_mark_shared_group: [0, 0, 3, 0, 0, 0, 4]
    b_mark_shared_group 中每一个不为0的位置都代表其与前面多少个请求形成一个共享前缀组。属于
    同一个共享前缀组的请求, 其在对应的 b_shared_seq_len 中的内容必然相同。
    """
    if not run_config:
        avg_seq_len_in_batch = max_len_in_batch

        run_config = GQADiverseDecodeStage1KernelConfig.try_to_get_best_config(
            batch_size=int(q.shape[0]),
            avg_seq_len_in_batch=avg_seq_len_in_batch,
            gqa_group_size=int(q.shape[1] // k.shape[1]),
            max_batch_group_size=max_batch_group_size,
            q_head_dim=int(q.shape[2]),
            block_seq=block_seq,
            out_dtype=q.dtype,
        )

    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    BLOCK_SEQ = block_seq
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, kv_head_num = B_req_idx.shape[0], k.shape[1]
    grid = (batch, kv_head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))
    gqa_group_size = q.shape[1] // k.shape[1]
    assert triton.next_power_of_2(Lk) == Lk
    KV_QUANT_GROUP_SIZE = v.shape[-1] // v_scale.shape[-1]
    assert KV_QUANT_GROUP_SIZE == 8
    BLOCK_HEAD = triton.next_power_of_2(gqa_group_size)
    BLOCK_BATCH = triton.next_power_of_2(max_batch_group_size)
    if BLOCK_HEAD * BLOCK_BATCH < 16:
        BLOCK_BATCH = 16 // BLOCK_HEAD

    _fwd_kernel_flash_decode_diverse_stage1[grid](
        Q=q,
        stride_qbs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        K=k,
        K_scale=k_scale,
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        V=v,
        V_scale=v_scale,
        stride_vbs=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        sm_scale=sm_scale,
        Req_to_tokens=Req_to_tokens,
        stride_req_to_tokens_b=Req_to_tokens.stride(0),
        stride_req_to_tokens_s=Req_to_tokens.stride(1),
        B_req_idx=B_req_idx,
        b_shared_seq_len=b_shared_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        Mid_O=mid_out,
        stride_mid_ob=mid_out.stride(0),
        stride_mid_oh=mid_out.stride(1),
        stride_mid_os=mid_out.stride(2),
        stride_mid_od=mid_out.stride(3),
        Mid_O_LogExpSum=mid_out_logsumexp,  # [batch, head, seq_block_num]
        stride_mid_o_eb=mid_out_logsumexp.stride(0),
        stride_mid_o_eh=mid_out_logsumexp.stride(1),
        stride_mid_o_es=mid_out_logsumexp.stride(2),
        gqa_group_size=gqa_group_size,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=block_seq,
        BLOCK_HEADDIM=Lk,
        BLOCK_N=BLOCK_N,
        BLOCK_BATCH=BLOCK_BATCH,
        KV_QUANT_GROUP_SIZE=KV_QUANT_GROUP_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return
