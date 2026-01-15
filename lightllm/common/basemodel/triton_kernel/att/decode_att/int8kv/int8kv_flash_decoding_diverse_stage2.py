import torch
import triton
import triton.language as tl
from typing import Optional

from lightllm.common.kernel_config import KernelConfigs
from frozendict import frozendict
from functools import lru_cache
from typing import Dict
from lightllm.common.triton_utils.autotuner import autotune, Autotuner


class GQADiverseDecodeStage2KernelConfig(KernelConfigs):
    kernel_name: str = "_fwd_kernel_flash_decode_diverse_stage2:v1"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        batch_size: int,
        avg_seq_len_in_batch: int,
        gqa_group_size: int,
        q_head_dim: int,
        block_seq: int,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "gqa_group_size": gqa_group_size,
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
        q_head_dim: int,
        block_seq: int,
        out_dtype: str,
        config_json: Dict[int, Dict[int, Dict]],
    ):
        key_params = {
            "gqa_group_size": gqa_group_size,
            "q_head_dim": q_head_dim,
            "block_seq": block_seq,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)


@triton.jit
def _fwd_kernel_flash_decode_diverse_stage2(
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
    B_Seqlen,
    b_shared_seq_len,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_QUANT_GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    cur_q_head_range = cur_kv_head * gqa_group_size + tl.arange(0, gqa_group_size)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d_scale = tl.arange(0, NUM_GROUPS)
    cur_batch_shared_len = tl.load(b_shared_seq_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ + cur_batch_shared_len
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)
    store_seq_block = seq_start_block + tl.cdiv(cur_batch_shared_len, BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_q_head_range[:, None] * stride_qh + offs_d[None, :]

    block_n_size = tl.cdiv(
        tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index),
        BLOCK_N,
    )

    if block_n_size == 0:
        return

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    q = tl.load(Q + off_q)

    sum_exp = tl.zeros([gqa_group_size], dtype=tl.float32)
    max_logic = tl.zeros([gqa_group_size], dtype=tl.float32) - float("inf")
    acc = tl.zeros([gqa_group_size, BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        n_mask = offs_n_new < cur_batch_end_index
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=n_mask,
            other=0,
        ).to(tl.int64)
        off_k_base = k_loc * stride_kbs + cur_kv_head * stride_kh
        # (128, 16)
        off_k = off_k_base[None, :] + offs_d[:, None]
        # off_k_scale = off_k // KV_QUANT_GROUP_SIZE
        # (16, 16)
        off_k_scale = off_k_base[None, :] // KV_QUANT_GROUP_SIZE + offs_d_scale[:, None]
        k = tl.load(K + off_k, mask=n_mask[None, :], other=0)
        k = tl.reshape(k, (NUM_GROUPS, KV_QUANT_GROUP_SIZE, BLOCK_N))
        k_scale = tl.load(K_scale + off_k_scale, mask=n_mask[None, :], other=0.0)
        k_scale = tl.reshape(k_scale, (NUM_GROUPS, 1, BLOCK_N))
        k = k * k_scale
        k = tl.reshape(k, (BLOCK_HEADDIM, BLOCK_N))
        # q (4, 128) k (128, BLOCK_N)
        att_value = tl.dot(q, k.to(q.dtype))
        att_value *= sm_scale
        att_value = tl.where(n_mask[None, :], att_value, float("-inf"))
        off_v = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        v = tl.load(
            V + off_v,
            mask=n_mask[:, None],
            other=0,
        )
        v = tl.reshape(v, (BLOCK_N, NUM_GROUPS, KV_QUANT_GROUP_SIZE))
        v_scale = tl.load(
            V_scale + off_k_scale,
            mask=n_mask[None, :],
            other=0.0,
        )
        v_scale = tl.trans(v_scale)
        v_scale = tl.reshape(v_scale, (BLOCK_N, NUM_GROUPS, 1))
        v = v * v_scale
        v = tl.reshape(v, (BLOCK_N, BLOCK_HEADDIM))

        cur_max_logic = tl.max(att_value, axis=1)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic[:, None])
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(q.dtype), v.to(q.dtype))

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic

    off_mid_o = (
        cur_batch * stride_mid_ob
        + cur_q_head_range[:, None] * stride_mid_oh
        + store_seq_block * stride_mid_os
        + offs_d[None, :]
    )
    off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_q_head_range * stride_mid_o_eh + store_seq_block
    tl.store(
        Mid_O + off_mid_o,
        (acc / sum_exp[:, None]),
    )
    tl.store(
        Mid_O_LogExpSum + off_mid_o_logexpsum,
        (max_logic + tl.log(sum_exp)),
    )
    return


def get_test_configs():
    test_configs = []

    for block_n in [8, 16, 32, 64]:
        for num_warps in [
            2,
            4,
            8,
            16,
        ]:
            # for stage1_num_warps in [2, 4, 8, 16]:
            for num_stages in [
                1,
                2,
                3,
                4,
                5,
                7,
                9,
                10,
                11,
            ]:
                config = {
                    "BLOCK_N": block_n,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
                test_configs.append(config)

    return test_configs


def _get_static_key(q, k, block_seq):
    q_head_dim = q.shape[-1]
    gqa_group_size = q.shape[1] // k.shape[1]
    out_dtype = q.dtype
    return {
        "gqa_group_size": gqa_group_size,
        "q_head_dim": q_head_dim,
        "block_seq": block_seq,
        "out_dtype": str(out_dtype),
    }


def run_key_func(q, max_len_in_batch):
    return f"{q.shape[0]}_{max_len_in_batch}"


@torch.no_grad()
def flash_decode_stage2(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    B_Seqlen: torch.Tensor,
    b_shared_seq_len: torch.Tensor,
    max_len_in_batch: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    block_seq: int,
    run_config: Optional[dict] = None,
):
    if not run_config:
        run_config = GQADiverseDecodeStage2KernelConfig.try_to_get_best_config(
            batch_size=int(q.shape[0]),
            avg_seq_len_in_batch=max_len_in_batch,
            gqa_group_size=int(q.shape[1] // k.shape[1]),
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
    NUM_GROUPS = Lk // KV_QUANT_GROUP_SIZE
    assert triton.next_power_of_2(NUM_GROUPS) == NUM_GROUPS

    _fwd_kernel_flash_decode_diverse_stage2[grid](
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
        B_Seqlen=B_Seqlen,
        b_shared_seq_len=b_shared_seq_len,
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
        BLOCK_SEQ=block_seq,
        BLOCK_HEADDIM=Lk,
        BLOCK_N=BLOCK_N,
        KV_QUANT_GROUP_SIZE=KV_QUANT_GROUP_SIZE,
        NUM_GROUPS=NUM_GROUPS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return
