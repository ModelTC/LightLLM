"""
MTP Diverse Attention Stage1 Kernel - Single Token Per Request Mode

简化版本（参考 int8kv diverse stage1）：
- 组内请求 [q1, q2, q3, q4]，group mark [0, 0, 0, 4]
- KV slots [kv0, kv1, kv2, kv3, kv4]
- 可见性：q1->[kv0], q2->[kv0,kv1], q3->[kv0,kv1,kv2], q4->[kv0,kv1,kv2,kv3,kv4]

核心逻辑：
- 只由组内最后一个请求（b_mark_shared_group != 0）触发计算
- 一次加载组内所有请求的 Q 和 KV
- 每个请求单独做可见性检查（基于各自的 seq_len）
- 中间结果按 kv block 存储，供 Stage2 聚合
"""
import torch
import triton
import triton.language as tl
from typing import Optional
from lightllm.common.triton_utils.autotuner import autotune, Autotuner, AutotuneKernelType, AutotuneLevel
from lightllm.utils.device_utils import is_hopper
from lightllm.utils.envs_utils import get_decode_attn_autotune_seq_len, get_triton_autotune_level


def get_test_configs():
    configs = []
    for block_n in [16, 32, 64]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                for warp_specialize in [True, False] if is_hopper() else [False]:
                    # warp_specialize only support hopper
                    configs.append(
                        {
                            "BLOCK_N": block_n,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "warp_specialize": warp_specialize,
                        }
                    )
    return configs


def get_static_key(q, k, block_batch):
    key_params = {
        "gqa_group_size": int(q.shape[1] // k.shape[1]),
        "q_head_dim": int(q.shape[2]),
        "block_batch": block_batch,
        "out_dtype": str(q.dtype),
    }
    return key_params


def get_run_key(q, max_kv_len):
    batch_size = q.shape[0]
    # 正常查找使用调用方在 Graph 改写长度上限前保存的真实 KV 长度，不读取 GPU 张量或请求表容量。
    max_kv_len = int(max_kv_len)
    if Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION) and get_triton_autotune_level() in [
        AutotuneLevel.ADAPTIVE_AUTOTUNE,
        AutotuneLevel.FORCE_AUTOTUNE,
    ]:
        max_kv_len = get_decode_attn_autotune_seq_len()
    # 调优和正常查找统一按 512 token 向上分桶；实际 benchmark 保留环境变量指定的精确长度。
    max_kv_len = (max_kv_len + 511) // 512 * 512
    return batch_size * 1000 * 1000 * 1000 + max_kv_len


def rebuild_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    max_kv_len: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    block_batch: int,
    **kwargs,
):
    # Graph 初始化输入只有很短的 HOLD 请求，且每行独立成组，不能代表 MTP 的长 KV 共享计算。
    # 仅在实际搜索前构造一次调优输入，开销不计入 benchmark；正式执行和捕获仍使用原始输入。
    batch_size = q.shape[0]
    max_kv_len = get_decode_attn_autotune_seq_len()
    assert k.shape[0] == v.shape[0], "K/V caches must have the same number of tokens"
    num_tokens = k.shape[0]
    if num_tokens == 0:
        raise ValueError("MTP decode autotuning requires a non-empty KV cache")
    assert block_batch > 0, "block_batch must be positive"

    # 以 block_batch 为代表性共享组大小，尾组允许不足；KV 长度不改变分组方式。
    # 调优使用固定代表性分组，不从 HOLD 标记推断真实组大小，也不改动正常请求的动态分组。
    group_size = block_batch
    # 组内长度递增且组末长度等于目标长度，目标长度必须能保证组首至少有一个可见 KV。
    if max_kv_len < min(batch_size, group_size):
        raise ValueError("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN must be at least the largest MTP group size")
    num_groups = (batch_size + group_size - 1) // group_size

    # 按每组实际行数生成列表，再构造 CPU tensor 并转回原设备；尾组可能不足 group_size。
    group_sizes = [min(group_size, batch_size - start) for start in range(0, batch_size, group_size)]
    cpu_req_idx = torch.tensor(
        [group_idx for group_idx, size in enumerate(group_sizes) for _ in range(size)],
        dtype=B_req_idx.dtype,
        device="cpu",
    )
    # 同组请求共享同一行 KV 映射；例如目标长度 16384、组大小 3，长度为 [16382, 16383, 16384]。
    cpu_seq_len = torch.tensor(
        [length for size in group_sizes for length in range(max_kv_len - size + 1, max_kv_len + 1)],
        dtype=b_seq_len.dtype,
        device="cpu",
    )
    # 只有组末行标记组大小并启动计算；其他行保持 0，由组末行一次处理整组 Q。
    cpu_mark_shared_group = torch.tensor(
        [size if offset == size - 1 else 0 for size in group_sizes for offset in range(size)],
        dtype=b_mark_shared_group.dtype,
        device="cpu",
    )
    B_req_idx = cpu_req_idx.to(device=B_req_idx.device)
    b_seq_len = cpu_seq_len.to(device=b_seq_len.device)
    b_mark_shared_group = cpu_mark_shared_group.to(device=b_mark_shared_group.device)

    # 每组只需要一行已初始化的映射，不能扩展长度后读取原请求表中的未初始化条目。
    # 取模将所有物理索引限制在已有 K/V 池内，避免为长请求重新分配完整缓存；物理容量不足时，
    # 不同位置会复用 K/V，可能提高 GPU 缓存命中率，因此调优的访存特征仍受现有缓存容量影响。
    Req_to_tokens = torch.arange(num_groups * max_kv_len, dtype=Req_to_tokens.dtype, device=Req_to_tokens.device)
    Req_to_tokens = Req_to_tokens.remainder_(num_tokens).view(num_groups, max_kv_len)

    # 保留 Q/K/V、block_batch 和中间缓冲区布局；stage1 的每个 program 可循环处理多个 KV 块。
    # 选好配置后使用原始输入重新覆盖有效中间块，并返回对应的 BLOCK_N 供 stage2 归约。
    return (
        q,
        k,
        v,
        Req_to_tokens,
        B_req_idx,
        b_seq_len,
        b_mark_shared_group,
        max_kv_len,
        mid_out,
        mid_out_logsumexp,
        block_batch,
    ), kwargs


@triton.jit
def _fwd_kernel_mtp_diverse_stage1_single_token(
    Q,
    stride_qb,
    stride_qh,
    stride_qd,
    K,
    stride_kbs,
    stride_kh,
    stride_kd,
    V,
    stride_vbs,
    stride_vh,
    stride_vd,
    sm_scale,
    Req_to_tokens,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    B_req_idx,
    b_seq_len,
    b_mark_shared_group,
    Mid_O,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    Mid_O_LogExpSum,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    block_index = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_batch = tl.program_id(2)
    grid_block_num = tl.num_programs(0)

    shared_batch_group_size = tl.load(b_mark_shared_group + cur_batch)
    if shared_batch_group_size == 0:
        return

    cur_batch_start = cur_batch - (shared_batch_group_size - 1)

    # ---- batch lane: 不再回卷索引，使用mask ----
    offs_b = tl.arange(0, BLOCK_BATCH)
    batch_idx = tl.where(offs_b < shared_batch_group_size, cur_batch_start + offs_b, cur_batch)

    # load seq len
    batch_seq_lens = tl.load(b_seq_len + batch_idx)
    max_seq_len = tl.max(batch_seq_lens, axis=0)
    block_num = tl.cdiv(max_seq_len, BLOCK_N)
    if block_index >= block_num:
        return

    batch_seq_lens = tl.broadcast_to(batch_seq_lens[:, None], (BLOCK_BATCH, BLOCK_HEAD))
    batch_seq_lens = batch_seq_lens.reshape((BLOCK_BATCH * BLOCK_HEAD,))

    # ---- head lane: 不再next_pow2回卷，使用mask  ---    -
    offs_h = tl.arange(0, BLOCK_HEAD)
    q_head_idx = tl.where(offs_h < gqa_group_size, cur_kv_head * gqa_group_size + offs_h, cur_kv_head * gqa_group_size)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    off_q = batch_idx[:, None, None] * stride_qb + q_head_idx[None, :, None] * stride_qh + offs_d[None, None, :]
    q = tl.load(Q + off_q)
    q_flat = tl.reshape(q, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM))

    sum_exp = tl.zeros([BLOCK_BATCH * BLOCK_HEAD], dtype=tl.float32)
    max_logic = tl.full([BLOCK_BATCH * BLOCK_HEAD], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM], dtype=tl.float32)

    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    for iter_block_index in tl.range(block_index, block_num, grid_block_num, warp_specialize=warp_specialize):
        offs_n_new = iter_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n_refator = tl.where(offs_n_new < max_seq_len, offs_n_new, max_seq_len - 1)

        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_refator * stride_req_to_tokens_s,
        ).to(tl.int64)
        off_k = k_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        off_v = k_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
        k = tl.load(K + off_k)
        v = tl.load(V + off_v)
        att = tl.dot(q_flat, k)
        att *= sm_scale
        att = tl.where(offs_n_new[None, :] < batch_seq_lens[:, None], att, -1000000000.0)
        cur_max = tl.max(att, axis=1)
        new_max = tl.maximum(cur_max, max_logic)

        exp_logic = tl.exp(att - new_max[:, None])
        logic_scale = tl.exp(max_logic - new_max)

        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(v.dtype), v)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max

    mid_o_val = acc / sum_exp[:, None]
    mid_lse_val = max_logic + tl.log(sum_exp)

    off_mid_o = (
        batch_idx[:, None, None] * stride_mid_ob
        + q_head_idx[None, :, None] * stride_mid_oh
        + block_index * stride_mid_os
        + offs_d[None, None, :] * stride_mid_od
    )
    off_mid_lse = (
        batch_idx[:, None] * stride_mid_o_eb + q_head_idx[None, :] * stride_mid_o_eh + block_index * stride_mid_o_es
    )

    tl.store(Mid_O + off_mid_o, mid_o_val.reshape((BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM)))
    tl.store(Mid_O_LogExpSum + off_mid_lse, mid_lse_val.reshape((BLOCK_BATCH, BLOCK_HEAD)))


@autotune(
    kernel_name="_fwd_kernel_mtp_diverse_stage1_single_token:v3",
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    rebuild_input_func=rebuild_inputs,
    # stage1 对有效中间块执行覆盖写，候选配置不会读取已有输出，正式执行也会重新覆盖真实请求的有效块。
    # 不标记这两个大缓冲区，避免每个候选配置 benchmark 时反复 clone，增加显存峰值和拷贝开销。
    # mutates_args=["mid_out", "mid_out_logsumexp"],
)
def mtp_diverse_stage1_single_token(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    max_kv_len: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    block_batch: int,
    run_config: Optional[dict] = None,
):
    """
    MTP Diverse Attention Stage1 - Single Token Per Request Mode

    b_seq_len: 每个请求可见的 KV 数量，组内递增
    例如组内 [q1, q2, q3, q4] 对应 b_seq_len [2, 3, 4, 5]
    """
    if not run_config:
        run_config = {"BLOCK_N": 16, "num_warps": 2, "num_stages": 2, "warp_specialize": False}  # 默认配置

    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]
    warp_specialize = run_config.get("warp_specialize", False)
    BLOCK_BATCH = triton.next_power_of_2(block_batch)

    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert Req_to_tokens.is_cuda and B_req_idx.is_cuda and b_seq_len.is_cuda and b_mark_shared_group.is_cuda
    assert mid_out.is_cuda and mid_out_logsumexp.is_cuda

    Lq, Lk = int(q.shape[2]), int(k.shape[2])
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    batch = int(B_req_idx.shape[0])
    kv_head_num = int(k.shape[1])
    q_head_num = int(q.shape[1])
    assert q_head_num % kv_head_num == 0
    gqa_group_size = q_head_num // kv_head_num
    BLOCK_HEAD = triton.next_power_of_2(gqa_group_size)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1

    grid_num = mid_out.shape[2]

    sm_scale = 1.0 / (Lk ** 0.5)
    # 固定 grid（graph friendly）
    grid = (grid_num, kv_head_num, batch)
    _fwd_kernel_mtp_diverse_stage1_single_token[grid](
        Q=q,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        K=k,
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        V=v,
        stride_vbs=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        sm_scale=sm_scale,
        Req_to_tokens=Req_to_tokens,
        stride_req_to_tokens_b=Req_to_tokens.stride(0),
        stride_req_to_tokens_s=Req_to_tokens.stride(1),
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        Mid_O=mid_out,
        stride_mid_ob=mid_out.stride(0),
        stride_mid_oh=mid_out.stride(1),
        stride_mid_os=mid_out.stride(2),
        stride_mid_od=mid_out.stride(3),
        Mid_O_LogExpSum=mid_out_logsumexp,
        stride_mid_o_eb=mid_out_logsumexp.stride(0),
        stride_mid_o_eh=mid_out_logsumexp.stride(1),
        stride_mid_o_es=mid_out_logsumexp.stride(2),
        gqa_group_size=gqa_group_size,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_HEADDIM=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
        warp_specialize=warp_specialize,
    )
    return BLOCK_N


if __name__ == "__main__":
    if get_triton_autotune_level() != 2:
        raise Exception("you need set env LIGHTLLM_TRITON_AUTOTUNE_LEVEL=2 to start program.")

    # static params
    q_head_dim = 128
    block_batch = 4
    out_dtype = torch.bfloat16

    batch_sizes = [1, 8, 16, 32, 64, 128]
    decode_lengths = [get_decode_attn_autotune_seq_len()]

    tp_world_size = 2
    q_head_num = 64 // tp_world_size
    k_head_num = 8 // tp_world_size

    gqa_group_size = q_head_num // k_head_num

    Autotuner.start_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
    # autotuing kernel
    for batch_size in batch_sizes:
        for length in decode_lengths:
            # Setup test tensors
            q = torch.randn(batch_size, q_head_num, q_head_dim, dtype=out_dtype, device="cuda")
            k = torch.randn(batch_size * length, k_head_num, q_head_dim, dtype=out_dtype, device="cuda")
            v = torch.randn(batch_size * length, k_head_num, q_head_dim, dtype=out_dtype, device="cuda")
            Req_to_tokens = torch.arange(0, batch_size * length, dtype=torch.int32, device="cuda").view(
                batch_size, length
            )
            B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
            B_seq_len = torch.full((batch_size,), length, dtype=torch.int32, device="cuda")
            b_mark_shared_group = torch.ones(batch_size, dtype=torch.int32, device="cuda")

            if batch_size <= 16:
                block_num = 128
            elif batch_size <= 64:
                block_num = 64
            else:
                block_num = 32

            mid_out = torch.zeros(batch_size, q_head_num, block_num, q_head_dim, dtype=out_dtype, device="cuda")
            mid_out_logsumexp = torch.zeros(batch_size, q_head_num, block_num, dtype=out_dtype, device="cuda")

            mtp_diverse_stage1_single_token(
                q=q,
                k=k,
                v=v,
                Req_to_tokens=Req_to_tokens,
                B_req_idx=B_req_idx,
                b_seq_len=B_seq_len,
                b_mark_shared_group=b_mark_shared_group,
                max_kv_len=length,
                mid_out=mid_out,
                mid_out_logsumexp=mid_out_logsumexp,
                block_batch=block_batch,
            )

    Autotuner.end_autotune_warmup()
