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


@autotune(
    kernel_name="_fwd_kernel_flash_decode_diverse_stage2:v1",
    configs_gen_func=get_test_configs,
    static_key_func=_get_static_key,
    run_key_func=run_key_func,
    mutates_args=["mid_out", "mid_out_logsumexp"],
)
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


def create_tensors(batch_size, seq_len):
    shared_seq_len = 0
    num_heads = 32
    kv_head_num = 8
    head_dim = 128
    # 修复：max_len_in_batch 在实际上是 graph_max_len_in_batch，默认8192
    # 否则 kernel 会尝试用 max_len_in_batch 索引 Req_to_tokens 和 KV cache，导致越界访问
    max_len_in_batch = 8192
    block_seq = 256
    quant_group_size = 8

    test_dtype = torch.bfloat16

    kv_shape = (batch_size * seq_len, kv_head_num, head_dim)
    kv_scale_shape = (batch_size * seq_len, kv_head_num, head_dim // quant_group_size)

    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device="cuda")
    k = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    k_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda")
    v = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    v_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda")
    Req_to_tokens = torch.arange(0, seq_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, seq_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.full((batch_size,), shared_seq_len, dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.ones(batch_size, dtype=torch.int32, device="cuda")
    mid_out = torch.zeros(
        size=(batch_size, num_heads, (max_len_in_batch // block_seq) + 2, head_dim), dtype=q.dtype, device="cuda"
    )
    mid_out_logsumexp = torch.zeros(
        size=(batch_size, num_heads, (max_len_in_batch // block_seq) + 2), dtype=q.dtype, device="cuda"
    )
    return {
        "q": q,
        "k": k,
        "k_scale": k_scale,
        "v": v,
        "v_scale": v_scale,
        "Req_to_tokens": Req_to_tokens,
        "B_req_idx": B_req_idx,
        "b_seq_len": b_seq_len,
        "b_shared_seq_len": b_shared_seq_len,
        "b_mark_shared_group": b_mark_shared_group,
        "max_len_in_batch": max_len_in_batch,
        "mid_out": mid_out,
        "mid_out_logsumexp": mid_out_logsumexp,
        "block_seq": block_seq,
        "head_dim": head_dim,
    }


if __name__ == "__main__":
    batch_sizes = [8, 16, 32, 64]
    seq_lens = [32, 64, 128, 256]
    from lightllm.utils.light_utils import light_ops

    # autotune
    Autotuner.start_autotune_warmup()
    for batch in batch_sizes:
        for seq in seq_lens:
            print(f"\n[batch={batch}, seq={seq}] Running autotune...")
            setup_tensors = create_tensors(batch, seq)
            flash_decode_stage2(
                q=setup_tensors["q"],
                k=setup_tensors["k"],
                k_scale=setup_tensors["k_scale"],
                v=setup_tensors["v"],
                v_scale=setup_tensors["v_scale"],
                Req_to_tokens=setup_tensors["Req_to_tokens"],
                B_req_idx=setup_tensors["B_req_idx"],
                B_Seqlen=setup_tensors["b_seq_len"],
                b_shared_seq_len=setup_tensors["b_shared_seq_len"],
                max_len_in_batch=setup_tensors["max_len_in_batch"],
                mid_out=setup_tensors["mid_out"],
                mid_out_logsumexp=setup_tensors["mid_out_logsumexp"],
                block_seq=setup_tensors["block_seq"],
            )
            print(f"Autotune completed for batch {batch} and seq {seq}")
            del setup_tensors
            torch.cuda.empty_cache()

    Autotuner.end_autotune_warmup()
    print("\n" + "=" * 80)
    print("All autotune completed! Now starting benchmarks...")
    print("=" * 80)

    results = []
    for batch in batch_sizes:
        for seq in seq_lens:
            # 清理 GPU 缓存，避免内存碎片化导致 CUDA Graph 捕获失败
            torch.cuda.empty_cache()
            # torch.cuda.synchronize()

            setup_tensors = create_tensors(batch, seq)

            # 准备 CUDA 实现的输出 tensor
            mid_out_cuda = setup_tensors["mid_out"].clone()
            mid_out_logsumexp_cuda = setup_tensors["mid_out_logsumexp"].clone()

            # 准备 Triton 实现的输出 tensor
            mid_out_triton = setup_tensors["mid_out"].clone()
            mid_out_logsumexp_triton = setup_tensors["mid_out_logsumexp"].clone()

            # 先运行一次 CUDA 获取结果
            light_ops.group8_int8kv_flashdecoding_diverse_stage2(
                setup_tensors["block_seq"],
                mid_out_cuda,
                mid_out_logsumexp_cuda,
                1.0 / (setup_tensors["head_dim"] ** 0.5),
                setup_tensors["q"],
                setup_tensors["k"],
                setup_tensors["k_scale"],
                setup_tensors["v"],
                setup_tensors["v_scale"],
                setup_tensors["Req_to_tokens"],
                setup_tensors["B_req_idx"],
                setup_tensors["b_seq_len"],
                setup_tensors["b_shared_seq_len"],
                setup_tensors["max_len_in_batch"],
            )

            # 再运行一次 Triton 获取结果
            flash_decode_stage2(
                q=setup_tensors["q"],
                k=setup_tensors["k"],
                k_scale=setup_tensors["k_scale"],
                v=setup_tensors["v"],
                v_scale=setup_tensors["v_scale"],
                Req_to_tokens=setup_tensors["Req_to_tokens"],
                B_req_idx=setup_tensors["B_req_idx"],
                B_Seqlen=setup_tensors["b_seq_len"],
                b_shared_seq_len=setup_tensors["b_shared_seq_len"],
                max_len_in_batch=setup_tensors["max_len_in_batch"],
                mid_out=mid_out_triton,
                mid_out_logsumexp=mid_out_logsumexp_triton,
                block_seq=setup_tensors["block_seq"],
            )

            # 比较结果一致性
            diff_mid_out = torch.abs(mid_out_cuda - mid_out_triton)
            diff_logsumexp = torch.abs(mid_out_logsumexp_cuda - mid_out_logsumexp_triton)
            max_diff_out = diff_mid_out.max().item()
            max_diff_logsumexp = diff_logsumexp.max().item()
            mean_diff_out = diff_mid_out.mean().item()
            mean_diff_logsumexp = diff_logsumexp.mean().item()

            # 计算余弦相似度
            cos_sim_out = torch.nn.functional.cosine_similarity(
                mid_out_cuda.flatten(), mid_out_triton.flatten(), dim=0
            ).item()
            cos_sim_logsumexp = torch.nn.functional.cosine_similarity(
                mid_out_logsumexp_cuda.flatten(), mid_out_logsumexp_triton.flatten(), dim=0
            ).item()

            print(f"\n[batch={batch}, seq={seq}] 一致性检查:")
            print("  mid_out:")
            print(f"    max_diff: {max_diff_out:.6f}, mean_diff: {mean_diff_out:.6f}, cosine_sim: {cos_sim_out:.8f}")
            print("  logsumexp:")
            print(
                f"    max_diff: {max_diff_logsumexp:.6f}, "
                f"mean_diff: {mean_diff_logsumexp:.6f}, "
                f"cosine_sim: {cos_sim_logsumexp:.8f}"
            )

            # 性能测试
            fn_cuda = lambda: light_ops.group8_int8kv_flashdecoding_diverse_stage2(
                setup_tensors["block_seq"],
                setup_tensors["mid_out"],
                setup_tensors["mid_out_logsumexp"],
                1.0 / (setup_tensors["head_dim"] ** 0.5),
                setup_tensors["q"],
                setup_tensors["k"],
                setup_tensors["k_scale"],
                setup_tensors["v"],
                setup_tensors["v_scale"],
                setup_tensors["Req_to_tokens"],
                setup_tensors["B_req_idx"],
                setup_tensors["b_seq_len"],
                setup_tensors["b_shared_seq_len"],
                setup_tensors["max_len_in_batch"],
            )
            ms_cuda = triton.testing.do_bench_cudagraph(fn_cuda, rep=100)

            fn_triton = lambda: flash_decode_stage2(
                q=setup_tensors["q"],
                k=setup_tensors["k"],
                k_scale=setup_tensors["k_scale"],
                v=setup_tensors["v"],
                v_scale=setup_tensors["v_scale"],
                Req_to_tokens=setup_tensors["Req_to_tokens"],
                B_req_idx=setup_tensors["B_req_idx"],
                B_Seqlen=setup_tensors["b_seq_len"],
                b_shared_seq_len=setup_tensors["b_shared_seq_len"],
                max_len_in_batch=setup_tensors["max_len_in_batch"],
                mid_out=setup_tensors["mid_out"],
                mid_out_logsumexp=setup_tensors["mid_out_logsumexp"],
                block_seq=setup_tensors["block_seq"],
            )
            ms_triton = triton.testing.do_bench_cudagraph(fn_triton, rep=100)

            results.append(
                {
                    "batch_size": batch,
                    "seq_len": seq,
                    "triton_ms": ms_triton,
                    "cuda_ms": ms_cuda,
                }
            )
            print(results[-1])

            # 清理本次迭代的张量
            del setup_tensors

    # 打印汇总结果
    print(f"\n{'='*80}")
    print("SUMMARY - Performance Comparison")
    print(f"{'='*80}")
    print(f"{'batch_size':<8} {'seq_len':<12} {'triton_ms':<12} {'cuda_ms':<12} {'vs cuda':<10}")
    print(f"{'-'*80}")
    for r in results:
        vs_cuda = f"{r['cuda_ms']/r['triton_ms']:.2f}x"
        emoji = "🎉" if r["triton_ms"] < r["cuda_ms"] else ""
        print(
            f"{r['batch_size']:<8} {r['seq_len']:<12} {r['triton_ms']:<12.3f} {r['cuda_ms']:<12.3f}"
            f"{vs_cuda:<10} {emoji}"
        )
    print(f"{'='*80}")
