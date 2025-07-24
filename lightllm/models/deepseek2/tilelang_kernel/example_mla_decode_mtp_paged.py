# type: ignore
import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
from tilelang.profiler import do_bench
import math


@tilelang.jit(out_idx=[8])
def mla_decode_tilelang2(mtp_size, batch, h_q, h_kv, max_seqlen_pad, dv, dpe, block_N, block_H, num_split,
                        block_size):
    scale = (1.0 / (dv + dpe))**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = h_q // h_kv
    VALID_BLOCK_H = block_H
    assert h_kv == 1, "h_kv must be 1"
    assert batch % mtp_size == 0
    real_batch = batch // mtp_size
    
    # 因为block_size保证了KV token的连续，然后load kv的时候可以直接load连续的block_N个token
    # assert block_size >= block_N and block_size % block_N == 0, "block_size must be larger than block_N and a multiple of block_N"

    @T.macro
    def flash_mla_kernel(
            Q: T.Tensor([real_batch, h_q * mtp_size, dv], dtype),
            Q_pe: T.Tensor([real_batch, h_q * mtp_size, dpe], dtype),
            KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
            K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
            BLOCK_TABLE: T.Tensor([batch, max_seqlen_pad // block_size], "int32"),
            CACHE_SEQLENS: T.Tensor([batch], "int32"),
            Output: T.Tensor([real_batch, h_q * mtp_size, dv], dtype),
    ):
        # (64, 1)
        with T.Kernel(real_batch, 1, threads=256) as (bx, by):
            # shared memory
            Q_shared = T.alloc_shared([block_H, dv], dtype) # (32, 512)
            S_shared = T.alloc_shared([block_H, block_N], dtype) # (32, 64)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype) # (32, 64)
            KV_shared = T.alloc_shared([block_N, dv], dtype) # (64, 512)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype) # (64, 64)
            O_shared = T.alloc_shared([block_H, dv], dtype) # (32, 512)
            # registers
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype) # (32, 64)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype) # (32, 512)
            scores_max = T.alloc_fragment([block_H], accum_dtype) # (32)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype) # (32)
            scores_scale = T.alloc_fragment([block_H], accum_dtype) # (32)
            scores_sum = T.alloc_fragment([block_H], accum_dtype) # (32)
            logsum = T.alloc_fragment([block_H], accum_dtype) # (32)

            cur_kv_head = 0
            b_index_first = bx * mtp_size # origin req index
            b_index_last = b_index_first + mtp_size - 1
            seq_len = CACHE_SEQLENS[b_index_first]

            T.use_swizzle(10)
            T.annotate_layout({
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                S_shared: tilelang.layout.make_swizzled_layout(S_shared),
            })

            T.copy(Q[bx, 0:block_H, :], Q_shared)
            T.copy(Q_pe[bx, 0:block_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            
            # 取第一个req的长度，其他的另算
            # TODO: 要考虑CACHE_SEQLENS[b_index_first + 1]刚好比block_N多1的情况，直接mask可以搞定
            loop_range = T.ceildiv(seq_len + 1, block_N)
            for kr in T.Pipelined(loop_range, num_stages=2):
                k = loop_range - 1 - kr
                # block_size 为 page table的page大小1，token attention
                kv_start = BLOCK_TABLE[b_index_last, (k * block_N) //
                                        block_size] * block_size + (k * block_N) % block_size
                # 
                T.copy(KV[kv_start:kv_start + block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[kv_start:kv_start + block_N, cur_kv_head, :], K_pe_shared)
                T.clear(acc_s) # set to 0 for mma
                T.gemm( # (32, 512) * (64, 512)^T -> (32, 64)
                    Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                # (32, 64) * (64, 64)^T -> (32, 64)
                T.gemm(
                    Q_pe_shared,
                    K_pe_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol) #意思是计算的时候每列是完整的，也就是竖切
                T.copy(scores_max, scores_max_prev) # 把scores_max的值拷贝到scores_max_prev
                T.fill(scores_max, -T.infinity(accum_dtype)) # scores_max set为最小值
                if kr == 0: # 由于是从大到小遍历，所以kr == 0是最后的，这里是处理一下seqlen边界，也就是把第i个head 超过 seqlen的部分置为-inf
                    for i, j in T.Parallel(block_H, block_N):
                        seqlen_limit = T.if_then_else(i < h_q, seq_len, seq_len + 1)
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seqlen_limit,
                                                        -T.infinity(accum_dtype), acc_s[i, j])
                # 找到每个head的最大值, 但是 seq_len 为 block_N的整数倍的时候，最后一个块全被设置为-inf，scores_max的前16个也全是-inf
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale) # 矫正值 (block_H)

                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale) # 先把acc_s矩阵变成 exp(s - max(s))的形式
                
                if kr == 0 and seq_len % block_N == 0:
                    for i, j in T.Parallel(h_q, block_N):
                        acc_s[i, j] = 0.0
                        scores_scale[i] = 1.0

                T.reduce_sum(acc_s, scores_sum, dim=1) # sum起来得到每个H的分数和
                T.copy(acc_s, S_shared) # 寄存器的结果存到共享内存swizzled S_shared，并且accum_dtype到dtype
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i] # * 矫正值 + scores_sum
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i] # 乘以校正值
                # (32, 64) * (64, 512) -> (32, 512)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol) # acc_s_cast * o += acc_o
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] /= logsum[i] # 这里有可能是0
            T.copy(acc_o, O_shared) # 寄存器的结果存到共享内存swizzled O_shared，并且accum_dtype到dtype
            T.copy(O_shared, Output[bx, by * block_H:(by + 1) * block_H, :])

    @T.prim_func
    def main_no_split(
            Q: T.Tensor([real_batch, h_q * mtp_size, dv], dtype),
            Q_pe: T.Tensor([real_batch, h_q * mtp_size, dpe], dtype),
            KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
            K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
            block_table: T.Tensor([batch, max_seqlen_pad // block_size], "int32"),
            cache_seqlens: T.Tensor([batch], "int32"),
            glse: T.Tensor([batch, h_q, num_split], dtype),
            Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
            Output: T.Tensor([real_batch, h_q * mtp_size, dv], dtype),
    ):
        flash_mla_kernel(Q, Q_pe, KV, K_pe, block_table, cache_seqlens, Output)

    return main_no_split


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(
            s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q,
                  h_kv, d, dv, causal, dtype):
    # q: [b, s_q, h_q, d]
    # block_table: [b, max_seqlen_pad // block_size]
    # blocked_k: [b * max_seqlen_pad // block_size, block_size, h_kv, d]
    # cache_seqlens: [b]
    blocked_v = blocked_k[..., :dv]

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=q.device)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q.device)
        for i in range(b):
            # 获取该batch的有效序列长度
            seq_len = cache_seqlens[i]
            # 从block_table获取实际的kv索引
            kv_indices = block_table[i, :seq_len]  # 获取前seq_len个block索引

            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[kv_indices].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[kv_indices].transpose(0, 1),
                h_q,
                h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out.to(dtype), lse.to(dtype)

    out_torch, _ = ref_mla()
    return out_torch


def run_tilelang_mtp2(mtp_size, q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens,
                     h_q, h_kv, d, dv, causal, dtype):

    assert mtp_size == 2
    assert d > dv, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv].contiguous(), q[..., dv:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[...,
                                                                               dv:].contiguous()

    dpe = d - dv
    num_kv_splits = 1
    BLOCK_N = 64
    BLOCK_H = h_q * mtp_size

    out_partial = torch.empty(b, h_q, num_kv_splits, dv, dtype=dtype, device=q.device)
    glse = torch.empty(b, h_q, num_kv_splits, dtype=dtype, device=q.device)
    kernel = mla_decode_tilelang2(mtp_size, b, h_q, h_kv, max_seqlen_pad, dv, dpe, BLOCK_N, BLOCK_H,
                                 num_kv_splits, block_size)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    def flash_mla_tilelang():
        out = profiler.func(
            q_nope.view(-1, BLOCK_H, dv).contiguous(),
            q_pe.view(-1, BLOCK_H, dpe).contiguous(),
            blocked_k_nope.view(-1, h_kv, dv),
            blocked_k_pe.view(-1, h_kv, dpe),
            block_table,
            cache_seqlens,
            glse,
            out_partial,
        )
        return out.view([b, s_q, h_q, dv])

    out_flash = flash_mla_tilelang()
    t = do_bench(flash_mla_tilelang)
    out_ref = run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q,
                            cache_seqlens, h_q, h_kv, d, dv, causal, dtype)
    # close_batches = []
    # for i in range(b):
    #     if torch.allclose(out_flash[i], out_ref[i], rtol=0.01, atol=0.01):
    #         close_batches.append(i)
    with open(f"examples/deepseek_mla/example_mla_decode_update_mtp_only_{BLOCK_H}.cu", "w") as f:
        f.write(kernel.get_kernel_source())
    torch.testing.assert_close(out_flash, out_ref, rtol=0.01, atol=0.01)
    print("All close")
    return out_flash, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--h_q', type=int, default=16, help='q heads number')
    parser.add_argument('--h_kv', type=int, default=1, help='kv heads number')
    parser.add_argument('--cache_seqlen', type=int, default=8192, help='kv cache context length')
    parser.add_argument('--d', type=int, default=576, help='query/key head dim, d = dv + dpe')
    parser.add_argument('--dv', type=int, default=512, help='value head dim')
    parser.add_argument('--mtp_size', type=int, default=2, help='Specifies the number of tokens per prediction.')
    args = parser.parse_args()
    b, h_q, h_kv, cache_seqlen, d, dv = args.batch, args.h_q, args.h_kv, args.cache_seqlen, args.d, args.dv
    mtp_size = args.mtp_size

    device = "cuda"
    dtype = torch.float16

    s_q = 1  # for decode, s_q = 1
    block_size = 1
    cache_seqlens = torch.tensor([cache_seqlen + i for i in range(b)],
                                 dtype=torch.int32,
                                 device=device)
    dpe = d - dv
    causal = True

    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / 256) * 256 # 这里对齐到256，感觉和threads还真有点关系，每个thread执行一个元素似乎

    total_flops = s_q * total_seqlens * h_q * (d + dv) * 2

    q = torch.randn(b, s_q, h_q, d, dtype=dtype, device=device)
    block_table = torch.arange(
        b  * max_seqlen_pad // 2, dtype=torch.int32,
        device=device).view(b // 2, max_seqlen_pad)
    # Make adjacent rows identical (rows 0&1, 2&3, etc.)
    block_table = block_table.repeat_interleave(mtp_size, dim=0)

    blocked_k = torch.randn(block_table.numel(), 1, h_kv, d, dtype=dtype, device=device)
    if mtp_size == 2:
        out_flash, latency = run_tilelang_mtp2(mtp_size, q, block_table, blocked_k, max_seqlen_pad, 1, b,
                                            s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype)

    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
