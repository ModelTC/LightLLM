import pytest
import torch
import triton

from lightllm.models.deepseek3_2.triton_kernel.fp8_mqa_logits import fp8_paged_mqa_logits


def _require_deep_gemm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        pytest.skip("deep_gemm is not available")
    import deep_gemm

    return deep_gemm


def _build_case(batch_size, seq_len, mtp_step, head_num=32, head_dim=128, pool_size=None, seed=42):
    """Build decode-shaped inputs mirroring NsaInfer._get_indices + gen_nsa_ks_ke."""
    torch.manual_seed(seed)
    device = "cuda"
    req_num = batch_size
    q_num = req_num * (mtp_step + 1)

    b_seq_len_req = torch.randint(max(1, seq_len // 2), seq_len + 1, (req_num,), device=device, dtype=torch.int32)
    b_seq_len_req[0] = seq_len  # keep max_kv_seq_len == seq_len
    total_tokens = int(b_seq_len_req.sum().item())
    pool_size = pool_size or (total_tokens + 1024)

    # token-granular pool with shuffled slot assignment
    perm = torch.randperm(pool_size, device=device, dtype=torch.int32)
    indexer_k_buffer = torch.empty((pool_size, 1, 132), dtype=torch.uint8, device=device)
    k_bf16 = torch.randn(pool_size, head_dim, device=device, dtype=torch.float32)
    k_scale = (torch.rand(pool_size, device=device, dtype=torch.float32) + 0.5) / 100.0
    indexer_k_buffer[:, 0, 0:128] = k_bf16.to(torch.float8_e4m3fn).view(torch.uint8)
    indexer_k_buffer[:, 0, 128:132] = k_scale.view(-1, 1).view(torch.uint8)

    # ragged layout + per-row key windows (decode: q rows of one req share its K, ke grows by 1)
    ragged_mem_index = torch.empty(total_tokens, dtype=torch.int32, device=device)
    ks = torch.empty(q_num, dtype=torch.int32, device=device)
    ke = torch.empty(q_num, dtype=torch.int32, device=device)
    pre_sum = 0
    slot_cursor = 0
    req_to_slots = []
    for r in range(req_num):
        L = int(b_seq_len_req[r].item())
        slots = perm[slot_cursor : slot_cursor + L]
        slot_cursor += L
        req_to_slots.append(slots)
        ragged_mem_index[pre_sum : pre_sum + L] = slots
        for j in range(mtp_step + 1):
            row = r * (mtp_step + 1) + j
            ks[row] = pre_sum
            ke[row] = pre_sum + (L - mtp_step) + j
        pre_sum += L

    q_fp8 = torch.randn(q_num, head_num, head_dim, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    weights = torch.randn(q_num, head_num, device=device, dtype=torch.float32) / head_num

    max_kv_seq_len = seq_len
    out_col_num = q_num * max_kv_seq_len // (mtp_step + 1)  # K rows for the extract path
    return {
        "q_fp8": q_fp8,
        "indexer_k_buffer": indexer_k_buffer,
        "weights": weights,
        "ragged_mem_index": ragged_mem_index,
        "ks": ks,
        "ke": ke,
        "out_col_num": out_col_num,
        "max_kv_seq_len": max_kv_seq_len,
        "b_seq_len_req": b_seq_len_req,
        "mtp_step": mtp_step,
        "req_num": req_num,
        "total_tokens": total_tokens,
    }


def _ref_logits_deep_gemm(deep_gemm, case):
    """Reference: today's extract + deep_gemm.fp8_mqa_logits path (extraction done in torch)."""
    ragged = case["ragged_mem_index"][: case["total_tokens"]].long()
    buf = case["indexer_k_buffer"]
    k_fp8 = buf[:, 0, 0:128].view(torch.float8_e4m3fn)[ragged].contiguous()
    k_scale = buf[:, 0, 128:132].view(torch.float32)[ragged].reshape(-1).contiguous()
    pad = case["out_col_num"] - k_fp8.shape[0]
    assert pad >= 0
    if pad:
        k_fp8 = torch.cat([k_fp8, torch.zeros(pad, 128, dtype=k_fp8.dtype, device=k_fp8.device)])
        k_scale = torch.cat([k_scale, torch.zeros(pad, dtype=k_scale.dtype, device=k_scale.device)])
    return deep_gemm.fp8_mqa_logits(
        case["q_fp8"],
        (k_fp8, k_scale),
        case["weights"],
        case["ks"],
        case["ke"],
        clean_logits=False,
        max_seqlen_k=case["max_kv_seq_len"],
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,mtp_step",
    [
        (1, 128, 0),
        (8, 1024, 0),
        (32, 4096, 0),
        (8, 1024, 1),
        (16, 2048, 3),
        (64, 512, 0),
    ],
)
def test_fp8_paged_mqa_logits_matches_deep_gemm(batch_size, seq_len, mtp_step):
    deep_gemm = _require_deep_gemm()
    case = _build_case(batch_size, seq_len, mtp_step)

    ref = _ref_logits_deep_gemm(deep_gemm, case)
    out = fp8_paged_mqa_logits(
        q_fp8=case["q_fp8"],
        indexer_k_buffer=case["indexer_k_buffer"],
        weights=case["weights"],
        ragged_mem_index=case["ragged_mem_index"],
        cu_seqlen_ks=case["ks"],
        cu_seqlen_ke=case["ke"],
        max_kv_seq_len=case["max_kv_seq_len"],
    )

    assert out.shape == ref.shape
    # rows are compacted: row m valid at columns [0, ke-ks)
    for row in range(out.shape[0]):
        length = int(case["ke"][row]) - int(case["ks"][row])
        torch.testing.assert_close(out[row, :length], ref[row, :length], rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    # quick perf A/B vs today's real path (extract_indexer_ks + deep_gemm)
    from lightllm.models.deepseek3_2.triton_kernel.extract_indexer_ks import extract_indexer_ks

    deep_gemm = _require_deep_gemm()

    def _bench(fn):
        return triton.testing.do_bench_cudagraph(fn, return_mode="median")

    print(f"{'batch':>5} {'seq':>6} {'mtp':>3} | {'old (extract+dg)':>16} {'new (paged)':>12} {'speedup':>8}")
    for batch, seq, mtp in [
        (1, 4096, 0),
        (8, 4096, 0),
        (32, 4096, 0),
        (8, 16384, 0),
        (32, 16384, 0),
        (64, 8192, 0),
        (128, 2048, 0),
        (8, 60000, 0),
        (32, 60000, 0),
        (8, 16384, 1),
    ]:
        case = _build_case(batch, seq, mtp)
        q_num = case["q_fp8"].shape[0]

        # emulate the old path's inputs (b_seq_len/b_req_idx expanded per mtp row)
        b_seq_len = case["b_seq_len_req"].repeat_interleave(mtp + 1).to(torch.int32)
        b_req_idx = torch.arange(case["req_num"], device="cuda", dtype=torch.int32).repeat_interleave(mtp + 1)
        req_to_token = torch.zeros((case["req_num"], seq), dtype=torch.int32, device="cuda")
        pre = 0
        for r in range(case["req_num"]):
            L = int(case["b_seq_len_req"][r])
            req_to_token[r, :L] = case["ragged_mem_index"][pre : pre + L]
            pre += L

        def run_old():
            k_fp8_, k_scale_ = extract_indexer_ks(
                I_buffer=case["indexer_k_buffer"],
                b_seq_len=b_seq_len,
                b_req_idx=b_req_idx,
                req_to_token_indexs=req_to_token,
                out_token_num=q_num * seq,
                max_kv_seq_len=seq,
                mtp_step=mtp,
            )
            return deep_gemm.fp8_mqa_logits(
                case["q_fp8"],
                (k_fp8_, k_scale_),
                case["weights"],
                case["ks"],
                case["ke"],
                clean_logits=False,
                max_seqlen_k=seq,
            )

        def run_new():
            return fp8_paged_mqa_logits(
                q_fp8=case["q_fp8"],
                indexer_k_buffer=case["indexer_k_buffer"],
                weights=case["weights"],
                ragged_mem_index=case["ragged_mem_index"],
                cu_seqlen_ks=case["ks"],
                cu_seqlen_ke=case["ke"],
                max_kv_seq_len=case["max_kv_seq_len"],
            )

        t_old = _bench(run_old)
        t_new = _bench(run_new)
        print(f"{batch:>5} {seq:>6} {mtp:>3} | {t_old*1000:>13.1f}us {t_new*1000:>9.1f}us {t_old/t_new:>7.2f}x")
