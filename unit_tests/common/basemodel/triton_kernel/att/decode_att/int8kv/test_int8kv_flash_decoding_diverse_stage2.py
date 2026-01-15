import pytest
import torch
from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.int8kv_flash_decoding_diverse_stage2 import (
    flash_decode_stage2,
)


def create_tensors(
    shared_seq_len,
    batch_size=4,
    seq_len=256,
    max_len_in_batch=8192,
    max_batch_group_size=4,
    kv_len=None,
    req_to_tokens_len=None,
):
    num_heads = 32
    kv_head_num = 8
    head_dim = 128
    block_seq = 256
    quant_group_size = 8

    test_dtype = torch.bfloat16

    kv_len = max_len_in_batch if kv_len is None else kv_len
    req_to_tokens_len = max_len_in_batch if req_to_tokens_len is None else req_to_tokens_len

    kv_shape = (batch_size * kv_len, kv_head_num, head_dim)
    kv_scale_shape = (batch_size * kv_len, kv_head_num, head_dim // quant_group_size)

    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device="cuda")
    k = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    k_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda")
    v = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    v_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda")
    Req_to_tokens = torch.arange(0, req_to_tokens_len * batch_size, dtype=torch.int32, device="cuda").view(
        batch_size, req_to_tokens_len
    )
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
        "max_batch_group_size": max_batch_group_size,
        "head_dim": head_dim,
    }


@pytest.mark.parametrize("shared_seq_len", [0, 47, 77, 128, 200, 255])
def test_flash_decode_stage2_execution(shared_seq_len):
    setup_tensors = create_tensors(shared_seq_len)

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
    seq_block_idx = (setup_tensors["b_shared_seq_len"][0].item() + setup_tensors["block_seq"] - 1) // setup_tensors[
        "block_seq"
    ]
    mid_out = setup_tensors["mid_out"][:, :, seq_block_idx:, :]
    mid_out_logsumexp = setup_tensors["mid_out_logsumexp"][:, :, seq_block_idx:]

    q = setup_tensors["q"]
    k = setup_tensors["k"]
    v = setup_tensors["v"]
    true_mid_out = torch.zeros_like(mid_out)
    true_mid_out_logsumexp = torch.zeros_like(mid_out_logsumexp)
    new_q = q
    new_k = k.to(q.dtype)
    new_v = v.to(q.dtype)

    b_seq_len = setup_tensors["b_seq_len"] - setup_tensors["b_shared_seq_len"]
    req_to_tokens = setup_tensors["Req_to_tokens"][:, setup_tensors["b_shared_seq_len"][0].item() :]

    from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_stage1 import (
        flash_decode_stage1 as gqa_flash_decode_stage1,
    )

    gqa_flash_decode_stage1(
        q=new_q,
        k=new_k,
        v=new_v,
        Req_to_tokens=req_to_tokens,
        B_req_idx=setup_tensors["B_req_idx"],
        B_Seqlen=b_seq_len,
        max_len_in_batch=setup_tensors["max_len_in_batch"],
        mid_out=true_mid_out,
        mid_out_logsumexp=true_mid_out_logsumexp,
        block_seq=setup_tensors["block_seq"],
    )
    print(f"\nshared_seq_len={shared_seq_len}")
    print(f"mid_out: {mid_out[0:4, 0, 0, 0]}")
    print(f"true_mid_out: {true_mid_out[0:4, 0, 0, 0]}")
    abs_diff = (mid_out - true_mid_out).abs()
    max_diff = abs_diff.max()
    max_diff_idx = abs_diff.argmax()
    max_diff_idx_unraveled = torch.unravel_index(max_diff_idx, abs_diff.shape)
    mid_out_value = mid_out[max_diff_idx_unraveled]
    true_mid_out_value = true_mid_out[max_diff_idx_unraveled]
    print(f"max abs diff: {max_diff}, mid_out value: {mid_out_value}, " f"true_mid_out value: {true_mid_out_value}")

    assert torch.allclose(
        mid_out[0:4, 0, 0, 0], true_mid_out[0:4, 0, 0, 0], atol=1e-2
    ), f"Mid output does not match expected values for shared_seq_len={shared_seq_len}"
    assert torch.allclose(
        mid_out_logsumexp, true_mid_out_logsumexp, atol=1e-2
    ), f"LogSumExp output does not match expected values for shared_seq_len={shared_seq_len}"


if __name__ == "__main__":
    import importlib
    import triton
    from lightllm.utils.light_utils import light_ops

    batch_sizes = [8, 16, 32, 64]
    seq_lens = [32, 64, 128, 256]

    results = []
    for batch in batch_sizes:
        for seq in seq_lens:
            # Clear GPU cache to reduce CUDA Graph capture failures.
            torch.cuda.empty_cache()

            setup_tensors = create_tensors(
                shared_seq_len=0,
                batch_size=batch,
                seq_len=seq,
                max_len_in_batch=8192,
                kv_len=seq,
                req_to_tokens_len=seq,
            )

            # Outputs for CUDA implementation
            mid_out_cuda = setup_tensors["mid_out"].clone()
            mid_out_logsumexp_cuda = setup_tensors["mid_out_logsumexp"].clone()

            # Outputs for Triton implementation
            mid_out_triton = setup_tensors["mid_out"].clone()
            mid_out_logsumexp_triton = setup_tensors["mid_out_logsumexp"].clone()

            # Run CUDA to get reference
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

            # Run Triton
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

            # Compare results
            diff_mid_out = torch.abs(mid_out_cuda - mid_out_triton)
            diff_logsumexp = torch.abs(mid_out_logsumexp_cuda - mid_out_logsumexp_triton)
            max_diff_out = diff_mid_out.max().item()
            max_diff_logsumexp = diff_logsumexp.max().item()
            mean_diff_out = diff_mid_out.mean().item()
            mean_diff_logsumexp = diff_logsumexp.mean().item()

            cos_sim_out = torch.nn.functional.cosine_similarity(
                mid_out_cuda.flatten(), mid_out_triton.flatten(), dim=0
            ).item()
            cos_sim_logsumexp = torch.nn.functional.cosine_similarity(
                mid_out_logsumexp_cuda.flatten(), mid_out_logsumexp_triton.flatten(), dim=0
            ).item()

            print(f"\n[batch={batch}, seq={seq}] Consistency check:")
            print("  mid_out:")
            print(f"    max_diff: {max_diff_out:.6f}, mean_diff: {mean_diff_out:.6f}, cosine_sim: {cos_sim_out:.8f}")
            print("  logsumexp:")
            print(
                f"    max_diff: {max_diff_logsumexp:.6f}, "
                f"mean_diff: {mean_diff_logsumexp:.6f}, "
                f"cosine_sim: {cos_sim_logsumexp:.8f}"
            )

            # Performance
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

            del setup_tensors

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
