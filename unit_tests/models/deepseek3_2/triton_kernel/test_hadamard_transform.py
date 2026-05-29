import pytest
import torch
import triton

from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import hadamard_transform


TP = 8
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
TP_INDEX_N_HEADS = INDEX_N_HEADS // TP
SCALE = INDEX_HEAD_DIM ** -0.5


def _get_sgl_kernel_hadamard_transform():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for hadamard_transform comparison")
    try:
        from sgl_kernel import hadamard_transform as sgl_hadamard_transform
    except ImportError:
        pytest.skip("sgl_kernel.hadamard_transform is not available")
    return sgl_hadamard_transform


def _bench(fn, x):
    ms = triton.testing.do_bench_cudagraph(lambda: fn(x, scale=SCALE), return_mode="median")
    return ms, fn(x, scale=SCALE)


@pytest.mark.parametrize("tokens", [1, 16, 128, 512, 1024, 2048, 4096, 8192, 16384])
def test_hadamard_transform_matches_sgl_kernel_deepseek_v32_shapes(tokens):
    sgl_hadamard_transform = _get_sgl_kernel_hadamard_transform()

    q = torch.randn(tokens, TP_INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(tokens, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    q_expected = sgl_hadamard_transform(q, scale=SCALE)
    q_actual = hadamard_transform(q, scale=SCALE)
    k_expected = sgl_hadamard_transform(k, scale=SCALE)
    k_actual = hadamard_transform(k, scale=SCALE)
    torch.cuda.synchronize()

    assert torch.equal(q_actual, q_expected)
    assert torch.equal(k_actual, k_expected)


def test_hadamard_transform_perf_report_deepseek_v32_shapes():
    sgl_hadamard_transform = _get_sgl_kernel_hadamard_transform()

    print(
        "\nDeepSeek-V3.2 per-rank shapes with tp=8:"
        "\n  q: [tokens, 8, 128]"
        "\n  k: [tokens, 128]"
        "\n\ntokens | q_diff | k_diff | sgl_q ms | tri_q ms | sgl_k ms | tri_k ms | tri(q+k) ms | slowdown q+k"
    )

    for tokens in [1, 16, 128, 512, 1024, 2048, 4096, 8192, 16384]:
        q = torch.randn(tokens, TP_INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(tokens, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")

        q_expected = sgl_hadamard_transform(q, scale=SCALE)
        q_actual = hadamard_transform(q, scale=SCALE)
        k_expected = sgl_hadamard_transform(k, scale=SCALE)
        k_actual = hadamard_transform(k, scale=SCALE)
        torch.cuda.synchronize()

        q_diff = (q_expected.float() - q_actual.float()).abs().max().item()
        k_diff = (k_expected.float() - k_actual.float()).abs().max().item()
        sgl_q_ms, _ = _bench(sgl_hadamard_transform, q)
        tri_q_ms, _ = _bench(hadamard_transform, q)
        sgl_k_ms, _ = _bench(sgl_hadamard_transform, k)
        tri_k_ms, _ = _bench(hadamard_transform, k)
        sgl_sum_ms = sgl_q_ms + sgl_k_ms
        tri_sum_ms = tri_q_ms + tri_k_ms

        print(
            f"{tokens:6d} | {q_diff:6.1g} | {k_diff:6.1g} | "
            f"{sgl_q_ms:8.4f} | {tri_q_ms:8.4f} | {sgl_k_ms:8.4f} | {tri_k_ms:8.4f} | "
            f"{tri_sum_ms:11.4f} | {tri_sum_ms / sgl_sum_ms:10.2f}x"
        )

        assert q_diff == 0
        assert k_diff == 0
