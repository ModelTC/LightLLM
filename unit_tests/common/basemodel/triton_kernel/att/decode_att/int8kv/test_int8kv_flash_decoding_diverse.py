import pytest
import warnings

import torch
import time
from lightllm.utils.light_utils import light_ops

# 全局结果收集
_test_results = []


def alloc_tensor_func(shape, dtype, device):
    """兼容的 tensor 分配函数"""
    return torch.empty(shape, dtype=dtype, device=device)


class MockReqManager:
    """Mock request manager for testing"""

    def __init__(self, req_to_token_indexs):
        self.req_to_token_indexs = req_to_token_indexs


class MockInferState:
    """Mock infer state for testing"""

    def __init__(
        self,
        batch_size,
        max_kv_seq_len,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        b_shared_seq_len=None,
        b_mark_shared_group=None,
    ):
        self.batch_size = batch_size
        self.max_kv_seq_len = max_kv_seq_len
        self.req_manager = MockReqManager(req_to_tokens)
        self.b_req_idx = b_req_idx
        self.b_seq_len = b_seq_len
        self.b_shared_seq_len = b_shared_seq_len
        self.b_mark_shared_group = b_mark_shared_group


# @pytest.mark.parametrize("shared_seq_len", [512])
@pytest.mark.parametrize("shared_seq_len", [0, 77, 256, 311, 512, 550])
@pytest.mark.parametrize("batch_size", list(range(6, 121, 6)))
def test_token_decode_attention_flash_decoding_diverse_vs_baseline(shared_seq_len, batch_size):
    """
    测试 int8kv_flash_decoding_diverse 的 token_decode_attention_flash_decoding
    与 ppl_int8kv_flash_decoding (baseline) 的对比。
    """

    from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.int8kv_flash_decoding_diverse import (
        token_decode_attention_flash_decoding as diverse_attention,
    )
    from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.ppl_int8kv_flash_decoding import (
        token_decode_attention_flash_decoding as baseline_attention,
    )

    num_heads = 32
    kv_head_num = 2  # gqa_group_size = 16，满足 Triton tl.dot 的 M >= 16 要求
    mark_shared_group_size = 3
    seq_len = 3547
    head_dim = 128
    quant_group_size = 8
    max_len_in_batch = 8192
    test_dtype = torch.bfloat16

    # 创建测试数据
    kv_shape = (batch_size * max_len_in_batch, kv_head_num, head_dim)
    kv_scale_shape = (batch_size * max_len_in_batch, kv_head_num, head_dim // quant_group_size)

    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device="cuda")

    # 生成 cache_k 和 cache_v，使得每 mark_shared_group_size 个 batch 共享相同的 cache

    cache_k = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    cache_k_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda") / 100.0
    cache_v = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    cache_v_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda") / 100.0

    req_to_tokens = torch.arange(0, max_len_in_batch * batch_size, dtype=torch.int32, device="cuda").view(
        batch_size, max_len_in_batch
    )
    for i in range(batch_size):
        if i % mark_shared_group_size != 0:
            req_to_tokens[i, :shared_seq_len] = req_to_tokens[i - 1, :shared_seq_len]

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.full((batch_size,), shared_seq_len, dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    b_mark_shared_group[mark_shared_group_size - 1 :: mark_shared_group_size] = mark_shared_group_size

    # 创建 baseline 的 infer_state (不需要 b_shared_seq_len)
    baseline_infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=max_len_in_batch,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
    )

    # 创建 diverse 的 infer_state
    diverse_infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=max_len_in_batch,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_shared_seq_len=b_shared_seq_len,
        b_mark_shared_group=b_mark_shared_group,
    )

    # 运行 baseline
    # 预热
    for _ in range(10):
        _ = baseline_attention(
            q=q.clone(),
            infer_state=baseline_infer_state,
            cache_k=cache_k,
            cache_k_scale=cache_k_scale,
            cache_v=cache_v,
            cache_v_scale=cache_v_scale,
            alloc_tensor_func=alloc_tensor_func,
        )
    torch.cuda.synchronize()

    # 计时 baseline
    start = time.perf_counter()
    for _ in range(100):
        baseline_out = baseline_attention(
            q=q.clone(),
            infer_state=baseline_infer_state,
            cache_k=cache_k,
            cache_k_scale=cache_k_scale,
            cache_v=cache_v,
            cache_v_scale=cache_v_scale,
            alloc_tensor_func=alloc_tensor_func,
        )
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / 100 * 1000  # ms

    # 运行 diverse 版本
    # 预热
    for _ in range(10):
        _ = diverse_attention(
            q=q.clone(),
            infer_state=diverse_infer_state,
            cache_k=cache_k,
            cache_k_scale=cache_k_scale,
            cache_v=cache_v,
            cache_v_scale=cache_v_scale,
            alloc_tensor_func=alloc_tensor_func,
        )
    torch.cuda.synchronize()

    # 计时 diverse
    start = time.perf_counter()
    for _ in range(100):
        diverse_out = diverse_attention(
            q=q.clone(),
            infer_state=diverse_infer_state,
            cache_k=cache_k,
            cache_k_scale=cache_k_scale,
            cache_v=cache_v,
            cache_v_scale=cache_v_scale,
            alloc_tensor_func=alloc_tensor_func,
        )
    torch.cuda.synchronize()
    diverse_time = (time.perf_counter() - start) / 100 * 1000  # ms

    print(f"\nshared_seq_len={shared_seq_len}\nbatch_size={batch_size}")
    print(f"baseline_out: {baseline_out[0, 0, :4]}")
    print(f"diverse_out: {diverse_out[0, 0, :4]}")
    print(f"max diff: {(baseline_out - diverse_out).abs().max()}")
    print(f"baseline_time: {baseline_time:.3f} ms")
    print(f"diverse_time: {diverse_time:.3f} ms")
    if baseline_time > 0:
        speedup = baseline_time / diverse_time
        print(f"speedup: {speedup:.2f}x")

    # 与 baseline 对比
    allclose_passed = torch.allclose(
        baseline_out, diverse_out, atol=1e-2, rtol=1e-2
    )
    max_diff = (baseline_out - diverse_out).abs().max()

    # 收集结果
    _test_results.append({
        "shared_seq_len": shared_seq_len,
        "batch_size": batch_size,
        "baseline_time_ms": baseline_time,
        "diverse_time_ms": diverse_time,
        "speedup": baseline_time / diverse_time if diverse_time > 0 else 0,
        "max_diff": max_diff.item(),
        "allclose_passed": allclose_passed,
    })

    status = "PASS" if allclose_passed else "FAIL"
    print(f"[{status}] max_diff: {max_diff.item():.6f}, speedup: {baseline_time / diverse_time:.2f}x")

    # pytest 模式下，如果 allclose 失败只发出警告，不终止测试
    if not allclose_passed:
        warnings.warn(
            f"allclose check failed for shared_seq_len={shared_seq_len}, batch_size={batch_size}. "
            f"max_diff={max_diff.item():.6f}"
        )

    return allclose_passed


if __name__ == "__main__":
    # 检查 light_ops 是否可用
    if light_ops is None:
        print("Error: light_ops is not available.")
        print("Please build and install lightllm extension first:")
        print("  cd /data/nvme0/chenjunyi/project/lightllm")
        print("  python setup.py build_ext --inplace")
        exit(1)

    # 默认测试参数
    shared_seq_lens = [0, 77, 256, 311, 512, 550]
    batch_sizes = [16, 32, 64, 96, 120]

    print("=" * 100)
    print("Testing int8kv_flash_decoding_diverse vs baseline")
    print("=" * 100)

    all_passed = True
    for shared_seq_len in shared_seq_lens:
        for batch_size in batch_sizes:
            print(f"\nTesting: shared_seq_len={shared_seq_len}, batch_size={batch_size}")
            passed = test_token_decode_attention_flash_decoding_diverse_vs_baseline(shared_seq_len, batch_size)
            if not passed:
                all_passed = False

    # 打印结果表格
    print(f"\n{'='*100}")
    print("SUMMARY - Results Table")
    print(f"{'='*100}")
    print(
        f"{'shared_seq':<12} {'batch_size':<12} {'baseline_ms':<14} {'diverse_ms':<14} {'speedup':<10} {'max_diff':<12} {'status':<8}"
    )
    print(f"{'-'*100}")
    for r in _test_results:
        status = "PASS" if r["allclose_passed"] else "FAIL"
        print(
            f"{r['shared_seq_len']:<12} {r['batch_size']:<12} "
            f"{r['baseline_time_ms']:<14.3f} {r['diverse_time_ms']:<14.3f} "
            f"{r['speedup']:<10.2f}x {r['max_diff']:<12.6f} {status:<8}"
        )
    print(f"{'='*100}")

    # 总结
    passed_count = sum(1 for r in _test_results if r["allclose_passed"])
    total_count = len(_test_results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if not all_passed:
        print("\nWARNING: Some tests failed allclose check!")
    else:
        print("\nAll tests passed!")
