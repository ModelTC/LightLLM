import pytest
import torch

from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu_all

# =========================================================
# GPU guard
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
# =========================================================
# 工具函数：生成 GPU KV cache
def make_kv(L, T, H, D, device="cuda"):
    x = torch.arange(L * T * H * D, dtype=torch.float32, device=device)
    return x.view(L, T, H, D)


# =========================================================
# Test 1: 基础功能（每 token 对应一个 page slot）
def test_basic_copy():
    L, T, H, D = 2, 8, 4, 16
    B = 1   # token_block_size
    P = 4   # all_page_num

    gpu_kv = make_kv(L, T, H, D)
    cpu_kv = torch.zeros((P, L, B, H, D), dtype=torch.float32, pin_memory=True)

    token_indexes = torch.tensor([1, 3, 5, 7], device="cuda")
    page_indexes  = torch.tensor([0, 1, 2, 3], device="cuda")

    offload_gpu_kv_to_cpu_all(
        token_indexes,
        gpu_kv,
        None,
        cpu_kv,
        None,
        page_indexes,
        tp_index=0,
        tp_world_size=1,
        grid_num=1,
    )

    for i in range(len(token_indexes)):
        token = token_indexes[i].item()
        page  = page_indexes[i].item()

        expected = gpu_kv[:, token, :, :].cpu()  # (L,H,D)
        actual   = cpu_kv[page, :, 0, :, :]      # (L,H,D)

        assert torch.allclose(expected, actual)


# =========================================================
# Test 2: token乱序
def test_random_tokens():
    L, T, H, D = 2, 16, 4, 8
    B = 1
    P = 6

    gpu_kv = make_kv(L, T, H, D)
    cpu_kv = torch.zeros((P, L, B, H, D), pin_memory=True)

    token_indexes = torch.tensor([10, 2, 7, 15, 0, 3], device="cuda")
    page_indexes  = torch.arange(6, device="cuda")

    offload_gpu_kv_to_cpu_all(
        token_indexes,
        gpu_kv,
        None,
        cpu_kv,
        None,
        page_indexes,
        tp_index=0,
        tp_world_size=1,
        grid_num=1,
    )

    for i in range(6):
        t = token_indexes[i].item()
        p = page_indexes[i].item()

        assert torch.allclose(
            cpu_kv[p, :, 0, :, :],
            gpu_kv[:, t, :, :].cpu()
        )


# =========================================================
# Test 3: 带 scale
def test_with_scale():
    L, T, H, D = 2, 8, 4, 16
    B = 1
    P = 3

    gpu_kv = make_kv(L, T, H, D)
    gpu_scale = torch.ones((L, T, H, D // 8), device="cuda") * 2.0

    cpu_kv = torch.zeros((P, L, B, H, D), pin_memory=True)
    cpu_scale = torch.zeros((P, L, B, H, D // 8), pin_memory=True)

    token_indexes = torch.tensor([1, 2, 3], device="cuda")
    page_indexes  = torch.tensor([0, 1, 2], device="cuda")

    offload_gpu_kv_to_cpu_all(
        token_indexes,
        gpu_kv,
        gpu_scale,
        cpu_kv,
        cpu_scale,
        page_indexes,
        tp_index=0,
        tp_world_size=1,
        grid_num=1,
    )

    for i in range(3):
        t = token_indexes[i].item()
        p = page_indexes[i].item()

        # KV
        assert torch.allclose(
            cpu_kv[p, :, 0, :, :],
            gpu_kv[:, t, :, :].cpu()
        )

        # scale
        assert torch.allclose(
            cpu_scale[p, :, 0, :],
            gpu_scale[:, t, :].cpu()
        )


# =========================================================
# Test 4: Tensor Parallel (按 head 切)
def test_tp_split():
    L, T, H, D = 2, 8, 4, 16
    B = 1
    P = 2
    tp_world_size = 2

    gpu_kv = make_kv(L, T, H, D)

    cpu_kv = torch.zeros((P, L, B, H, D), pin_memory=True)

    token_indexes = torch.tensor([1, 2], device="cuda")
    page_indexes  = torch.tensor([0, 1], device="cuda")

    offload_gpu_kv_to_cpu_all(
        token_indexes, gpu_kv, None,
        cpu_kv, None,
        page_indexes,
        tp_index=0,
        tp_world_size=tp_world_size,
        grid_num=1,
    )

    offload_gpu_kv_to_cpu_all(
        token_indexes, gpu_kv, None,
        cpu_kv, None,
        page_indexes,
        tp_index=1,
        tp_world_size=tp_world_size,
        grid_num=1,
    )

    split = H // tp_world_size

    for i in range(2):
        t = token_indexes[i].item()
        p = page_indexes[i].item()

        assert torch.allclose(
            cpu_kv[p, :, 0, :split, :],
            gpu_kv[:, t, :split, :].cpu()
        )

        assert torch.allclose(
            cpu_kv[p, :, 0, split:, :],
            gpu_kv[:, t, split:, :].cpu()
        )


# =========================================================
# Test 5: 空输入
def test_empty():
    L, T, H, D = 2, 8, 4, 16
    B = 1
    P = 2

    gpu_kv = make_kv(L, T, H, D)
    cpu_kv = torch.zeros((P, L, B, H, D), pin_memory=True)

    token_indexes = torch.tensor([], dtype=torch.long, device="cuda")
    page_indexes  = torch.tensor([], dtype=torch.long, device="cuda")

    offload_gpu_kv_to_cpu_all(
        token_indexes,
        gpu_kv,
        None,
        cpu_kv,
        None,
        page_indexes,
        tp_index=0,
        tp_world_size=1,
        grid_num=1,
    )

    assert torch.all(cpu_kv == 0)

if __name__ == "__main__":
    pytest.main()