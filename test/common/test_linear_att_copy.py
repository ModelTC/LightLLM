import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att_copy import copy_linear_att_state_to_kv_buffer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_linear_att_state_to_cpu_cache_buffer():
    layer_num = 2
    req_slots = 4
    conv_dim = 3
    persisted_width = 2
    widened_width = 4
    ssm_dim = 5
    cache_size = 6

    gpu_conv_state = torch.arange(
        layer_num * req_slots * conv_dim * widened_width,
        dtype=torch.float16,
        device="cuda",
    ).view(layer_num, req_slots, conv_dim, widened_width)
    gpu_ssm_state = torch.arange(
        layer_num * req_slots * ssm_dim,
        dtype=torch.float32,
        device="cuda",
    ).view(layer_num, req_slots, ssm_dim)

    cpu_kv_conv_state = torch.empty(
        cache_size, layer_num, conv_dim, persisted_width, dtype=torch.float16, device="cpu"
    ).fill_(-1)
    cpu_kv_ssm_state = torch.empty(cache_size, layer_num, ssm_dim, dtype=torch.float32, device="cpu").fill_(-1)

    b_req_idx = torch.tensor([1, 3, 0], dtype=torch.int32, device="cuda")
    big_page_buffer_ids = torch.tensor([4, -1, 2], dtype=torch.int32, device="cuda")

    copy_linear_att_state_to_kv_buffer(
        b_req_idx=b_req_idx,
        big_page_buffer_ids=big_page_buffer_ids,
        gpu_conv_state=gpu_conv_state,
        gpu_ssm_state=gpu_ssm_state,
        cpu_kv_conv_state=cpu_kv_conv_state,
        cpu_kv_ssm_state=cpu_kv_ssm_state,
        mtp_step=0,
    )

    expected_conv = torch.empty_like(cpu_kv_conv_state).fill_(-1)
    expected_ssm = torch.empty_like(cpu_kv_ssm_state).fill_(-1)
    expected_conv[4].copy_(gpu_conv_state[:, 1, :, :persisted_width].cpu())
    expected_conv[2].copy_(gpu_conv_state[:, 0, :, :persisted_width].cpu())
    expected_ssm[4].copy_(gpu_ssm_state[:, 1, :].cpu())
    expected_ssm[2].copy_(gpu_ssm_state[:, 0, :].cpu())

    assert torch.equal(cpu_kv_conv_state, expected_conv)
    assert torch.equal(cpu_kv_ssm_state, expected_ssm)
