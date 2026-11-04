# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import triton

from lightllm.common.basemodel.triton_kernel.mhc import head, post, pre_norm
from lightllm.common.basemodel.triton_kernel.mhc.configs import post_projection_configs
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneLevel
from unit_tests.common.basemodel.triton_kernel.mhc.test_mhc import _hc_post_reference, _hc_pre_reference
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward


@pytest.fixture(autouse=True)
def tuning_environment(monkeypatch):
    torch.manual_seed(1525)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.CLOSE_AUTOTUNE)
    if torch.cuda.is_available():
        triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens,hidden", [(3, 4096), (16, 6144)])
def test_all_pre_and_post_projection_configs_match_reference(tokens, hidden):
    x = torch.randn(tokens, 4 * hidden, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 4 * hidden, device="cuda") * 0.005
    scale, base = torch.randn(3, device="cuda"), torch.randn(24, device="cuda")
    norm = torch.randn(hidden, device="cuda", dtype=x.dtype)
    expected = _hc_pre_reference(x, fn, scale, base, 4, 1e-6, 1e-6, 20)

    def check(actual, reference):
        torch.testing.assert_close(actual[0], rmsnorm_forward(reference[0], norm, 1e-6), atol=0.04, rtol=0.03)
        for a, b in zip(actual[1:], reference[1:]):
            torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)

    for config in pre_norm._hc_prenorm_projection.configs_gen_func():
        gp, sp = pre_norm._hc_prenorm_projection(x, fn, run_config=config)
        check(pre_norm._finish_pre_norm(x, gp, sp, scale, base, norm, 4, 1e-6, 1e-6, 1e-6, 20, 2.0), expected)
    gp, sp = pre_norm._hc_prenorm_projection(x, fn)
    for config in pre_norm._finish_pre_norm.configs_gen_func():
        check(
            pre_norm._finish_pre_norm(x, gp, sp, scale, base, norm, 4, 1e-6, 1e-6, 1e-6, 20, 2.0, run_config=config),
            expected,
        )

    layer_output = torch.randn(tokens, hidden, device="cuda", dtype=x.dtype)
    residual_mix = torch.randn(tokens, 4, 4, device="cuda").softmax(-1)
    post_mix = torch.rand(tokens, 4, device="cuda") * 2
    expected_residual = _hc_post_reference(layer_output, x, residual_mix, post_mix, 4)
    expected = _hc_pre_reference(expected_residual, fn, scale, base, 4, 1e-6, 1e-6, 20)
    before = x.clone()
    for config in post_projection_configs():
        updated, gp, sp = pre_norm._hc_post_prenorm_projection(
            layer_output, x, residual_mix, post_mix, fn, run_config=config
        )
        torch.testing.assert_close(updated, expected_residual, atol=0.04, rtol=0.02)
        check(pre_norm._finish_pre_norm(updated, gp, sp, scale, base, norm, 4, 1e-6, 1e-6, 1e-6, 20, 2.0), expected)
    torch.testing.assert_close(x, before, atol=0, rtol=0)
    for config in pre_norm._hc_post_pre_norm_decode.configs_gen_func():
        actual = pre_norm._hc_post_pre_norm_decode(
            layer_output,
            x,
            residual_mix,
            post_mix,
            fn,
            scale,
            base,
            norm,
            4,
            1e-6,
            1e-6,
            1e-6,
            20,
            2.0,
            run_config=config,
        )
        torch.testing.assert_close(actual[1], expected_residual, atol=0.04, rtol=0.02)
        check((actual[0], actual[2], actual[3]), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hidden", [4096, 6144])
def test_all_post_and_head_configs_match_reference(hidden):
    tokens = 19
    x = torch.randn(tokens, 4 * hidden, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(tokens, hidden, device="cuda", dtype=x.dtype)
    mix, post_mix = torch.randn(tokens, 4, 4, device="cuda").softmax(-1), torch.rand(tokens, 4, device="cuda") * 2
    expected_post = _hc_post_reference(y, x, mix, post_mix, 4)
    for config in post.hc_post.configs_gen_func():
        actual = post.hc_post(y, x, mix, post_mix, 4, run_config=config)
        torch.testing.assert_close(actual, expected_post, atol=0.04, rtol=0.02)
    fn, scale, base = (
        torch.randn(4, 4 * hidden, device="cuda") * 0.005,
        torch.randn(1, device="cuda"),
        torch.randn(4, device="cuda"),
    )
    inv_rms = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    gates = torch.sigmoid((x.float() @ fn.T) * inv_rms * scale + base) + 1e-6
    expected_head = (x.view(tokens, 4, hidden).float() * gates.unsqueeze(-1)).sum(1).bfloat16()
    for config in head.hc_head.configs_gen_func():
        actual = head.hc_head(x, fn, scale, base, 4, 1e-6, 1e-6, run_config=config)
        torch.testing.assert_close(actual, expected_head, atol=0.04, rtol=0.02)
