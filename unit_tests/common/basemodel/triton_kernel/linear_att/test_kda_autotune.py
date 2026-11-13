import collections
import functools
import inspect
import json
from itertools import accumulate

import pytest
import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import kda
from lightllm.common.kernel_config import KernelConfigs
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneLevel


KERNELS = [
    ("_chunk_kda_scaled_dot_kkt_sub_inter", "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter"),
    ("_chunk_kda_scaled_dot_kkt_sub_intra", "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra"),
    ("recompute_w_u_fwd", "recompute_w_u_fwd_kernel"),
    ("chunk_gla_fwd_o_gk", "chunk_gla_fwd_kernel_o"),
]


@pytest.fixture(autouse=True)
def autotune_environment(monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.CLOSE_AUTOTUNE)


def wrapper_inputs(kernel, tokens=65):
    q = torch.empty(1, tokens, 2, 128, dtype=torch.bfloat16)
    v = torch.empty(1, tokens, 2, 80, dtype=q.dtype)
    g = torch.empty_like(q, dtype=torch.float32)
    chunk_indices = torch.tensor([[0, 0], *[[1, i] for i in range(triton.cdiv(tokens - 3, 64))]])
    values = dict(
        q=q,
        k=q,
        v=v,
        g=g,
        gk=g,
        beta=torch.empty(1, tokens, 2),
        Akk=torch.empty(1, tokens, 2, 64),
        Aqk=torch.empty(1, tokens, 2, 64),
        R=torch.empty(1, tokens, 2, 64, dtype=q.dtype),
        h=torch.empty(1, len(chunk_indices), 2, 128, 80, dtype=q.dtype),
        o=torch.empty_like(v),
        scale=128 ** -0.5,
        cu_seqlens=torch.tensor([0, 3, tokens]),
        chunk_indices=chunk_indices,
    )
    return {name: values[name] for name in inspect.signature(kernel.fn).parameters if name in values}


@pytest.mark.parametrize("wrapper_name,jit_name", KERNELS)
def test_lightllm_tuning_selects_persists_and_reuses_launch_config(monkeypatch, tmp_path, wrapper_name, jit_name):
    kernel = getattr(kda, wrapper_name)
    assert isinstance(kernel, Autotuner)
    inputs = wrapper_inputs(kernel)
    candidates = [kernel.configs_gen_func()[0], kernel.configs_gen_func()[-1]]
    launches, benchmarks = [], []

    class LaunchRecorder:
        def __getitem__(self, grid):
            def launch(**kwargs):
                launches.append((grid, kwargs))

            return launch

    def bench(*args, run_config, **kwargs):
        benchmarks.append(run_config)
        return 1.0 if run_config == candidates[-1] else 2.0

    monkeypatch.setattr(kda, jit_name, LaunchRecorder())
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr("lightllm.common.kernel_config.get_current_device_name", lambda: "test-device")
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(kernel, "warmuped_configs_set", set())
    monkeypatch.setattr(kernel, "configs_gen_func", lambda: candidates)
    monkeypatch.setattr(kernel, "_bench", bench)

    with Autotuner.autotune_warmup():
        kernel(**inputs)
    assert benchmarks == candidates
    for name, value in candidates[-1].items():
        assert launches[-1][1][name] == value

    cache_file = tmp_path / KernelConfigs.get_config_file_name(kernel._static_key(**inputs))
    assert json.loads(cache_file.read_text()) == {"65": candidates[-1]}

    # A new wrapper invocation must reload the same selected config from disk.
    kernel.cached_configs.clear()
    kernel.fast_match_configs.clear()
    kernel.warmuped_configs_set.clear()
    benchmarks.clear()
    kernel(**inputs)
    assert benchmarks == []
    for name, value in candidates[-1].items():
        assert launches[-1][1][name] == value

    # Packed B=1 still needs different run keys as the token count grows.
    assert kernel._run_key(**inputs) == 65
    assert kernel._run_key(**wrapper_inputs(kernel, tokens=129)) == 129
    assert kernel._static_key(**inputs) == kernel._static_key(**wrapper_inputs(kernel, tokens=129))
    # Request boundaries are runtime data; all calls use the packed path.
    single_request = dict(inputs, cu_seqlens=torch.tensor([0, 65]))
    assert kernel._static_key(**inputs) == kernel._static_key(**single_request)


def recurrent_reference(q, k, v, g, beta, initial_state, sequences):
    q, k, v, g, beta = [x.float().cpu() for x in (q, k, v, g, beta)]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() / q.shape[-1] ** 0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    output = torch.empty_like(v)
    final = initial_state.float().cpu().clone()
    for n, (batch, start, end) in enumerate(sequences):
        state = final[n]
        for t in range(start, end):
            kt = k[batch, t]
            state = state * g[batch, t].exp()[..., None]
            delta = beta[batch, t, :, None] * (v[batch, t] - torch.einsum("hk,hkv->hv", kt, state))
            state = state + kt[..., None] * delta[:, None, :]
            output[batch, t] = torch.einsum("hk,hkv->hv", q[batch, t], state)
        final[n] = state
    return output, final


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("packed", [False, True], ids=["single-sequence", "packed-sequences"])
def test_kda_state_offsets_above_int32_range(packed):
    if torch.cuda.mem_get_info()[0] < 32 * 1024 ** 3:
        pytest.skip("Large-state regression requires 32 GiB of available GPU memory")
    from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    heads, dim, chunk_size = 64, 128, 64
    boundary_chunks = 2 ** 31 // (heads * dim * dim)
    boundary_tokens = boundary_chunks * chunk_size
    tokens = boundary_tokens + 128
    boundaries = [0, boundary_tokens, tokens] if packed else [0, tokens]
    cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    zeros = torch.zeros(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    ones = torch.ones_like(zeros)
    initial = torch.ones(len(boundaries) - 1, heads, dim, dim, device="cuda")
    if packed:
        initial[1].mul_(2)
    state, residuals, final = chunk_gated_delta_rule_fwd_h(
        k=zeros, w=zeros, u=ones, initial_state=initial, output_final_state=True, cu_seqlens=cu_seqlens
    )
    chunk_rows = [0, boundary_chunks - 1, boundary_chunks, boundary_chunks + 1]
    expected_values = torch.tensor([1, 1, 2, 2] if packed else [1, 1, 1, 1], device="cuda")
    expected_state = expected_values[:, None, None, None].expand(4, heads, dim, dim).to(state.dtype)
    torch.testing.assert_close(state[0, chunk_rows], expected_state, atol=0, rtol=0)
    torch.testing.assert_close(final, initial, atol=0, rtol=0)
    rows = [0, boundary_tokens - 1, boundary_tokens, tokens - 1]
    torch.testing.assert_close(residuals[0, rows], ones[0, rows], atol=0, rtol=0)

    output = torch.empty_like(ones)
    gates = torch.zeros_like(zeros, dtype=torch.float32)
    local_weights = torch.zeros(1, tokens, heads, chunk_size, device="cuda", dtype=ones.dtype)
    kda.chunk_gla_fwd_o_gk(
        q=ones,
        v=residuals,
        g=gates,
        Aqk=local_weights,
        h=state,
        o=output,
        scale=1 / dim,
        cu_seqlens=cu_seqlens,
    )
    expected_output = expected_values[:, None, None].expand(4, heads, dim).to(output.dtype)
    torch.testing.assert_close(output[0, rows], expected_output, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("config_index", [None, 0, -1], ids=["default", "first-candidate", "last-candidate"])
@pytest.mark.parametrize(
    "seq_lens,key_dim,value_dim",
    [((65, 65), 64, 96), ((3, 65, 129), 128, 80)],
    ids=["equal-length-packed", "ragged-packed"],
)
def test_kda_configs_match_token_recurrence(monkeypatch, config_index, seq_lens, key_dim, value_dim):
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    if config_index is not None:
        for wrapper_name, _ in KERNELS:
            kernel = getattr(kda, wrapper_name)
            config = kernel.configs_gen_func()[config_index]
            monkeypatch.setattr(kda, wrapper_name, functools.partial(kernel, run_config=config))

    # Include partial final chunks, distinct K/V sizes, and V tiles with masked columns.
    batch, tokens, heads = 1, sum(seq_lens), 2
    boundaries = [0, *accumulate(seq_lens)]
    sequences = [(0, start, end) for start, end in zip(boundaries, boundaries[1:])]
    cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    q, k = [torch.randn(batch, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(batch, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(batch, tokens, heads, key_dim, device="cuda") * 0.1
    beta = torch.rand(batch, tokens, heads, device="cuda")
    initial = torch.randn(len(sequences), heads, key_dim, value_dim, device="cuda") * 0.1
    initial_before = initial.clone()
    expected, expected_final = recurrent_reference(q, k, v, g, beta, initial, sequences)

    actual, final = kda.chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(actual.float().cpu(), expected, atol=4e-3, rtol=3e-2)
    torch.testing.assert_close(final.cpu(), expected_final, atol=8e-3, rtol=3e-2)
    torch.testing.assert_close(initial, initial_before, atol=0, rtol=0)


@pytest.mark.parametrize("fused_gate", [False, True])
@pytest.mark.parametrize("invalid_input", ["batch-dimension", "missing-boundaries"])
def test_kda_requires_packed_inputs(fused_gate, invalid_input):
    batch = 2 if invalid_input == "batch-dimension" else 1
    q = torch.empty(batch, 3, 2, 128)
    inputs = dict(
        q=q,
        k=q,
        v=q,
        beta=torch.empty(batch, 3, 2),
        cu_seqlens=None if invalid_input == "missing-boundaries" else torch.tensor([0, 3, 6]),
    )
    if fused_gate:
        kernel = kda.chunk_kda_with_fused_gate
        inputs.update(raw_g=q, A_log=torch.empty(2), g_bias=None)
    else:
        kernel = kda.chunk_kda
        inputs.update(g=q)
    expected_message = "packed q" if invalid_input == "batch-dimension" else "cu_seqlens is required"
    with pytest.raises(AssertionError, match=expected_message):
        kernel(**inputs)
