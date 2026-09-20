import collections
import inspect
import json
import math

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.linear_att import (
    mtp_fused_recurrent as kernel_module,
)
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import (
    Autotuner,
    AutotuneKernelType,
    AutotuneLevel,
)


@pytest.fixture(autouse=True)
def autotune_environment(monkeypatch, tmp_path):
    torch.manual_seed(42)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(
        autotuner_module,
        "get_triton_autotune_level",
        lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE,
    )
    kernel = kernel_module.mtp_fused_recurrent_gated_delta_rule
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(kernel, "warmuped_configs_set", set())


def make_inputs(
    lengths=(3, 2, 0),
    mtp_size=3,
    device="cpu",
    head_k_dim=64,
    head_v_dim=64,
    heads=2,
    value_heads=4,
    dtype=torch.bfloat16,
    state_dtype=torch.float32,
    separate_write=False,
    padding=0,
):
    num_seqs = len(lengths)
    num_tokens = sum(lengths) + padding
    # 模拟生产环境从 mixed QKV 和 gate 投影切分的非连续 token 视图。
    mixed = torch.randn(
        num_tokens,
        heads * head_k_dim * 2 + value_heads * head_v_dim,
        dtype=dtype,
        device=device,
    )
    q, k, v = mixed.split([heads * head_k_dim, heads * head_k_dim, value_heads * head_v_dim], dim=-1)
    gates = torch.randn(num_tokens, value_heads * 3, dtype=dtype, device=device)
    indices = torch.arange(1, num_seqs * mtp_size + 1, dtype=torch.int32, device=device).view(num_seqs, mtp_size)
    return dict(
        q=q.view(1, num_tokens, heads, head_k_dim),
        k=k.view(1, num_tokens, heads, head_k_dim),
        v=v.view(1, num_tokens, value_heads, head_v_dim),
        initial_state=torch.randn(
            2 * num_seqs * mtp_size + 5,
            value_heads,
            head_k_dim,
            head_v_dim,
            dtype=state_dtype,
            device=device,
        )
        * 0.25,
        cu_seqlens=torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int64,
            device=device,
        ),
        ssm_state_indices=indices,
        ssm_state_write_indices=(indices + num_seqs * mtp_size if separate_write else indices),
        num_accepted_tokens=torch.tensor(
            [i % mtp_size + 1 for i in range(num_seqs)],
            dtype=torch.int32,
            device=device,
        ),
        A_log=torch.randn(value_heads, dtype=torch.float32, device=device) * 0.1,
        dt_bias=torch.randn(value_heads, dtype=torch.float32, device=device) * 0.1,
        a_raw=gates[:, :value_heads],
        b_raw=gates[:, value_heads : 2 * value_heads],
    )


def rebuild(inputs):
    args, kwargs = kernel_module.rebuild_inputs(**inputs)
    return inspect.signature(kernel_module.mtp_fused_recurrent_gated_delta_rule.fn).bind(*args, **kwargs).arguments


def reference(inputs):
    state = inputs["initial_state"].clone()
    output = torch.zeros_like(inputs["v"], dtype=torch.float32)
    q, k, v = [inputs[name][0].float() for name in ["q", "k", "v"]]
    group_size = v.shape[1] // q.shape[1]
    q = (q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(group_size, dim=1)
    k = (k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(group_size, dim=1)
    q *= q.shape[-1] ** -0.5
    cumulative = inputs["cu_seqlens"].tolist()
    for row, (start, end) in enumerate(zip(cumulative, cumulative[1:])):
        if start == end:
            continue
        read_index = inputs["ssm_state_indices"][row, inputs["num_accepted_tokens"][row] - 1]
        h = inputs["initial_state"][read_index].float().clone()
        for token in range(start, end):
            g = -inputs["A_log"].float().exp() * F.softplus(inputs["a_raw"][token].float() + inputs["dt_bias"].float())
            h *= g.exp()[:, None, None]
            delta = (v[token] - (h * k[token, :, :, None]).sum(1)) * inputs["b_raw"][token].float().sigmoid()[:, None]
            h += k[token, :, :, None] * delta[:, None, :]
            output[0, token] = (h * q[token, :, :, None]).sum(1)
            state[inputs["ssm_state_write_indices"][row, token - start]] = h.to(state.dtype)
    return output, state


@pytest.mark.parametrize("lengths,mtp_size,padding", [((3, 3), 3, 0), ((0,) * 7, 3, 7), ((2, 1, 0), 4, 2)])
def test_rebuild_uses_independent_state_slots(lengths, mtp_size, padding):
    inputs = make_inputs(lengths=lengths, mtp_size=mtp_size, padding=padding)
    snapshots = {name: value.clone() for name, value in inputs.items()}
    rebuilt = rebuild(inputs)
    tokens = inputs["q"].shape[1]
    active_seqs = (tokens + mtp_size - 1) // mtp_size
    assert rebuilt["cu_seqlens"].tolist() == [min(i * mtp_size, tokens) for i in range(len(lengths) + 1)]
    assert rebuilt["initial_state"] is inputs["initial_state"]
    assert rebuilt["num_accepted_tokens"].tolist() == [1] * len(lengths)
    assert rebuilt["ssm_state_write_indices"][:active_seqs].flatten().tolist() == list(range(active_seqs * mtp_size))
    kernel = kernel_module.mtp_fused_recurrent_gated_delta_rule
    assert kernel._run_key(**inputs) == kernel._run_key(**rebuilt)
    for name in ["q", "k", "v", "A_log", "dt_bias", "a_raw", "b_raw"]:
        assert rebuilt[name] is inputs[name]
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(inputs[name], snapshot)


def test_rebuild_checks_active_state_capacity():
    inputs = make_inputs(lengths=(0,) * 7, padding=7)
    # 7 个有效 token 只需 7 个槽；空序列及末组未使用的索引取余后也应在池内。
    inputs["initial_state"] = inputs["initial_state"][:7]
    rebuilt = rebuild(inputs)
    assert rebuilt["initial_state"] is inputs["initial_state"]
    for name in ["ssm_state_indices", "ssm_state_write_indices"]:
        indices = rebuilt[name]
        assert torch.all((indices >= 0) & (indices < 7))
        assert indices.flatten()[:7].tolist() == list(range(7))
        assert indices[2].tolist() == [6, 0, 1]
    # 槽数不足时仍拒绝调优，不能靠取余让有效请求的状态槽相互重叠。
    inputs["initial_state"] = inputs["initial_state"][:6]
    with pytest.raises(AssertionError, match="Not enough SSM state slots"):
        rebuild(inputs)


def test_keys_use_mtp_workload_and_state_dtype(monkeypatch):
    # 历史上下文长度与递推计算量无关，不应读取 full attention 的调优长度环境变量。
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", "invalid")
    kernel = kernel_module.mtp_fused_recurrent_gated_delta_rule
    inputs = make_inputs()
    key = kernel._static_key(**inputs)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert kernel._run_key(**inputs) == 12
    assert kernel._run_key(**make_inputs(lengths=(3, 3, 0))) == kernel._run_key(**inputs)
    assert kernel._run_key(**make_inputs(lengths=(3, 2))) == kernel._run_key(**inputs)
    assert kernel._run_key(**make_inputs(mtp_size=4)) == 16
    # 不同序列数共享 token 桶，11/12 个 token 共用一桶，第 13 个 token 进入下一桶。
    assert kernel._run_key(**make_inputs(lengths=(3, 3, 3, 2, 0))) == 12
    assert kernel._run_key(**make_inputs(lengths=(3, 3, 3, 3, 0))) == 12
    assert kernel._run_key(**make_inputs(lengths=(3, 3, 3, 3, 1))) == 24
    # head 数不再编码进 run key，但不同模型/TP 的 head 规模应使用不同静态配置文件。
    more_heads = make_inputs(heads=4, value_heads=8)
    assert kernel._run_key(**more_heads) == kernel._run_key(**inputs)
    assert kernel._static_key(**more_heads) != key
    assert kernel._static_key(**make_inputs(state_dtype=torch.bfloat16)) != key
    assert kernel._static_key(**make_inputs(mtp_size=4)) != key
    assert kernel._static_key(**make_inputs(padding=1)) == key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "lengths,mtp_size,head_k_dim,head_v_dim,heads,value_heads,dtype,state_dtype,separate_write",
    [
        ((1, 0, 1), 1, 64, 64, 2, 8, torch.bfloat16, torch.bfloat16, False),
        ((4, 2, 0), 4, 128, 128, 4, 8, torch.bfloat16, torch.float32, True),
        ((2, 3, 1), 3, 64, 80, 2, 2, torch.float16, torch.float32, False),
        ((3, 3), 3, 128, 128, 16, 16, torch.bfloat16, torch.float32, False),
        ((3, 1), 3, 256, 128, 2, 4, torch.bfloat16, torch.bfloat16, True),
    ],
)
def test_all_candidates_match_fp32(
    lengths,
    mtp_size,
    head_k_dim,
    head_v_dim,
    heads,
    value_heads,
    dtype,
    state_dtype,
    separate_write,
):
    inputs = make_inputs(
        lengths,
        mtp_size,
        "cuda",
        head_k_dim,
        head_v_dim,
        heads,
        value_heads,
        dtype,
        state_dtype,
        separate_write,
    )
    output_ref, state_ref = reference(inputs)
    state_before = inputs["initial_state"].clone()
    kernel = kernel_module.mtp_fused_recurrent_gated_delta_rule
    for config in kernel_module.get_test_configs():
        inputs["initial_state"].copy_(state_before)
        output, state = kernel(**inputs, run_config=config)
        assert state is inputs["initial_state"]
        torch.testing.assert_close(output.float(), output_ref, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(state.float(), state_ref.float(), atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dynamic", [False, True])
def test_autotune_reuses_state_cache_and_graph(monkeypatch, tmp_path, dynamic):
    inputs = make_inputs(
        lengths=(3, 2, 0) if dynamic else (3, 3),
        device="cuda",
        padding=2 if dynamic else 0,
    )
    kernel = kernel_module.mtp_fused_recurrent_gated_delta_rule
    state_before = inputs["initial_state"].clone()
    output_ref, state_ref = reference(inputs)
    benchmark = kernel._bench
    timings = []
    num_configs = len(kernel_module.get_test_configs())
    post_tuning_reference = None

    def checked_bench(*args, **kwargs):
        nonlocal post_tuning_reference
        rebuilt = inspect.signature(kernel.fn).bind(*args, **kwargs).arguments
        assert rebuilt["cu_seqlens"].tolist() == ([0, 3, 6, 7] if dynamic else [0, 3, 6])
        assert rebuilt["initial_state"] is inputs["initial_state"]
        elapsed = benchmark(*args, **kwargs)
        timings.append(elapsed)
        # benchmark 原地更新池内前 num_tokens 个槽，未使用的槽应保持不变。
        assert torch.isfinite(rebuilt["initial_state"]).all()
        num_tokens = inputs["q"].shape[1]
        assert not torch.equal(inputs["initial_state"][:num_tokens], state_before[:num_tokens])
        torch.testing.assert_close(inputs["initial_state"][num_tokens:], state_before[num_tokens:], atol=0, rtol=0)
        if len(timings) == num_configs:
            # 最后一次 benchmark 后的状态作为正式执行的初值，不再假定调优前后状态隔离。
            post_tuning_reference = reference(inputs)
        return elapsed

    monkeypatch.setattr(kernel, "_bench", checked_bench)
    # 普通 warmup 不搜索；decode 调优会原地更新状态，随后按原始请求布局正式执行。
    with Autotuner.autotune_warmup(AutotuneKernelType.GENERAL):
        kernel(**inputs)
    assert not timings
    inputs["initial_state"].copy_(state_before)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        output, _ = kernel(**inputs)
    assert len(timings) == num_configs and all(math.isfinite(t) for t in timings)
    tuned_output_ref, tuned_state_ref = post_tuning_reference
    valid_tokens = int(inputs["cu_seqlens"][-1])
    torch.testing.assert_close(
        output[:, :valid_tokens].float(),
        tuned_output_ref[:, :valid_tokens],
        atol=2e-3,
        rtol=2e-2,
    )
    torch.testing.assert_close(inputs["initial_state"], tuned_state_ref, atol=2e-3, rtol=2e-2)
    config_path = next(tmp_path.glob("*.json"))
    assert str(kernel._run_key(**inputs)) in json.loads(config_path.read_text())

    # 关闭已有配置预热后，文件重载和 FORCE 复用都只能正式执行一次，即使在 decode warmup 阶段。
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(
        autotuner_module,
        "get_triton_autotune_level",
        lambda: AutotuneLevel.FORCE_AUTOTUNE,
    )
    inputs["initial_state"].copy_(state_before)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    assert len(timings) == num_configs
    torch.testing.assert_close(inputs["initial_state"], state_ref, atol=2e-3, rtol=2e-2)
    assert not kernel.warmuped_configs_set
    # 预热结束后恢复测试请求的初始状态，正常调用及 Graph 回放每次只推进一次。
    inputs["initial_state"].copy_(state_before)
    kernel(**inputs)
    torch.testing.assert_close(inputs["initial_state"], state_ref, atol=2e-3, rtol=2e-2)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_state = kernel(**inputs)
    inputs["initial_state"].copy_(state_before)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        graph_output[:, :valid_tokens].float(),
        output_ref[:, :valid_tokens],
        atol=2e-3,
        rtol=2e-2,
    )
    torch.testing.assert_close(graph_state, state_ref, atol=2e-3, rtol=2e-2)
    # 同一 Graph 更新变长分组和接受位置，确保回放依赖原始输入而非调优构造的元数据。
    inputs["cu_seqlens"].copy_(torch.tensor([0, 1, 3, 3] if dynamic else [0, 2, 3], device="cuda"))
    inputs["num_accepted_tokens"].fill_(2)
    inputs["initial_state"].copy_(state_before)
    new_output_ref, new_state_ref = reference(inputs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output[:, :3].float(), new_output_ref[:, :3], atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(graph_state, new_state_ref, atol=2e-3, rtol=2e-2)
    assert len(timings) == num_configs

    # 改变序列数和 token 数但仍在同一桶内，decode warmup 应复用已有配置，不再搜索。
    same_bucket_inputs = make_inputs(lengths=(3,), device="cuda")
    assert kernel._static_key(**same_bucket_inputs) == kernel._static_key(**inputs)
    assert kernel._run_key(**same_bucket_inputs) == kernel._run_key(**inputs)
    same_output_ref, same_state_ref = reference(same_bucket_inputs)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        same_output, same_state = kernel(**same_bucket_inputs)
    assert len(timings) == num_configs
    torch.testing.assert_close(same_output.float(), same_output_ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(same_state, same_state_ref, atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("level", [0, 1, 2])
def test_first_decode_loads_history_without_advancing_state_twice(monkeypatch, tmp_path, level):
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: level)
    inputs = make_inputs(lengths=(3, 3), device="cuda")
    # 模拟两个刚完成 prefill 的请求，从各自的 canonical 状态槽开始执行首次 decode。
    inputs["ssm_state_indices"] = torch.arange(6, dtype=torch.int32, device="cuda").view(2, 3)
    inputs["ssm_state_write_indices"] = inputs["ssm_state_indices"]
    inputs["num_accepted_tokens"].fill_(1)
    output_ref, state_ref = reference(inputs)
    kernel = kernel_module.mtp_fused_recurrent_gated_delta_rule
    config = {"BV": 8, "num_warps": 1, "num_stages": 1}
    filename = autotuner_module.KernelConfigs.get_config_file_name(kernel._static_key(**inputs))
    cache_file = tmp_path / filename
    cache_file.write_text(
        json.dumps(
            {
                str(kernel._run_key(**inputs)): config,
                "24": {"BV": 16, "num_warps": 2, "num_stages": 1},
            }
        )
    )
    calls = []
    fn = kernel.fn

    def count_calls(*args, **kwargs):
        calls.append(kwargs.get("run_config"))
        return fn(*args, **kwargs)

    monkeypatch.setattr(kernel, "fn", count_calls)
    assert not Autotuner.is_autotune_warmup()
    output, state = kernel(**inputs)
    assert calls == [config]
    assert state is inputs["initial_state"]
    assert not kernel.warmuped_configs_set
    torch.testing.assert_close(output.float(), output_ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, state_ref, atol=2e-3, rtol=2e-2)
