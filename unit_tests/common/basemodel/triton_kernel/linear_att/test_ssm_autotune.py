import collections
import math

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att import ssm_autotune as tuning
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache
from lightllm.common.kernel_config import KernelConfigs
from lightllm.common import kernel_config as kernel_config_module
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType


def make_inputs(mode, dtype=torch.bfloat16, device="cpu", width=4, projection_mode="inline", kda=False):
    layers, slots, batch, h, hv, kd, vd = 2, 9, 3, 2, 4, 32, 64
    state = torch.randn(layers, slots, hv, kd, vd, dtype=dtype, device=device) * 0.01
    if mode == "replay":
        cache = ReplaySSMCache(state, 8, width, num_key_heads=h, projection_mode=projection_mode, kda=kda)
    else:
        cache = CompactSSMCache(state, width, torch.bfloat16, num_key_heads=h, kda=kda)
    tokens = batch * width
    packed = torch.randn(1, tokens, 2 * h * kd + hv * vd, dtype=torch.bfloat16, device=device)
    q, k, v = [
        x.view(1, tokens, heads, dim)
        for x, heads, dim in zip(packed.split([h * kd, h * kd, hv * vd], -1), [h, h, hv], [kd, kd, vd])
    ]
    gates = torch.randn(tokens, hv * (kd + 1) if kda else hv * 2, device=device, dtype=q.dtype)
    a, b = gates[:, :-hv], gates[:, -hv:]
    return dict(
        cache=cache,
        mode=mode,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        a_log=torch.randn(hv, device=device) * 0.1,
        bias=torch.randn(a.shape[-1], device=device) * 0.1,
        cu_seqlens=torch.arange(batch + 1, device=device, dtype=torch.int32) * width,
    )


def tensors(cache):
    return {name: x for name, x in vars(cache).items() if isinstance(x, torch.Tensor)}


def test_replay_projection_has_separate_autotune_cache():
    inline = make_inputs("replay")
    precompute = make_inputs("replay", projection_mode="precompute")
    assert tuning.select_config._static_key(**inline) != tuning.select_config._static_key(**precompute)
    _, rebuilt = tuning.rebuild_inputs(**precompute)
    assert rebuilt["cache"].projection_mode == "precompute"
    assert rebuilt["cache"].keys.dtype == precompute["cache"].keys.dtype == torch.bfloat16
    assert tuning.select_config._static_key(**precompute) == tuning.select_config._static_key(**rebuilt)
    key = tuning.select_config._static_key(**precompute)
    legacy = dict(key, v=2)
    assert KernelConfigs.get_config_file_name(key) != KernelConfigs.get_config_file_name(legacy)
    assert len(KernelConfigs.get_config_file_name(key).encode()) <= 255
    precompute["cache"].keys = precompute["cache"].keys.float()
    assert tuning.select_config._static_key(**precompute) != key


@pytest.fixture(autouse=True)
def isolate_tuner(monkeypatch, tmp_path):
    torch.manual_seed(42)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: 1)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(kernel_config_module, "get_current_device_name", lambda: "NVIDIA H200")
    tuner = tuning.select_config
    monkeypatch.setattr(tuner, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(tuner, "cached_configs", {})
    monkeypatch.setattr(tuner, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(tuner, "warmuped_configs_set", set())


@pytest.mark.parametrize("mode", ["gdn", "replay"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_rebuild_has_real_requests_and_independent_packed_inputs(mode, dtype, kda):
    inputs = make_inputs(mode, dtype, kda=kda)
    cache = inputs["cache"]
    for x in tensors(cache).values():
        x.fill_(1)
    args, rebuilt = tuning.rebuild_inputs(**inputs)
    assert not args
    scratch = rebuilt["cache"]
    assert scratch.state.shape[:2] == (2, 4)  # Active batch + HOLD, not the serving pool.
    assert scratch._config_is_fixed
    assert rebuilt["workload"][0].tolist() == [0, 1, 2]
    assert rebuilt["workload"][1].tolist() == [3, 3, 3, 0]
    assert rebuilt["workload"][2].tolist() == [0, 1, 2, 0]
    for name in ("q", "k", "v", "a", "b", "a_log", "bias"):
        assert rebuilt[name].stride() == inputs[name].stride()
        assert rebuilt[name].data_ptr() != inputs[name].data_ptr()
    assert tuning.select_config._static_key(**inputs) == tuning.select_config._static_key(**rebuilt)
    assert tuning.select_config._run_key(**inputs) == tuning.select_config._run_key(**rebuilt)
    for x in tensors(scratch).values():
        x.zero_()
    for x in tensors(cache).values():
        assert torch.all(x == 1)
    key = tuning.select_config._static_key(**inputs)
    assert len(KernelConfigs.get_config_file_name(key).encode()) <= 255
    other = make_inputs(mode, torch.float32 if dtype == torch.bfloat16 else torch.bfloat16, kda=kda)
    assert tuning.select_config._static_key(**other) != key
    other = make_inputs(mode, dtype, width=3, kda=kda)
    assert tuning.select_config._static_key(**other) != key
    inputs["q"] = inputs["q"].contiguous()
    assert tuning.select_config._static_key(**inputs) != key


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_layout_is_fixed_before_graph_capture_and_cached_configs_do_not_execute(monkeypatch, level):
    inputs = make_inputs("gdn")
    cache = inputs["cache"]
    config = {"BV": 16, "num_warps": 1, "num_stages": 2}
    tuner = tuning.select_config
    key = autotuner_module.frozendict(tuner._static_key(**inputs))
    tuner.cached_configs[key] = {str(tuner._run_key(**inputs)): config}
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: level)
    monkeypatch.setattr(tuner, "_bench", lambda *args, **kwargs: pytest.fail("cached layout must not benchmark"))
    default = cache.run_config.copy()
    with Autotuner.autotune_warmup(AutotuneKernelType.GENERAL):
        tuning.configure_cache(**inputs)
    assert cache.run_config == default and not cache._config_is_fixed
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        selected = tuning.configure_cache(**inputs)
    assert selected == (default if level == 3 else config)
    assert cache.run_config == default and not cache._config_is_fixed
    # Reusing the same bucket must not tune again.
    monkeypatch.setattr(tuning, "select_config", lambda *args, **kwargs: pytest.fail("layout changed after capture"))
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        tuning.configure_cache(**inputs)
    assert not tuner.warmuped_configs_set


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "mode,width,projection_mode",
    [
        ("gdn", 4, "inline"),
        ("replay", 4, "inline"),
        ("replay", 1, "inline"),
        ("replay", 4, "precompute"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_joint_tuning_isolated_state_and_graph_replay(monkeypatch, mode, dtype, width, projection_mode, kda):
    inputs = make_inputs(mode, dtype, "cuda", width=width, projection_mode=projection_mode, kda=kda)
    if width == 1:
        for name in ("q", "k", "v"):
            inputs[name] = inputs[name].transpose(0, 1)
        inputs["cu_seqlens"] = None
    cache = inputs["cache"]
    for name, x in tensors(cache).items():
        if name != "state":
            x.zero_()
    before = {name: x.clone() for name, x in tensors(cache).items()}
    tuner = tuning.select_config
    configs = [{"BV": 16, "num_warps": 1, "num_stages": 1}, {"BV": 32, "num_warps": 2, "num_stages": 2}]
    monkeypatch.setattr(tuner, "configs_gen_func", lambda: configs)
    bench = tuner._bench
    timings = []

    def checked_bench(*args, **kwargs):
        assert kwargs["cache"] is not cache
        elapsed = bench(*args, **kwargs)
        assert math.isfinite(elapsed), f"failed {mode} {kwargs['run_config']}"
        timings.append(elapsed)
        for name, x in tensors(cache).items():
            torch.testing.assert_close(x, before[name], atol=0, rtol=0)
        return elapsed

    monkeypatch.setattr(tuner, "_bench", checked_bench)
    args = [inputs[name] for name in ("q", "k", "v", "a", "b", "a_log", "bias")]
    reqs = torch.full((3,), cache.hold, device="cuda", dtype=torch.int32)
    cache.prepare_decode(reqs, inputs["cu_seqlens"])
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        # Exercise the real forward hook with the HOLD-only graph warmup input.
        cache.forward(0, *args, reqs, inputs["cu_seqlens"])
    assert len(timings) == len(configs)
    selected = cache.get_run_config(*args, inputs["cu_seqlens"])
    assert selected in configs
    # An explicit identical layout supplies a reference for capture/replay and
    # alternating real/HOLD rows. It must never trigger its own tuning.
    expected = make_inputs(mode, dtype, "cuda", width=width, projection_mode=projection_mode, kda=kda)["cache"]
    expected.run_config = cache.run_config
    expected._config_is_fixed = True
    expected.state.copy_(before["state"])
    accepted = torch.zeros(cache.state.shape[1], device="cuda", dtype=torch.int32)

    def step(target):
        target.prepare_decode(reqs, inputs["cu_seqlens"])
        out = [
            target.forward(layer, *args, reqs, inputs["cu_seqlens"], run_config=selected)
            for layer in range(target.state.shape[0])
        ]
        if mode == "gdn":
            target.accept_updates(reqs, accepted, run_config=selected)
        else:
            target.accept_updates(reqs, accepted)
        return out

    reqs.fill_(cache.hold)
    step(cache)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_out = step(cache)
    reqs.copy_(torch.tensor([2, 0, cache.hold], device="cuda", dtype=torch.int32))
    for iteration in range(12):
        accepted[0] = iteration % width
        accepted[2] = (iteration + 1) % width
        graph.replay()
        expected_out = step(expected)
        for actual, ref in zip(actual_out, expected_out):
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        torch.testing.assert_close(cache.state, expected.state, atol=0, rtol=0)
        if mode == "replay":
            cursor = cache.cursors.clone()
            snapshot = cache.snapshot_accepted_state(0)
            torch.testing.assert_close(cache.cursors, cursor, atol=0, rtol=0)
            torch.testing.assert_close(snapshot, expected.snapshot_accepted_state(0), atol=0, rtol=0)
    cache.merge_accepted_updates(reqs[:2])
    expected.merge_accepted_updates(reqs[:2])
    torch.testing.assert_close(cache.state, expected.state, atol=0, rtol=0)
    torch.testing.assert_close(cache.state[:, cache.hold], before["state"][:, cache.hold], atol=0, rtol=0)
    assert len(timings) == len(configs)


@pytest.mark.parametrize("mode", ["gdn", "replay"])
def test_kda_rule_and_gate_bound_separate_tuning_cache(mode):
    inputs = make_inputs(mode, kda=True)
    key = tuning.select_config._static_key(**inputs)
    assert key != tuning.select_config._static_key(**make_inputs(mode))
    _, rebuilt = tuning.rebuild_inputs(**inputs)
    assert rebuilt["cache"].kda
    assert rebuilt["cache"].lower_bound == inputs["cache"].lower_bound
    assert key == tuning.select_config._static_key(**rebuilt)
    inputs["cache"].lower_bound = -2.5
    assert key != tuning.select_config._static_key(**inputs)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kda_production_tuning_cache_can_be_written(monkeypatch, tmp_path, tp, dtype):
    from types import SimpleNamespace

    monkeypatch.setattr(kernel_config_module, "get_current_device_name", lambda: "NVIDIA H100 80GB HBM3")
    heads = 64 // tp
    cache = SimpleNamespace(
        state=SimpleNamespace(shape=(34, 1, heads, 128, 128), dtype=dtype),
        keys=SimpleNamespace(dtype=torch.bfloat16),
        verify_width=3,
        num_key_heads=heads,
        capacity=8,
        kda=True,
        lower_bound=-5.0,
        projection_mode="precompute",
    )
    packed = torch.empty(1, 3, heads * 128 * 3, dtype=torch.bfloat16)
    q, k, v = [x.view(1, 3, heads, 128) for x in packed.chunk(3, -1)]
    a = torch.empty(3, heads * 128, dtype=q.dtype)
    b = torch.empty(3, heads, dtype=q.dtype)
    key = tuning.static_key(cache, "replay", q, k, v, a, b, torch.tensor([0, 3]))
    path = tmp_path / KernelConfigs.get_config_file_name(key)
    path.write_text("{}")
    assert path.is_file()


def test_batch_configs_survive_other_warmups_and_preserve_padding(monkeypatch):
    inputs = make_inputs("gdn")
    cache = inputs["cache"]
    small = dict(inputs, **{name: inputs[name][:, :4] for name in ("q", "k", "v")})
    small["a"], small["b"] = inputs["a"][:4], inputs["b"][:4]
    small["cu_seqlens"] = torch.tensor([0, 4, 4], dtype=torch.int32)
    selected = []

    def choose(cache, mode, q, *args):
        config = {"BV": 16 if q.shape[1] == 4 else 32, "num_warps": 2}
        selected.append(config)
        return config

    monkeypatch.setattr(tuning, "select_config", choose)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        large_config = tuning.configure_cache(**inputs)
        small_config = tuning.configure_cache(**small)
        assert tuning.configure_cache(**inputs) is large_config
    assert len(selected) == 2 and large_config != small_config
    assert tuning.configure_cache(**small) is small_config
    assert cache.run_config == {"BV": 8, "num_warps": 1}
    _, rebuilt = tuning.rebuild_inputs(**small)
    assert rebuilt["workload"][0].tolist() == [0, rebuilt["cache"].hold]
    assert rebuilt["cu_seqlens"].tolist() == [0, 4, 4]
    assert tuning.run_key(**{k: rebuilt[k] for k in ("cache", "q", "cu_seqlens")}) == (4, 2)
    # A graph token bucket can also exceed the request-pool capacity.
    _, limited = tuning.rebuild_inputs(**dict(inputs, cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32)))
    assert limited["workload"][0].tolist() == [0, 1]
    assert limited["cu_seqlens"].tolist() == [0, 4, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode,projection", [("gdn", "inline"), ("replay", "inline"), ("replay", "precompute")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_two_verify_graphs_accept_with_their_own_configs(monkeypatch, mode, projection, dtype, kda, dynamic):
    from types import SimpleNamespace
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.batch_objs import PostLayerOutput, ModelMtpOutputCollector
    from lightllm.common.req_manager.linear_att import ReqManagerForMamba

    inputs = make_inputs(mode, dtype, "cuda", projection_mode=projection, kda=kda)
    cache = inputs["cache"]
    initial = cache.state.clone()
    configs = [{"BV": 8, "num_warps": 1, "num_stages": 1}, {"BV": 32, "num_warps": 2, "num_stages": 1}]
    monkeypatch.setattr(tuning, "select_config", lambda *args: configs[args[2].shape[1] > 8])
    manager = object.__new__(ReqManagerForMamba)
    manager.ssm_update_cache = cache
    manager.req_to_mtp_state_index = torch.zeros(9, dtype=torch.int32, device="cuda")
    buckets = []
    for index, ids in enumerate(([0], [1, 3])):
        # The physical verify buckets include HOLD, but acceptance is unpadded.
        count = len(ids) + 1
        args = [inputs[name] for name in ("q", "k", "v", "a", "b", "a_log", "bias")]
        args[:3] = [x[:, : count * 4].contiguous() for x in args[:3]]
        args[3:5] = [x[: count * 4].contiguous() for x in args[3:5]]
        capacity = min(count * 4, cache.hold) if dynamic else count
        reqs = torch.tensor(ids + [cache.hold] * (capacity - len(ids)), dtype=torch.int32, device="cuda")
        cu = (torch.arange(capacity + 1, dtype=torch.int32, device="cuda") * 4).clamp_max(count * 4)
        with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
            config = cache.get_run_config(*args, cu)
        assert config == configs[index]
        ref = make_inputs(mode, dtype, "cuda", projection_mode=projection, kda=kda)["cache"]
        ref.state.copy_(initial)
        ref.run_config = config
        ref._config_is_fixed = True

        def forward(target):
            target.prepare_decode(reqs, cu)
            return [target.forward(layer, *args, reqs, cu, run_config=config) for layer in range(2)]

        forward(cache)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = forward(cache)
        state = SimpleNamespace(
            hidden_collector=SimpleNamespace(finish_output=lambda **kw: ModelMtpOutputCollector()),
            prompt_logics=None,
            ssm_run_config=config,
        )
        output = TpPartBaseModel._create_model_output(
            None, PostLayerOutput(logits=outputs[0].view(count * 4, -1)), state
        )
        output = TpPartBaseModel._create_unpad_decode_model_output(None, output, len(ids) * 4)
        assert output.ssm_run_config is config
        buckets.append((graph, outputs, output, ref, args, reqs, cu, ids))
    cache.state.copy_(initial)
    if mode == "replay":
        cache.cursors.zero_()
    req_order = [0, 1, 3]
    rows = torch.tensor([r for r in req_order for _ in range(4)], dtype=torch.int32, device="cuda")
    mtp = torch.arange(4, dtype=torch.int32, device="cuda").repeat(3)
    starts = torch.arange(3, dtype=torch.int32, device="cuda") * 4
    for iteration in range(24):
        # Alternate replay order; neither Graph replay executes Python config selection.
        for bucket_index in [0, 1] if iteration % 2 else [1, 0]:
            graph, outputs, output, ref, args, reqs, cu, ids = buckets[bucket_index]
            graph.replay()
            ref.prepare_decode(reqs, cu)
            for layer in range(2):
                expected = ref.forward(layer, *args, reqs, cu)
                torch.testing.assert_close(outputs[layer], expected, atol=0, rtol=0)
        counts = [1 + (iteration + i) % 4 for i in range(3)]
        flags = torch.tensor([int(j < n) for n in counts for j in range(4)], dtype=torch.int32, device="cuda")
        manager.update_mtp_state(
            starts,
            rows,
            mtp,
            flags,
            4,
            ssm_accept_batches=tuple((len(b[-1]), b[2].ssm_run_config) for b in buckets),
        )
        for _, _, _, ref, _, reqs, _, ids in buckets:
            ref.accept_updates(reqs[:-1], manager.req_to_mtp_state_index)
            torch.testing.assert_close(cache.state[:, ids], ref.state[:, ids], atol=0, rtol=0)
            if mode == "replay":
                assert torch.equal(cache.cursors[ids], ref.cursors[ids])
    assert len(cache.batch_configs) == 2
