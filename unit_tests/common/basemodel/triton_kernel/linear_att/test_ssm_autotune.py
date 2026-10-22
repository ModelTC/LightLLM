import collections
import math
from dataclasses import asdict

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att import ssm_autotune as tuning
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMBackend
from lightllm.common.basemodel.triton_kernel.linear_att.compact_ssm import CompactSSMBackend
from lightllm.common.linear_state import LinearStateManager
from lightllm.common.kernel_config import KernelConfigs
from lightllm.common import kernel_config as kernel_config_module
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType


def make_inputs(mode, dtype=torch.bfloat16, device="cpu", width=4, compute_mode="inline", kda=False):
    layers, slots, batch, h, hv, kd, vd = 2, 9, 3, 2, 4, 32, 64
    state = torch.randn(layers, slots, hv, kd, vd, dtype=dtype, device=device) * 0.01
    if mode == "replay":
        ssm_backend = ReplaySSMBackend(state, 8, width, num_key_heads=h, compute_mode=compute_mode, kda=kda)
    else:
        ssm_backend = CompactSSMBackend(state, width, torch.bfloat16, num_key_heads=h, kda=kda)
    tokens = batch * width
    packed = torch.randn(1, tokens, 2 * h * kd + hv * vd, dtype=torch.bfloat16, device=device)
    q, k, v = [
        x.view(1, tokens, heads, dim)
        for x, heads, dim in zip(packed.split([h * kd, h * kd, hv * vd], -1), [h, h, hv], [kd, kd, vd])
    ]
    gates = torch.randn(tokens, hv * (kd + 1) if kda else hv * 2, device=device, dtype=q.dtype)
    a, b = gates[:, :-hv], gates[:, -hv:]
    return dict(
        ssm_backend=ssm_backend,
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


def tensors(ssm_backend):
    return {name: x for name, x in vars(ssm_backend).items() if isinstance(x, torch.Tensor)}


def test_replay_projection_has_separate_autotune_cache():
    inline = make_inputs("replay")
    precompute = make_inputs("replay", compute_mode="precompute")
    assert tuning.select_config._static_key(**inline) != tuning.select_config._static_key(**precompute)
    _, rebuilt = tuning.rebuild_inputs(**precompute)
    assert rebuilt["ssm_backend"].compute_mode == "precompute"
    assert rebuilt["ssm_backend"].keys.dtype == precompute["ssm_backend"].keys.dtype == torch.bfloat16
    assert tuning.select_config._static_key(**precompute) == tuning.select_config._static_key(**rebuilt)
    key = tuning.select_config._static_key(**precompute)
    legacy = dict(key, v=2)
    assert KernelConfigs.get_config_file_name(key) != KernelConfigs.get_config_file_name(legacy)
    assert len(KernelConfigs.get_config_file_name(key).encode()) <= 255
    precompute["ssm_backend"].keys = precompute["ssm_backend"].keys.float()
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
    monkeypatch.setattr(tuner, "_oom_keys", set())


@pytest.mark.parametrize("mode", ["compact", "replay"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_rebuild_has_real_requests_and_independent_packed_inputs(mode, dtype, kda):
    inputs = make_inputs(mode, dtype, kda=kda)
    ssm_backend = inputs["ssm_backend"]
    for x in tensors(ssm_backend).values():
        x.fill_(1)
    args, rebuilt = tuning.rebuild_inputs(**inputs)
    assert not args
    scratch = rebuilt["ssm_backend"]
    assert scratch.state.shape[:2] == (1, 4)  # One layer, active batch + HOLD.
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
    for x in tensors(ssm_backend).values():
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
    inputs = make_inputs("compact")
    ssm_backend = inputs["ssm_backend"]
    config = {"BV": 16, "num_warps": 1, "num_stages": 2}
    tuner = tuning.select_config
    key = autotuner_module.frozendict(tuner._static_key(**inputs))
    tuner.cached_configs[key] = {str(tuner._run_key(**inputs)): config}
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: level)
    monkeypatch.setattr(tuner, "_bench", lambda *args, **kwargs: pytest.fail("cached layout must not benchmark"))
    default = ssm_backend.run_config.copy()
    with Autotuner.autotune_warmup(AutotuneKernelType.GENERAL):
        tuning.resolve_run_config(**inputs)
    assert ssm_backend.run_config == default and not ssm_backend._config_is_fixed
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        selected = tuning.resolve_run_config(**inputs)
    assert selected == (default if level == 3 else config)
    assert ssm_backend.run_config == default and not ssm_backend._config_is_fixed
    # Reusing the same bucket must not tune again.
    monkeypatch.setattr(tuning, "select_config", lambda *args, **kwargs: pytest.fail("layout changed after capture"))
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        tuning.resolve_run_config(**inputs)
    assert not tuner.warmuped_configs_set


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "mode,width,compute_mode",
    [
        ("compact", 4, "inline"),
        ("replay", 4, "inline"),
        ("replay", 1, "inline"),
        ("replay", 4, "precompute"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_joint_tuning_isolated_state_and_graph_replay(monkeypatch, mode, dtype, width, compute_mode, kda):
    inputs = make_inputs(mode, dtype, "cuda", width=width, compute_mode=compute_mode, kda=kda)
    if width == 1:
        for name in ("q", "k", "v"):
            inputs[name] = inputs[name].transpose(0, 1)
        inputs["cu_seqlens"] = None
    ssm_backend = inputs["ssm_backend"]
    for name, x in tensors(ssm_backend).items():
        if name != "state":
            x.zero_()
    before = {name: x.clone() for name, x in tensors(ssm_backend).items()}
    tuner = tuning.select_config
    configs = [{"BV": 16, "num_warps": 1, "num_stages": 1}, {"BV": 32, "num_warps": 2, "num_stages": 2}]
    monkeypatch.setattr(tuner, "configs_gen_func", lambda: configs)
    bench = tuner._bench
    timings = []

    def checked_bench(*args, **kwargs):
        assert kwargs["ssm_backend"] is not ssm_backend
        assert kwargs["ssm_backend"].state.shape[0] == 1
        elapsed = bench(*args, **kwargs)
        assert math.isfinite(elapsed), f"failed {mode} {kwargs['run_config']}"
        if mode == "replay":
            # The measured cycle must reach a fold, not just empty-history updates.
            scratch = kwargs["ssm_backend"]
            assert scratch.cursors[0].item() >= 2 * scratch.capacity
        timings.append(elapsed)
        for name, x in tensors(ssm_backend).items():
            torch.testing.assert_close(x, before[name], atol=0, rtol=0)
        return elapsed

    monkeypatch.setattr(tuner, "_bench", checked_bench)
    args = [inputs[name] for name in ("q", "k", "v", "a", "b", "a_log", "bias")]
    reqs = torch.full((3,), ssm_backend.hold_req_id, device="cuda", dtype=torch.int32)
    ssm_backend.prepare_decode(reqs, inputs["cu_seqlens"])
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        # Exercise the real forward hook with the HOLD-only graph warmup input.
        ssm_backend.forward(0, *args, reqs, inputs["cu_seqlens"])
    assert len(timings) == len(configs)
    selected = ssm_backend.get_run_config(*args, inputs["cu_seqlens"])
    assert asdict(selected) in configs
    # An explicit identical layout supplies a reference for capture/replay and
    # alternating real/HOLD rows. It must never trigger its own tuning.
    expected = make_inputs(mode, dtype, "cuda", width=width, compute_mode=compute_mode, kda=kda)["ssm_backend"]
    expected.run_config = ssm_backend.run_config
    expected._config_is_fixed = True
    expected.state.copy_(before["state"])
    accepted = torch.zeros(ssm_backend.state.shape[1], device="cuda", dtype=torch.int32)

    def step(target):
        target.prepare_decode(reqs, inputs["cu_seqlens"])
        out = [
            target.forward(layer, *args, reqs, inputs["cu_seqlens"], run_config=selected)[0]
            for layer in range(target.state.shape[0])
        ]
        if mode == "compact":
            target.accept_updates(reqs, accepted, run_config=selected)
        else:
            target.accept_updates(reqs, accepted)
        return out

    reqs.fill_(ssm_backend.hold_req_id)
    step(ssm_backend)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_out = step(ssm_backend)
    reqs.copy_(torch.tensor([2, 0, ssm_backend.hold_req_id], device="cuda", dtype=torch.int32))
    for iteration in range(12):
        accepted[0] = iteration % width
        accepted[2] = (iteration + 1) % width
        graph.replay()
        expected_out = step(expected)
        for actual, ref in zip(actual_out, expected_out):
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        torch.testing.assert_close(ssm_backend.state, expected.state, atol=0, rtol=0)
        if mode == "replay":
            cursor = ssm_backend.cursors.clone()
            snapshot = ssm_backend.get_accepted_state(0)
            torch.testing.assert_close(ssm_backend.cursors, cursor, atol=0, rtol=0)
            torch.testing.assert_close(snapshot, expected.get_accepted_state(0), atol=0, rtol=0)
    ssm_backend.merge_accepted_updates(reqs[:2])
    expected.merge_accepted_updates(reqs[:2])
    torch.testing.assert_close(ssm_backend.state, expected.state, atol=0, rtol=0)
    torch.testing.assert_close(
        ssm_backend.state[:, ssm_backend.hold_req_id], before["state"][:, ssm_backend.hold_req_id], atol=0, rtol=0
    )
    assert len(timings) == len(configs)


@pytest.mark.parametrize("mode", ["compact", "replay"])
def test_kda_rule_and_gate_bound_separate_tuning_cache(mode):
    inputs = make_inputs(mode, kda=True)
    key = tuning.select_config._static_key(**inputs)
    assert key != tuning.select_config._static_key(**make_inputs(mode))
    _, rebuilt = tuning.rebuild_inputs(**inputs)
    assert rebuilt["ssm_backend"].kda
    assert rebuilt["ssm_backend"].gate_lower_bound == inputs["ssm_backend"].gate_lower_bound
    assert key == tuning.select_config._static_key(**rebuilt)
    inputs["ssm_backend"].gate_lower_bound = -2.5
    assert key != tuning.select_config._static_key(**inputs)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_production_tuning_cache_can_be_written(monkeypatch, tmp_path, tp, dtype, kda):
    from types import SimpleNamespace

    monkeypatch.setattr(kernel_config_module, "get_current_device_name", lambda: "NVIDIA H100 80GB HBM3 MIG 3g.40gb")
    heads = 64 // tp
    ssm_backend = SimpleNamespace(
        state=SimpleNamespace(shape=(34, 1, heads, 128, 128), dtype=dtype),
        keys=SimpleNamespace(dtype=torch.bfloat16),
        verify_width=3,
        num_key_heads=heads,
        capacity=8,
        kda=kda,
        gate_lower_bound=-5.0,
        compute_mode="precompute",
    )
    packed = torch.empty(1, 3, heads * 128 * 3, dtype=torch.bfloat16)
    q, k, v = [x.view(1, 3, heads, 128) for x in packed.chunk(3, -1)]
    a = torch.empty(3, heads * 128 if kda else heads, dtype=q.dtype)
    b = torch.empty(3, heads, dtype=q.dtype)
    key = tuning.static_key(ssm_backend, "replay", q, k, v, a, b, torch.tensor([0, 3]))
    path = tmp_path / KernelConfigs.get_config_file_name(key)
    path.write_text("{}")
    assert path.is_file()


def test_batch_configs_survive_other_warmups_and_preserve_padding(monkeypatch):
    inputs = make_inputs("compact")
    ssm_backend = inputs["ssm_backend"]
    small = dict(inputs, **{name: inputs[name][:, :4] for name in ("q", "k", "v")})
    small["a"], small["b"] = inputs["a"][:4], inputs["b"][:4]
    small["cu_seqlens"] = torch.tensor([0, 4, 4], dtype=torch.int32)
    selected = []

    def choose(ssm_backend, mode, q, *args):
        config = {"BV": 16 if q.shape[1] == 4 else 32, "num_warps": 2}
        selected.append(config)
        return config

    monkeypatch.setattr(tuning, "select_config", choose)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        large_config = tuning.resolve_run_config(**inputs)
        small_config = tuning.resolve_run_config(**small)
        assert tuning.resolve_run_config(**inputs) is large_config
    assert len(selected) == 2 and large_config != small_config
    assert tuning.resolve_run_config(**small) is small_config
    assert ssm_backend.run_config == {"BV": 8, "num_warps": 1}
    _, rebuilt = tuning.rebuild_inputs(**small)
    assert rebuilt["workload"][0].tolist() == [0, rebuilt["ssm_backend"].hold_req_id]
    assert rebuilt["cu_seqlens"].tolist() == [0, 4, 4]
    assert tuning.run_key(**{k: rebuilt[k] for k in ("ssm_backend", "q", "cu_seqlens")}) == (4, 2)
    # A graph token bucket can also exceed the request-pool capacity.
    _, limited = tuning.rebuild_inputs(**dict(inputs, cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32)))
    assert limited["workload"][0].tolist() == [0, 1]
    assert limited["cu_seqlens"].tolist() == [0, 4, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode,projection", [("compact", "inline"), ("replay", "inline"), ("replay", "precompute")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_two_verify_graphs_accept_with_their_own_configs(monkeypatch, mode, projection, dtype, kda, dynamic):
    from types import SimpleNamespace
    import weakref
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.common.basemodel.attention.linear.gdn import GDNLinearAttDecodeAttState
    from lightllm.common.basemodel.attention.linear.kda import KDALinearAttDecodeAttState
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.batch_objs import PostLayerOutput, ModelMtpOutputCollector
    from lightllm.common.req_manager.linear_att import ReqManagerForMamba

    inputs = make_inputs(mode, dtype, "cuda", compute_mode=projection, kda=kda)
    ssm_backend = inputs["ssm_backend"]
    initial = ssm_backend.state.clone()
    configs = [{"BV": 8, "num_warps": 1, "num_stages": 1}, {"BV": 32, "num_warps": 2, "num_stages": 1}]
    monkeypatch.setattr(tuning, "select_config", lambda *args: configs[args[2].shape[1] > 8])
    manager = object.__new__(ReqManagerForMamba)
    manager.linear_state_manager = object.__new__(LinearStateManager)
    manager.linear_state_manager.ssm_backend = ssm_backend
    manager.linear_state_manager.accepted_state_index = torch.zeros(9, dtype=torch.int32, device="cuda")
    buckets = []
    for index, ids in enumerate(([0], [1, 3])):
        # The physical verify buckets include HOLD, but acceptance is unpadded.
        count = len(ids) + 1
        args = [inputs[name] for name in ("q", "k", "v", "a", "b", "a_log", "bias")]
        args[:3] = [x[:, : count * 4].contiguous() for x in args[:3]]
        args[3:5] = [x[: count * 4].contiguous() for x in args[3:5]]
        capacity = min(count * 4, ssm_backend.hold_req_id) if dynamic else count
        reqs = torch.tensor(ids + [ssm_backend.hold_req_id] * (capacity - len(ids)), dtype=torch.int32, device="cuda")
        cu = (torch.arange(capacity + 1, dtype=torch.int32, device="cuda") * 4).clamp_max(count * 4)
        with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
            config = ssm_backend.get_run_config(*args, cu)
        assert asdict(config) == configs[index]
        ref = make_inputs(mode, dtype, "cuda", compute_mode=projection, kda=kda)["ssm_backend"]
        ref.state.copy_(initial)
        ref.run_config = asdict(config)
        ref._config_is_fixed = True

        def forward(target):
            target.prepare_decode(reqs, cu)
            outputs = []
            selected = None
            for layer in range(2):
                result, actual_config = target.forward(layer, *args, reqs, cu, run_config=selected)
                if selected is not None:
                    assert actual_config is selected
                selected = actual_config
                outputs.append(result)
            return outputs, selected

        forward(ssm_backend)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs, config = forward(ssm_backend)
        state = InferStateInfo()
        state.req_manager = manager
        state.hidden_collector = SimpleNamespace(finish_output=lambda **kw: ModelMtpOutputCollector())
        att_class = KDALinearAttDecodeAttState if kda else GDNLinearAttDecodeAttState
        state.decode_att_state1 = att_class(_infer_state_weak_ref=weakref.ref(state), run_config=config)
        output = TpPartBaseModel._create_model_output(
            None, PostLayerOutput(logits=outputs[0].view(count * 4, -1)), state
        )
        commit = output.linear_state_commit
        assert (commit is not None) == (mode == "compact")
        output = TpPartBaseModel._create_unpad_decode_model_output(None, output, len(ids) * 4)
        assert output.linear_state_commit is commit
        buckets.append((graph, outputs, output, ref, args, reqs, cu, ids))
    # Published contexts and captured launches must survive later changes to
    # the tuner's dictionaries, including when two graphs replay alternately.
    for config in configs:
        config.update(BV=128, num_warps=4, num_stages=3)
    ssm_backend.state.copy_(initial)
    if mode == "replay":
        ssm_backend.cursors.zero_()
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
                expected = ref.forward(layer, *args, reqs, cu)[0]
                torch.testing.assert_close(outputs[layer], expected, atol=0, rtol=0)
        counts = [1 + (iteration + i) % 4 for i in range(3)]
        flags = torch.tensor([int(j < n) for n in counts for j in range(4)], dtype=torch.int32, device="cuda")
        manager.update_mtp_state(
            starts,
            rows,
            mtp,
            flags,
            4,
            linear_state_batches=tuple((len(b[-1]), b[2].linear_state_commit) for b in buckets),
        )
        for _, _, _, ref, _, reqs, _, ids in buckets:
            ref.accept_updates(reqs[:-1], manager.linear_state_manager.accepted_state_index)
            torch.testing.assert_close(ssm_backend.state[:, ids], ref.state[:, ids], atol=0, rtol=0)
            if mode == "replay":
                assert torch.equal(ssm_backend.cursors[ids], ref.cursors[ids])
    assert len(ssm_backend.batch_configs) == 2
