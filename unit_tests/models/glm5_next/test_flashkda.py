import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.linear import flashkda
from lightllm.common.basemodel.attention.linear.kda import KDALinearAttBackend
from lightllm.common.basemodel.attention.linear.replay_kda import ReplayKDALinearAttBackend
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda import chunk_kda_with_fused_gate
from lightllm.common.basemodel.triton_kernel.linear_att.kda_state_io import gather_kda_state, scatter_kda_state
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs


def make_model(backend="flashkda"):
    model = type("Model", (), {})()
    model.args = StartArgs(llm_prefill_att_backend=["auto", backend], mtp_step=2)
    model.config = {
        "linear_attn_config": {
            "num_heads": 4,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        }
    }
    model.data_type = torch.bfloat16
    model.tp_world_size_ = 2
    return model


@pytest.fixture
def flash_module(monkeypatch):
    module = ModuleType("flash_kda")
    module.fwd = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "flash_kda", module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    return module


@pytest.mark.parametrize("backend_cls", [KDALinearAttBackend, ReplayKDALinearAttBackend])
def test_flashkda_selected_for_native_and_replay(flash_module, backend_cls):
    backend = backend_cls(model=make_model())
    assert backend.flash_kda_fwd is flash_module.fwd
    assert backend.tp_num_heads == 2


@pytest.mark.parametrize("backends", [["auto"], ["auto", "auto"], ["auto", "triton"]])
def test_fp32_auto_and_explicit_triton_do_not_import_flashkda(monkeypatch, flash_module, backends):
    monkeypatch.setitem(sys.modules, "flash_kda", None)
    model = make_model()
    model.args.llm_prefill_att_backend = backends
    assert flashkda.get_flash_kda_fwd(model) is None


def test_missing_explicit_flashkda_reports_install_command(monkeypatch, flash_module):
    monkeypatch.setitem(sys.modules, "flash_kda", None)
    with pytest.raises(ImportError, match="pip install --no-build-isolation"):
        flashkda.get_flash_kda_fwd(make_model())


@pytest.mark.parametrize("unsupported", ["dtype", "state_dtype", "cpu", "sm80", "sm110"])
def test_auto_skips_unsupported_flashkda_configuration(monkeypatch, flash_module, unsupported):
    model = make_model("auto")
    model.args.linear_att_ssm_data_type = "bfloat16"
    if unsupported == "dtype":
        model.data_type = torch.float16
    elif unsupported == "state_dtype":
        model.args.linear_att_ssm_data_type = "float8_e4m3fn"
    elif unsupported == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    else:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0) if unsupported == "sm80" else (11, 0))
    assert flashkda.get_flash_kda_fwd(model) is None


def test_missing_optional_flashkda_falls_back(monkeypatch, flash_module):
    model = make_model("auto")
    model.args.linear_att_ssm_data_type = "bfloat16"
    monkeypatch.setitem(sys.modules, "flash_kda", None)
    assert flashkda.get_flash_kda_fwd(model) is None


@pytest.mark.parametrize("requested", ["auto", "flashkda"])
def test_extension_load_failure_is_reported_or_falls_back(monkeypatch, flash_module, requested):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "flash_kda":
            raise OSError("undefined CUDA symbol")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    model = make_model(requested)
    model.args.linear_att_ssm_data_type = "bfloat16"
    if requested == "auto":
        assert flashkda.get_flash_kda_fwd(model) is None
    else:
        with pytest.raises(ImportError, match="PyTorch/CUDA") as exc:
            flashkda.get_flash_kda_fwd(model)
        assert isinstance(exc.value.__cause__, OSError)


@pytest.mark.parametrize("backend_cls", [KDALinearAttBackend, ReplayKDALinearAttBackend])
def test_prefill_metadata_is_rebuilt_for_each_forward(flash_module, backend_cls):
    model = make_model("auto")
    model.args.linear_att_ssm_data_type = "bfloat16"
    backend = backend_cls(model=model)
    infer = type("Infer", (), {})()
    infer.b_req_idx = torch.tensor([3, 1], dtype=torch.int32)
    infer.b1_cu_q_seq_len = torch.tensor([0, 65, 68], dtype=torch.int32)
    infer.max_q_seq_len = 65
    att = backend.create_att_prefill_state(infer)
    att.init_state()
    first_cu = att.cu_seqlens
    assert att.flash_kda_fwd is flash_module.fwd
    assert first_cu.dtype == torch.int64
    torch.testing.assert_close(first_cu, infer.b1_cu_q_seq_len.long())
    infer.b1_cu_q_seq_len = torch.tensor([0, 2049, 2050], dtype=torch.int32)
    infer.max_q_seq_len = 2049
    att.init_state()
    assert att.flash_kda_fwd is None
    assert att.cu_seqlens is infer.b1_cu_q_seq_len
    torch.testing.assert_close(first_cu, torch.tensor([0, 65, 68], dtype=torch.int64))


@pytest.mark.parametrize(
    "heads,length,uses_flash",
    [(8, 2048, True), (8, 2049, False), (16, 8192, True), (64, 8192, True), (64, 8193, False)],
)
@pytest.mark.parametrize("backend_cls", [KDALinearAttBackend, ReplayKDALinearAttBackend])
def test_automatic_dispatch_accounts_for_heads_and_length(flash_module, heads, length, uses_flash, backend_cls):
    model = make_model("auto")
    model.args.linear_att_ssm_data_type = "bfloat16"
    model.config["linear_attn_config"]["num_heads"] = 64
    model.tp_world_size_ = 64 // heads
    backend = backend_cls(model=model)
    assert (flashkda.select_kda_prefill(backend, length) is flash_module.fwd) == uses_flash


def test_explicit_flashkda_overrides_performance_policy(flash_module):
    backend = KDALinearAttBackend(model=make_model())
    assert flashkda.select_kda_prefill(backend, 32768) is flash_module.fwd


def test_fp8_sparse_kv_does_not_change_kda_selection(flash_module):
    model = make_model("auto")
    model.args.linear_att_ssm_data_type = "bfloat16"
    model.args.llm_kv_type = "fp8kv_dsa"
    assert flashkda.get_flash_kda_fwd(model) is flash_module.fwd


def test_invalid_kda_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported KDA prefill backend"):
        flashkda.get_flash_kda_fwd(make_model("flashqla"))


def test_cli_accepts_flashkda_prefill():
    args = make_argument_parser().parse_args(["--llm_prefill_att_backend", "auto", "flashkda"])
    assert args.llm_prefill_att_backend == ["auto", "flashkda"]
    assert "flashkda" in StartArgs.__dataclass_fields__["llm_prefill_att_backend"].metadata["choices"]


def test_ep_startup_accepts_flashkda(monkeypatch):
    from lightllm.server import api_start

    args = StartArgs(
        enable_ep_moe=True,
        llm_prefill_att_backend=["auto", "flashkda"],
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
        page_size=0,
    )
    for name in (
        "configure_decode_open_file_limit",
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
        "_auto_set_prefill_queue_strategy",
        "auto_set_page_size",
    ):
        monkeypatch.setattr(api_start, name, lambda args: None)
    monkeypatch.setattr(api_start, "is_hybrid_att_model", lambda model_dir: True)
    monkeypatch.setattr(
        api_start.process_manager,
        "start_submodule_processes",
        lambda *args, **kwargs: pytest.fail("subprocess startup must not be reached"),
    )
    with pytest.raises(ValueError, match="--page_size"):
        api_start._launch_subprocesses(args)


def make_inputs(lengths, state_dtype, device):
    heads, dim, tokens = 2, 128, sum(lengths)
    packed = torch.randn(1, tokens, heads * (4 * dim + 1), device=device, dtype=torch.bfloat16)
    q, k, v, g, beta = packed.split([heads * dim] * 4 + [heads], dim=-1)
    return dict(
        q=q.contiguous().view(1, tokens, heads, dim),
        k=k.contiguous().view(1, tokens, heads, dim),
        v=v.contiguous().view(1, tokens, heads, dim),
        raw_g=g.contiguous().view(1, tokens, heads, dim),
        raw_beta=beta.contiguous().view(1, tokens, heads),
        A_log=torch.randn(heads, device=device) * 0.1,
        g_bias=torch.randn(heads * dim, device=device) * 0.1,
        initial_state=torch.randn(len(lengths), heads, dim, dim, device=device, dtype=state_dtype) * 0.1,
        cu_seqlens=torch.tensor([0, *lengths], device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32),
        lower_bound=-5.0,
        qk_normalized=False,
    )


def run_prefill(inputs, fwd):
    kwargs = inputs.copy()
    states = kwargs.pop("initial_state").clone()
    kwargs["cu_seqlens"] = kwargs["cu_seqlens"].to(torch.int64 if fwd is not None else torch.int32)
    output = flashkda.kda_prefill(
        **kwargs,
        ssm_states=states,
        state_indices=torch.arange(states.shape[0], device=states.device, dtype=torch.int32),
        flash_kda_fwd=fwd,
    )
    return output, states


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_flashkda_adapter_preserves_raw_inputs_and_transposes_state(cuda_flashkda, state_dtype):
    inputs = make_inputs([0, 3], state_dtype, "cuda")
    initial = inputs["initial_state"].clone()
    expected_final = torch.randn_like(initial)
    calls = []

    def fwd(**kwargs):
        calls.append(kwargs)
        for name, source in (("q", "q"), ("k", "k"), ("v", "v"), ("g", "raw_g"), ("beta", "raw_beta")):
            assert kwargs[name].is_contiguous()
            torch.testing.assert_close(kwargs[name], inputs[source], rtol=0, atol=0)
        torch.testing.assert_close(kwargs["initial_state"], initial.transpose(-1, -2), rtol=0, atol=0)
        assert kwargs["initial_state"].is_contiguous()
        assert kwargs["final_state"].dtype == state_dtype
        assert kwargs["cu_seqlens"].dtype == torch.int64
        assert kwargs["dt_bias"].shape == (2, 128)
        assert kwargs["A_log"].dtype == kwargs["dt_bias"].dtype == torch.float32
        kwargs["final_state"].copy_(kwargs["initial_state"].bfloat16())
        kwargs["final_state"][1].copy_(expected_final[1].transpose(-1, -2))
        kwargs["out"].fill_(0.5)

    output, final = run_prefill(inputs, fwd)
    assert len(calls) == 1
    assert output.shape == inputs["v"].shape
    assert torch.all(output == 0.5)
    torch.testing.assert_close(final[0], initial[0], rtol=0, atol=0)
    torch.testing.assert_close(final[1], expected_final[1], rtol=0, atol=0)
    torch.testing.assert_close(inputs["initial_state"], initial, rtol=0, atol=0)


def test_flashkda_all_empty_preserves_state():
    inputs = make_inputs([0, 0], torch.float32, "cpu")

    def fwd(**kwargs):
        pytest.fail("An empty batch must not launch FlashKDA")

    output, final = run_prefill(inputs, fwd)
    assert output.shape == (1, 0, 2, 128)
    torch.testing.assert_close(final, inputs["initial_state"], rtol=0, atol=0)


def test_flashkda_rejects_already_normalized_qk():
    inputs = make_inputs([1], torch.float32, "cpu")
    inputs["qk_normalized"] = True
    with pytest.raises(AssertionError, match="requires raw Q/K"):
        run_prefill(inputs, lambda **kwargs: None)


def triton_reference(inputs, lengths):
    output = torch.empty_like(inputs["v"])
    final = inputs["initial_state"].float().clone()
    start = 0
    for seq, length in enumerate(lengths):
        if length == 0:
            continue
        end = start + length
        expected, state = chunk_kda_with_fused_gate(
            q=inputs["q"][:, start:end],
            k=inputs["k"][:, start:end],
            v=inputs["v"][:, start:end].clone(),
            raw_g=inputs["raw_g"][:, start:end],
            beta=inputs["raw_beta"][:, start:end].float().sigmoid(),
            A_log=inputs["A_log"],
            g_bias=inputs["g_bias"],
            initial_state=inputs["initial_state"][seq : seq + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, length], device=output.device, dtype=torch.int32),
            safe_gate=True,
            lower_bound=inputs["lower_bound"],
        )
        output[:, start:end] = expected
        final[seq] = state[0]
        start = end
    return output, final


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_triton_fallback_preserves_empty_requests(cuda_flashkda, state_dtype):
    lengths = (0, 3, 129, 0)
    inputs = make_inputs(lengths, state_dtype, "cuda")
    expected, expected_state = triton_reference(inputs, lengths)
    actual, actual_state = run_prefill(inputs, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state.to(state_dtype), atol=0, rtol=0)
    for row in (0, 3):
        torch.testing.assert_close(actual_state[row], inputs["initial_state"][row], atol=0, rtol=0)


@pytest.fixture
def cuda_flashkda():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FlashKDA requires SM90+")
    torch.manual_seed(53)
    return pytest.importorskip("flash_kda").fwd


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [(1,), (63, 65), (0, 3, 129, 0), (2049, 257)])
@pytest.mark.parametrize("input_scale", [1.0, 0.001])
@pytest.mark.parametrize("lower_bound", [-5.0, -1.0])
def test_flashkda_matches_triton(cuda_flashkda, state_dtype, lengths, input_scale, lower_bound):
    inputs = make_inputs(lengths, state_dtype, "cuda")
    inputs["q"].mul_(input_scale)
    inputs["k"].mul_(input_scale)
    inputs["lower_bound"] = lower_bound
    original_state = inputs["initial_state"].clone()
    expected, expected_final = triton_reference(inputs, lengths)
    output, final = run_prefill(inputs, cuda_flashkda)
    torch.testing.assert_close(output.float(), expected.float(), atol=4e-3, rtol=3e-2)
    torch.testing.assert_close(final.float(), expected_final.float(), atol=8e-3, rtol=3e-2)
    torch.testing.assert_close(inputs["initial_state"], original_state, rtol=0, atol=0)
    for seq, length in enumerate(lengths):
        if length == 0:
            torch.testing.assert_close(final[seq], original_state[seq], rtol=0, atol=0)


def fp64_reference(inputs, lengths):
    q, k, v, g, beta, a, bias, initial = [
        inputs[name].cpu().double() for name in ("q", "k", "v", "raw_g", "raw_beta", "A_log", "g_bias", "initial_state")
    ]
    heads, dim = q.shape[-2:]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    gate = (inputs["lower_bound"] * torch.sigmoid(a.exp()[None, None, :, None] * (g + bias.view(heads, dim)))).exp()
    beta = beta.sigmoid()
    output, final = torch.empty_like(v), initial.clone()
    start = 0
    for seq, length in enumerate(lengths):
        state = initial[seq].clone()
        for token in range(start, start + length):
            state *= gate[0, token, :, :, None]
            delta = beta[0, token, :, None] * (v[0, token] - (state * k[0, token, :, :, None]).sum(-2))
            state += k[0, token, :, :, None] * delta[:, None, :]
            output[0, token] = (state * q[0, token, :, :, None]).sum(-2) * dim ** -0.5
        final[seq] = state
        start += length
    return output, final


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [(129,), (2049, 257)])
@pytest.mark.parametrize("lower_bound,gate_shift", [(0.0, 0.0), (-5.0, -8.0)])
def test_flashkda_weak_decay_matches_fp64(cuda_flashkda, state_dtype, lengths, lower_bound, gate_shift):
    inputs = make_inputs(lengths, state_dtype, "cuda")
    inputs["lower_bound"] = lower_bound
    inputs["raw_g"].add_(gate_shift)
    expected = fp64_reference(inputs, lengths)
    actual = run_prefill(inputs, cuda_flashkda)
    for result, reference in zip(actual, expected):
        error = result.cpu().double() - reference
        relative_rmse = (error.square().mean() / reference.square().mean()).sqrt()
        assert relative_rmse < 0.01
        assert error.abs().max() < 2 * torch.finfo(torch.bfloat16).eps * reference.abs().max()


@pytest.mark.parametrize("backend_cls", [KDALinearAttBackend, ReplayKDALinearAttBackend])
@pytest.mark.parametrize(
    "state_dtype,requested", [(torch.float32, "flashkda"), (torch.bfloat16, "flashkda"), (torch.bfloat16, "auto")]
)
def test_flashkda_backend_continues_cached_prefill(cuda_flashkda, backend_cls, state_dtype, requested):
    model = make_model(requested)
    model.args.linear_att_ssm_data_type = "float32" if state_dtype == torch.float32 else "bfloat16"
    backend = backend_cls(model=model)
    triton_backend = backend_cls(model=make_model("triton"))
    heads, dim, channels = backend.tp_num_heads, backend.head_dim, backend.tp_hidden_size * 3
    opts = dict(device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, **opts) * 0.125
    layer_weight = SimpleNamespace(
        get_merged_kda_conv_weight=lambda: weight,
        linear_A_log=SimpleNamespace(weight=torch.randn(heads, device="cuda") * 0.1),
        linear_dt_bias=SimpleNamespace(weight=torch.randn(heads * dim, device="cuda") * 0.1),
    )
    conv = torch.randn(5, channels, 5, **opts)
    ssm = torch.randn(15, heads, dim, dim, device="cuda", dtype=state_dtype) * 0.1
    conv_ref, ssm_ref = conv.clone(), ssm.clone()
    before = ssm.clone()

    def create_att(att_backend, conv_states, ssm_states):
        infer = type("InferState", (), {})()
        state = SimpleNamespace(get_layer_view=lambda **kwargs: (conv_states, ssm_states))
        infer.att_state_manager = SimpleNamespace(get_state=lambda **kwargs: state)
        infer.att_state_role = "main"
        infer.b_req_idx = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
        infer.b_ready_cache_len = torch.tensor([0, 7], device="cuda", dtype=torch.int32)
        att = att_backend.create_att_prefill_state(infer)
        return infer, att

    infer, att = create_att(backend, conv, ssm)
    infer_ref, att_ref = create_att(triton_backend, conv_ref, ssm_ref)
    assert backend.flash_kda_fwd is cuda_flashkda
    for lengths in [(1, 0), (65, 3), (2049, 1), (1, 129)]:
        inputs = make_inputs(lengths, state_dtype, "cuda")
        for current in (infer, infer_ref):
            current.b1_cu_q_seq_len = inputs["cu_seqlens"]
            current.max_q_seq_len = max(lengths)
        att.init_state()
        att_ref.init_state()
        if requested == "auto":
            assert (att.flash_kda_fwd is not None) == (max(lengths) <= 2048)
        params = dict(
            layer_weight=layer_weight,
            layer_num=0,
            mixed_qkv=torch.randn(sum(lengths), channels + heads + 256, **opts)[:, :channels],
            raw_gate=inputs["raw_g"],
            raw_beta=inputs["raw_beta"],
        )
        reference_params = {**params, "mixed_qkv": params["mixed_qkv"].clone()}
        expected = att_ref.prefill_att(
            None, None, None, att_control=AttControl(linear_att_prefill=True, linear_att_prefill_dict=reference_params)
        )
        actual = att.prefill_att(
            None, None, None, att_control=AttControl(linear_att_prefill=True, linear_att_prefill_dict=params)
        )
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual.float(), expected.float(), atol=4e-3, rtol=3e-2)
        torch.testing.assert_close(conv, conv_ref, atol=0, rtol=0)
        torch.testing.assert_close(ssm.float(), ssm_ref.float(), atol=8e-3, rtol=3e-2)
        for current in (infer, infer_ref):
            current.b_ready_cache_len += torch.tensor(lengths, device="cuda", dtype=torch.int32)
    untouched = torch.ones(ssm.shape[0], device="cuda", dtype=torch.bool)
    untouched[att.b_ssm_buffer_idx] = False
    torch.testing.assert_close(ssm[untouched], before[untouched], rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_kda_state_io_preserves_empty_and_unselected_slots(cuda_flashkda, state_dtype, strided):
    pool = torch.randn(10, 3, 128, 128, device="cuda", dtype=state_dtype)
    states = pool[::2] if strided else pool
    if strided:
        states = states.transpose(-1, -2)
    indices = torch.tensor([3, 1, -1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 7, 7, 10], device="cuda", dtype=torch.int64)
    before = pool.clone()
    packed = gather_kda_state(states, indices)
    torch.testing.assert_close(packed, states[indices].transpose(-1, -2), atol=0, rtol=0)
    assert packed.is_contiguous()
    packed.add_(0.125)
    expected = states.clone()
    expected[indices[[0, 2]]] = packed[[0, 2]].transpose(-1, -2)
    scatter_kda_state(packed, states, indices, cu)
    torch.testing.assert_close(states, expected, atol=0, rtol=0)
    if strided:
        torch.testing.assert_close(pool[1::2], before[1::2], atol=0, rtol=0)
