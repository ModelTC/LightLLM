"""Native GDN contracts, independent recurrence and restored-state execution."""

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.attention_state import AttentionStateManager
from lightllm.core.compute.attention.dense.backend import DenseBatchMetadata
from lightllm.core.compute.attention.gdn.backend import GatedDeltaNetBackend, GDNStaticParams
from lightllm.models.base.adapter import GDNModelAdapter
from lightllm.core.planner.planner import Planner
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec, StateSpec
from lightllm.core.compute.attention.gdn.spec import GatedDeltaNetSpec
from lightllm.core.storage.attention import ConvSSMStateStore


def test_hybrid_declarations_follow_explicit_layers_and_tp():
    config = dict(
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=512,
        num_hidden_layers=3,
        full_attention_interval=2,
        layer_types=["full_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    adapter = GDNModelAdapter(config, role="draft0", dtype=torch.bfloat16, tp_world_size=2)
    specs = adapter.attention_specs()
    assert isinstance(specs[1], GatedDeltaNetSpec)
    assert (specs[1].k_heads, specs[1].v_heads) == (2, 4)
    assert adapter.state_specs()[1].shape == (4, 8, 64, 128, 3)
    assert [s.retention for s in adapter.state_specs()] == ["history", "recurrent", "history"]
    assert all(s.role == "draft0" for s in adapter.state_specs())
    config["layer_types"][1] = "full_attention"
    assert isinstance(adapter.attention_specs()[1], GatedDeltaNetSpec)


def test_draft_and_loop_layers_keep_their_own_numbering():
    from lightllm.models.qwen3next.config import is_linear_attention_layer
    from lightllm.models.sensenova_loop.layer_weights.transformer_layer_weight import (
        SenseNovaLoopTransformerLayerWeight,
    )

    assert not is_linear_attention_layer(dict(num_hidden_layers=1, layer_types=["full_attention"]), 24)
    config = dict(
        loop_physical_indices=[0, 0, 1, 2],
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    )
    assert not SenseNovaLoopTransformerLayerWeight.is_linear_layer(config, 1)
    assert SenseNovaLoopTransformerLayerWeight.is_linear_layer(config, 2)


@pytest.mark.parametrize("bits", [4, 8])
def test_hybrid_quantized_storage_matches_external_declaration(bits, monkeypatch):
    from lightllm.core.storage.attention import DenseStateStore
    from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
    from lightllm.server.core.objs.start_args_type import StartArgs
    from unit_tests.common.execution.test_group_quantized import make_format

    config = dict(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        hidden_size=256,
        num_hidden_layers=2,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    adapter = GDNModelAdapter(config, role="main", dtype=torch.float16)
    args = StartArgs(
        data_type="float16",
        llm_kv_type=f"int{bits}kv",
        llm_kv_quant_group_size=32,
        page_size=4,
        cpu_cache_token_page_size=4,
    )
    monkeypatch.setattr(
        "lightllm.common.attention_state.full_att.gqa.group_quantized_state.get_env_start_args", lambda: args
    )
    (external,) = Qwen3NextTpPartModel.create_full_att_states(config, args, model_logical_layer_ids=(0, 1))
    manager = AttentionStateManager(2)
    DenseStateStore(
        [s for s in adapter.state_specs() if s.retention == "history"], make_format(bits, torch.float16, 32), manager
    )
    manager.finish_state_registration()
    (native,) = manager.get_full_attention_states()
    assert native.config == external.config
    assert native.config.group_size == external.config.group_size == 32
    assert native.get_cpu_transfer_regions() == external.get_cpu_transfer_regions()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation,dim", [("triton", 32), ("triton", 128), ("flashqla", 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_gdn_prefix_state_restore_and_graph(implementation, dim, dtype, ssm_dtype, monkeypatch):
    torch.manual_seed(47)
    spec = GatedDeltaNetSpec(2, 4, dim, dim, 4, dtype)
    fmt = FormatSpec("conv_ssm", dtype, "request_conv_ssm", ("conv", "ssm"), 1, field_dtypes=(dtype, ssm_dtype))
    declaration = StateSpec("main", "gdn.1", 1, (1,), (2, 4, dim, dim, 3), "recurrent")
    manager = AttentionStateManager(4)
    store = ConvSSMStateStore((declaration,), fmt, manager, 4, "cuda")
    view = store.views()[declaration.key]
    conv, ssm = view.fields
    conv.normal_(std=0.1)
    ssm.normal_(std=0.05)
    params = GDNStaticParams(
        spec,
        torch.randn(spec.conv_channels, 4, device="cuda", dtype=dtype) * 0.2,
        None,
        torch.randn(4, device="cuda") * 0.1,
        torch.randn(4, device="cuda") * 0.1,
    )
    planner = Planner(
        tuple(GatedDeltaNetBackend(name, phase) for name, phase in ((implementation, "prefill"), ("triton", "decode"))),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(0, 4, 256, 2),
    )
    plan = planner.bind(
        [(1, phase, spec, view, params, "auto") for phase in ("prefill", "decode")],
        lambda _: pytest.fail("unexpected scratch allocation"),
    )
    prefill, decode = [b.execution for b in plan.bindings]
    monkeypatch.setattr(manager, "get_state", lambda *a, **k: pytest.fail("runtime state lookup"))
    expected_conv, expected_ssm = conv.clone(), ssm.clone()

    def metadata(lengths, rows, ready, phase, lane=0):
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        tensor = lambda v: torch.tensor(v, device="cuda", dtype=torch.int32)
        batch = DenseBatchMetadata(
            len(rows),
            max(lengths),
            max(lengths),
            lane,
            None,
            tensor(rows),
            tensor(lengths),
            tensor(cu[:-1]),
            tensor(ready),
            tensor(cu),
            tensor(cu),
        )
        return (prefill if phase == "prefill" else decode).create_metadata(batch)

    def reference(inputs, lengths, rows, ready):
        outputs, pos = [], 0
        for row, length, prefix in zip(rows, lengths, ready):
            history = (
                expected_conv[row].float() if prefix else torch.zeros_like(expected_conv[row], dtype=torch.float32)
            )
            state = expected_ssm[row].float() if prefix else torch.zeros_like(expected_ssm[row], dtype=torch.float32)
            for raw in inputs[pos : pos + length]:
                raw_qkv, _, b, a = raw.split((spec.conv_channels, 4 * dim, 4, 4))
                window = torch.cat((history, raw_qkv.float()[:, None]), dim=1)
                convolved = F.silu((window * params.conv_weight.float()).sum(-1)).to(dtype).float()
                history = window[:, 1:]
                q, k, v = convolved.split((2 * dim, 2 * dim, 4 * dim))
                q, k = [t.view(2, dim).repeat_interleave(2, dim=0) for t in (q, k)]
                q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
                k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
                decay = torch.exp(-params.a_log.exp() * F.softplus(a.float() + params.dt_bias))
                state = state * decay[:, None, None]
                residual = v.view(4, dim) - (state * k[:, :, None]).sum(1)
                state = state + k[:, :, None] * (residual * b.float().sigmoid()[:, None])[:, None, :]
                outputs.append((state * (q * dim ** -0.5)[:, :, None]).sum(1))
            pos += length
            expected_conv[row] = history.to(dtype)
            expected_ssm[row] = state.to(ssm_dtype)
        return torch.stack(outputs)

    def check(output, target):
        torch.testing.assert_close(output.float(), target, rtol=3e-2, atol=3e-3)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(ssm.float(), expected_ssm.float(), rtol=3e-2, atol=4e-3)

    rows, lengths, ready = [2, 0], [67, 35], [5, 0]
    inputs = torch.randn(sum(lengths), spec.input_width, dtype=dtype, device="cuda") * 0.3
    target = reference(inputs, lengths, rows, ready)
    output, z = prefill.run(inputs, metadata(lengths, rows, ready, "prefill"))
    check(output, target)
    assert torch.equal(z.reshape(len(inputs), -1), inputs[:, spec.conv_channels : spec.conv_channels + 4 * dim])

    # Checkpoint restoration keeps the bound views valid while restoring both fields.
    state = store.resources[declaration.key][1]
    state.initialize_page_pools(2, device="cuda", pin_memory=False)
    state.save_bounded_att_runtime_to_pages(rows, [72, 35], "big", [0, 1])
    conv.zero_()
    ssm.zero_()
    state.restore_bounded_att_runtime_from_pages(rows, [72, 35], "big", [0, 1])
    expected_conv[1].zero_()
    expected_conv[3].zero_()
    expected_ssm[1].zero_()
    expected_ssm[3].zero_()

    inputs = torch.randn(2, spec.input_width, dtype=dtype, device="cuda") * 0.3
    meta = metadata([1, 1], rows, [72, 35], "decode")
    target = reference(inputs, [1, 1], rows, [72, 35])
    output, _ = decode.run(inputs, meta)
    check(output, target)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = decode.run(inputs, meta)
    rows = [0, 2]
    meta.copy_for_graph(metadata([1, 1], rows, [36, 73], "decode"))
    inputs.add_(0.01)
    target = reference(inputs, [1, 1], rows, [36, 73])
    graph.replay()
    check(captured, target)

    streams = [torch.cuda.Stream() for _ in range(2)]
    batches = [
        (torch.randn(1, spec.input_width, dtype=dtype, device="cuda") * 0.3, metadata([1], [row], [74], "decode", lane))
        for lane, row in enumerate(rows)
    ]
    targets = [reference(single, [1], [row], [74]) for (single, _), row in zip(batches, rows)]
    outputs = []
    for stream, (single, meta) in zip(streams, batches):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result, _ = decode.run(single, meta)
            outputs.append(result)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for output, target in zip(outputs, targets):
        check(output, target)
