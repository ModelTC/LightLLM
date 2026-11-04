from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.attention_state import AttentionStateManager
from lightllm.core.compute.attention.dense.backend import DenseStaticParams
from lightllm.core.planner.planner import Planner
from lightllm.common.execution.runner import attention_batch
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec, StateSpec
from lightllm.core.storage.attention import DenseStateStore
from lightllm.core.compute.attention.dense.static_fp8 import StaticFP8AttentionBackend
from lightllm.core.formats.kv.static_fp8 import StaticFP8KVWriter


def fp8_format(per_head, page=4):
    return FormatSpec(
        "fp8_per_head" if per_head else "fp8_per_tensor",
        torch.float8_e4m3fn,
        "token_head_dim",
        ("k", "v", "kv_scale", "q_scale"),
        page,
        calibration="test-calibration" if per_head else None,
        scale_dtype=torch.float32,
    )


def test_fp8_rejects_missing_calibration_and_incompatible_device():
    backend = StaticFP8AttentionBackend("decode")
    spec, format = DenseAttentionSpec(8, 2, 64, dtype=torch.bfloat16), fp8_format(True)
    caps, limits = DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(4096, 2, 128)
    assert backend.probe(spec, format, caps, limits).offers
    assert backend.probe(spec, replace(format, calibration=None), caps, limits).reason
    assert backend.probe(spec, format, replace(caps, capability=(8, 0)), limits).reason


def test_auto_per_tensor_selects_native_flashinfer_on_other_devices(monkeypatch):
    from lightllm.core.compute.attention.dense.flashinfer import FlashInferAttentionBackend

    monkeypatch.setattr("lightllm.core.compute.attention.dense.flashinfer.find_spec", lambda name: object())
    planner = Planner(
        (StaticFP8AttentionBackend("decode"), FlashInferAttentionBackend("decode")),
        DeviceCaps("cuda", 0, (8, 9)),
        ExecutionLimits(256 * 1024 ** 2, 2, 128),
    )
    backend, _ = planner.select(DenseAttentionSpec(8, 2, 64, dtype=torch.bfloat16), fp8_format(False), "decode")
    assert isinstance(backend, FlashInferAttentionBackend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,per_head,static_q",
    [("fa3", False, False), ("fa3", True, False), ("fa3", True, True), ("flashinfer", False, False)],
)
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page", [1, 4])
@pytest.mark.parametrize("window", [(-1, -1), (63, 0)])
def test_bound_fp8_matches_decoded_sdpa_and_replays_graph(
    implementation, per_head, static_q, phase, dtype, page, window
):
    from lightllm.core.compute.attention.dense.flashinfer import FlashInferAttentionBackend

    torch.manual_seed(40)
    spec = DenseAttentionSpec(8, 2, 64, dtype=dtype, window=window)
    state_spec = StateSpec("main", "kv.0", 0, (0,), (2, 64))
    manager = AttentionStateManager(2)
    store = DenseStateStore((state_spec,), fp8_format(per_head, page), manager)
    manager.finish_state_registration()
    state = manager.get_full_attention_states()[0]
    scales = [0.015625, 0.0078125, 0.0078125, 0.015625] if per_head else [0.015625, 0.0078125]
    calibration = dict(num_layers=1, num_head=2, scales=[scales], scales_shape=[1, len(scales)])
    if static_q:
        calibration["q_calibration"] = dict(num_head=2, scales_shape=[1, 2], scales=[[0.015625, 0.0078125]])
    state.bind_calibration(calibration, 0)
    state.device = torch.device("cuda")
    state.allocate_storage(2048)
    view = store.views()[state_spec.key]
    backend = FlashInferAttentionBackend(phase) if implementation == "flashinfer" else StaticFP8AttentionBackend(phase)
    planner = Planner((backend,), DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(512 * 1024 ** 2, 2, 768, 2))
    alloc = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda")
    bound = (
        planner.bind(
            [(0, phase, spec, view, DenseStaticParams(spec, alloc), implementation)], lambda n: alloc(n, torch.uint8)
        )
        .bindings[0]
        .execution
    )
    writer = StaticFP8KVWriter(view)
    ids = torch.randperm(2048, device="cuda")[:1536].int()
    table = ids.reshape(2, 768)
    kv = torch.randn(1536, 4, 64, dtype=dtype, device="cuda")
    writer(token_indices=ids, kv=kv)
    width = 2 if per_head else 1
    k_scale, v_scale = (v.expand(2) for v in view.fields[2].split(width))
    for values, payload, scale in ((kv[:, :2], bound.k, k_scale), (kv[:, 2:], bound.v, v_scale)):
        encoded = (values.float() / scale[None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        # Indexing native FP8 tensors is not available in every PyTorch build.
        assert torch.equal(payload.view(torch.uint8)[ids], encoded.view(torch.uint8))
    key = bound.k.float() * k_scale[None, :, None]
    value = bound.v.float() * v_scale[None, :, None]
    q_len = 3 if phase == "prefill" else 1

    def metadata(lengths, lane):
        return bound.create_metadata(
            attention_batch(
                SimpleNamespace(
                    batch_size=2,
                    max_q_seq_len=q_len,
                    max_kv_seq_len=max(lengths),
                    microbatch_index=lane,
                    req_manager=SimpleNamespace(req_to_token_indexs=table),
                    b_req_idx=torch.arange(2, device="cuda", dtype=torch.int32),
                    b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
                    b_q_start_loc=torch.tensor([0, q_len], device="cuda", dtype=torch.int32),
                    b_ready_cache_len=torch.tensor(lengths, device="cuda", dtype=torch.int32) - q_len,
                    b1_cu_q_seq_len=torch.tensor([0, q_len, 2 * q_len], device="cuda", dtype=torch.int32),
                    b1_cu_kv_seq_len=torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32),
                )
            )
        )

    def reference(q, lengths):
        outputs = []
        for i, length in enumerate(lengths):
            grouped = q[i * q_len : (i + 1) * q_len].float().reshape(q_len, 2, -1)
            if implementation == "flashinfer":
                decoded_q = grouped.reshape(q_len, 8, 64)
            else:
                scale = bound.q_scale if static_q else grouped.abs().amax((0, 2)) / 448
                decoded_q = (grouped / scale[None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                decoded_q = (decoded_q * scale[None, :, None]).reshape(q_len, 8, 64)
            distance = (
                torch.arange(length - q_len, length, device="cuda")[:, None]
                - torch.arange(length, device="cuda")[None, :]
            )
            mask = distance >= 0
            if window != (-1, -1):
                mask &= distance <= window[0]
            outputs.append(
                F.scaled_dot_product_attention(
                    decoded_q.transpose(0, 1),
                    key[table[i, :length]].transpose(0, 1),
                    value[table[i, :length]].transpose(0, 1),
                    attn_mask=mask,
                    enable_gqa=True,
                ).transpose(0, 1)
            )
        return torch.cat(outputs)

    lengths = [513, 65]
    metas = [metadata(lengths, lane) for lane in range(2)]
    tables = [meta.indices if implementation == "flashinfer" else meta.page_table for meta in metas]
    assert tables[0].data_ptr() != tables[1].data_ptr()
    q = torch.randn(2 * q_len, 8, 64, dtype=dtype, device="cuda")

    def check(output):
        target = reference(q, lengths).float()
        if implementation == "flashinfer":
            torch.testing.assert_close(output.float(), target, rtol=2e-2, atol=5e-3)
            return
        error = output.float() - target
        # FA3's FP8 computation adds error even when given these exact decoded inputs.
        assert (error.square().mean() / target.square().mean()).sqrt().item() < 0.035
        fp8_half_ulp = torch.finfo(torch.float8_e4m3fn).eps / 2
        assert error.abs().max().item() < fp8_half_ulp * target.abs().max().item()

    output = bound.run(q, metas[0])
    check(output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = bound.run(q, metas[0])
    lengths = [517, 68]
    metas[0].copy_for_graph(metadata(lengths, 0))
    q.add_(0.1)
    graph.replay()
    check(output)
