from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionPhase, CacheLayout, ContextCapacity, UnsupportedAttention
from lightllm.common.attention.api.inputs import DenseInputs, DenseParams
from lightllm.common.attention.backends._workspace import AttentionWorkspace
from lightllm.common.attention.registry import cache_format_to_backend, load_backend_class
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.basemodel.attention_runtime import AttentionRuntime
from lightllm.common.basemodel.graph import GraphExecution
from lightllm.platform import get_device_runtime
from unit_tests.common.attention.helpers import make_batch, make_layer_attention, make_spec, model_stub

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(query, key, value, lengths, q_lengths, window=None):
    rows, start = [], 0
    groups = query.shape[1] // key.shape[1]
    for request, (length, q_length) in enumerate(zip(lengths, q_lengths)):
        k = key[request * 16 : request * 16 + length].float().repeat_interleave(groups, dim=1)
        v = value[request * 16 : request * 16 + length].float().repeat_interleave(groups, dim=1)
        q = query[start : start + q_length].float()
        logits = torch.einsum("qhd,khd->hqk", q, k) * query.shape[-1] ** -0.5
        causal = torch.arange(length, device=q.device)[None, :] <= (
            length - q_length + torch.arange(q_length, device=q.device)[:, None]
        )
        if window is not None:
            causal &= torch.arange(length, device=q.device)[None, :] >= (
                length - q_length + torch.arange(q_length, device=q.device)[:, None] - window[0]
            )
        probs = logits.masked_fill(~causal, -torch.inf).softmax(-1)
        rows.append(torch.einsum("hqk,khd->qhd", probs, v))
        start += q_length
    return torch.cat(rows).to(query.dtype)


@pytest.mark.parametrize("provider", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("phase", list(AttentionPhase))
def test_startup_probe_runs_the_selected_dense_implementation(provider, phase):
    from lightllm.common.attention.probe import probe_dense

    cls = load_backend_class(cache_format_to_backend["None"][provider])
    runtime = get_device_runtime()
    try:
        cls.check_environment(runtime.caps)
    except UnsupportedAttention as exc:
        pytest.skip(str(exc))
    spec = make_spec(
        phase=phase, head_size=64, value_head_size=64, cache=CacheLayout("None", 16, torch.bfloat16, 2, 64, 64)
    )
    success, error = probe_dense(cls, spec, runtime)
    assert success, error


@pytest.mark.parametrize(
    "provider,kv_heads,window",
    [
        ("triton", 2, None),
        ("fa3", 2, None),
        ("flashinfer", 2, None),
        ("triton", 4, None),
        ("triton", 4, (3, 0)),
        ("fa3", 4, (3, 0)),
    ],
)
@pytest.mark.parametrize("phase", list(AttentionPhase))
@pytest.mark.parametrize("head_dim", [64, 128])
def test_dense_provider_reference_and_decode_graph(provider, phase, head_dim, kv_heads, window):
    cls = load_backend_class(cache_format_to_backend["None"][provider])
    try:
        cls.check_environment(get_device_runtime().caps)
    except UnsupportedAttention as exc:
        pytest.skip(str(exc))
    torch.manual_seed(616)
    model = model_stub(page_size=4)
    model.config = dict(
        num_attention_heads=4, num_key_value_heads=kv_heads, hidden_size=4 * head_dim, head_dim=head_dim
    )
    model.max_seq_length, model.data_type = 16, torch.bfloat16
    spec = make_spec(
        phase=phase,
        num_q_heads=4,
        head_size=head_dim,
        value_head_size=head_dim,
        cache=CacheLayout("None", 4, torch.bfloat16, kv_heads, head_dim, head_dim),
        sliding_window=window,
    )
    backend = cls.create(model, spec)
    key, value = (torch.randn((32, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda") for _ in range(2))
    q_lengths = (2, 3) if phase is AttentionPhase.PREFILL else (1, 1)
    query = torch.randn((sum(q_lengths), 4, head_dim), dtype=torch.bfloat16, device="cuda")
    table = torch.arange(32, device="cuda", dtype=torch.int32).view(2, 16)

    def batch_for(lengths):
        offsets = [0, q_lengths[0], sum(q_lengths)]
        return make_batch(
            lengths,
            phase=phase,
            device="cuda",
            token_num=sum(q_lengths),
            max_q_seq_len=max(q_lengths),
            req_to_token_indexs=table,
            hold_req_id=1,
            b_q_seq_len=torch.tensor(q_lengths, dtype=torch.int32, device="cuda"),
            b1_cu_q_seq_len=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
            b_ready_cache_len=torch.tensor(
                [n - q for n, q in zip(lengths, q_lengths)], dtype=torch.int32, device="cuda"
            ),
        )

    batch = batch_for((7, 13))
    capture = phase is AttentionPhase.DECODE
    pool = AttentionWorkspace(runtime=model.device_runtime)
    capacity = ContextCapacity(phase, batch.batch_size, batch.token_num, 16, batch.max_q_seq_len)
    context = backend.allocate_context(capacity, capture=capture, workspace=pool.fork() if capture else pool)
    backend.prepare(context, batch)
    attention = replace(make_layer_attention(backend, context, DenseParams(key=key, value=value)), spec=spec)
    inputs = DenseInputs(query)
    output = torch.empty_like(query)

    def run():
        backend.update(context)
        return attention.backend.forward(attention, context, inputs, out=output)

    actual = run()
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, _reference(query, key, value, (7, 13), q_lengths, window), rtol=0.02, atol=0.015)
    if not capture:
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    captured_pointer = context.batch.b_seq_len.data_ptr()
    manager = AttentionStateManager(request_capacity=2)

    def prepare(source):
        context.workspace.begin()
        backend.prepare(context, source)

    state = SimpleNamespace(
        att_state_manager=manager,
        attention_groups=(SimpleNamespace(backend_for=lambda phase: backend),),
        attention_contexts={0: context},
        copy_for_cuda_graph=prepare,
    )
    state.attention_runtime = AttentionRuntime()
    state.attention_runtime.groups = state.attention_groups
    state.attention_runtime.contexts = state.attention_contexts
    execution = GraphExecution(graph, (state,), (output,), runtime=model.device_runtime)
    try:
        # Eager execution may grow the shared pool after capture. The graph must
        # keep its original pointers and fence reuse across streams.
        captured_buffers = {key: tensor.data_ptr() for key, tensor in context.workspace._buffers.items()}
        for (name, dtype), tensor in tuple(pool._buffers.items()):
            pool.get(name, tensor.numel() + 1, dtype)
        for i, lengths in enumerate(((16, 3), (2, 15), (9, 8))):
            with torch.cuda.stream(stream if i % 2 else torch.cuda.current_stream()):
                query.normal_()
                actual = execution.replay(batch_for(lengths))
                assert actual is output
                assert context.batch.b_seq_len.data_ptr() == captured_pointer
                assert {
                    key: tensor.data_ptr() for key, tensor in context.workspace._buffers.items()
                } == captured_buffers
                torch.testing.assert_close(
                    output, _reference(query, key, value, lengths, q_lengths, window), rtol=0.02, atol=0.015
                )
    finally:
        manager.close_graphs()
    assert execution.graph is None


@pytest.mark.parametrize("provider", ["fa3", "flashinfer"])
def test_noncausal_prefill_probe(provider):
    from lightllm.common.attention.probe import probe_dense

    cls = load_backend_class(cache_format_to_backend["None"][provider])
    runtime = get_device_runtime()
    cls.check_environment(runtime.caps)
    spec = make_spec(
        phase=AttentionPhase.PREFILL,
        causal=False,
        head_size=64,
        value_head_size=64,
        cache=CacheLayout("None", 16, torch.bfloat16, 2, 64, 64),
    )
    cls.check_support(spec, runtime.caps)
    success, error = probe_dense(cls, spec, runtime)
    assert success, error


def test_triton_rejects_noncausal_prefill():
    cls = load_backend_class(cache_format_to_backend["None"]["triton"])
    with pytest.raises(UnsupportedAttention, match="non-causal prefill"):
        cls.check_support(make_spec(phase=AttentionPhase.PREFILL, causal=False))


@pytest.mark.parametrize("phase", list(AttentionPhase))
@pytest.mark.parametrize("scale", [0.07, 0.13])
def test_flashinfer_mla_spec_scale(phase, scale):
    from lightllm.common.attention.api import AttentionFamily, LayerAttentionBinding
    from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAParams, MLAPrefillInputs
    from lightllm.common.attention.backends.mla.flashinfer.backend import MlaFlashInferAttBackend

    torch.manual_seed(27)
    model = model_stub(page_size=1)
    model.config = {"num_attention_heads": 8, "rope_scaling": {"factor": 8, "mscale_all_dim": 1}}
    model.qk_nope_head_dim, model.qk_rope_head_dim = 128, 64
    model.kv_lora_rank, model.v_head_dim = 512, 128
    model.max_seq_length, model.data_type = 32, torch.bfloat16
    spec = make_spec(
        family=AttentionFamily.MLA,
        phase=phase,
        num_q_heads=8,
        head_size=192,
        value_head_size=512,
        cache=CacheLayout("None", 4, torch.bfloat16, 1, 192, 512),
        softmax_scale=scale,
    )
    backend = MlaFlashInferAttBackend.create(model, spec)
    assert backend.page_size == 4 and backend.max_page_num == 8
    lengths = (7, 13)
    q_lengths = (2, 3) if phase is AttentionPhase.PREFILL else (1, 1)
    batch = make_batch(
        lengths,
        phase=phase,
        device="cuda",
        token_num=sum(q_lengths),
        max_q_seq_len=max(q_lengths),
        b_q_seq_len=torch.tensor(q_lengths, dtype=torch.int32, device="cuda"),
        b1_cu_q_seq_len=torch.tensor([0, q_lengths[0], sum(q_lengths)], dtype=torch.int32, device="cuda"),
        req_to_token_indexs=torch.arange(32, dtype=torch.int32, device="cuda").view(2, 16),
        b_ready_cache_len=torch.tensor([n - q for n, q in zip(lengths, q_lengths)], dtype=torch.int32, device="cuda"),
    )
    workspace = AttentionWorkspace(runtime=model.device_runtime)
    capacity = ContextCapacity(phase, 2, sum(q_lengths), 16, max(q_lengths))
    context = backend.allocate_context(capacity, capture=False, workspace=workspace)
    backend.prepare(context, batch)
    backend.update(context)
    if phase is AttentionPhase.PREFILL:
        query = torch.randn(sum(q_lengths), 8, 192, device="cuda", dtype=torch.bfloat16)
        key_nope = torch.randn(sum(lengths), 8, 128, device="cuda", dtype=torch.bfloat16)
        key_rope = torch.randn(sum(lengths), 1, 64, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(sum(lengths), 8, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.cat((key_nope, key_rope.expand(-1, 8, -1)), dim=-1)
        inputs = MLAPrefillInputs(query, key_nope, key_rope, value)
        cache = None
    else:
        cache = torch.randn(32, 1, 576, device="cuda", dtype=torch.bfloat16)
        query = torch.randn(2, 8, 576, device="cuda", dtype=torch.bfloat16)
        inputs = MLADecodeInputs(query[..., :512], query[..., 512:])
        key = torch.cat([cache[i * 16 : i * 16 + length].expand(-1, 8, -1) for i, length in enumerate(lengths)])
        value = key[..., :512]
    binding = LayerAttentionBinding(0, 0, spec, MLAParams(cache, 512, 64), backend, torch.empty)
    actual = binding.backend.forward(binding, context, inputs)
    expected, q_start, k_start = [], 0, 0
    for length, q_length in zip(lengths, q_lengths):
        q, k, v = (
            query[q_start : q_start + q_length].float(),
            key[k_start : k_start + length].float(),
            value[k_start : k_start + length].float(),
        )
        logits = torch.einsum("qhd,khd->hqk", q, k) * scale
        mask = (
            torch.arange(length, device="cuda")[None, :]
            <= length - q_length + torch.arange(q_length, device="cuda")[:, None]
        )
        expected.append(torch.einsum("hqk,khd->qhd", logits.masked_fill(~mask, -torch.inf).softmax(-1), v))
        q_start += q_length
        k_start += length
    torch.testing.assert_close(actual, torch.cat(expected).to(actual.dtype), atol=0.03, rtol=0.03)
