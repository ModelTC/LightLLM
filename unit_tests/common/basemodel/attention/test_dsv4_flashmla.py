import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flash_mla")

from lightllm.common.basemodel.attention.nsa import dsv4_fp8_flashmla_sparse as attention


def test_backend_import_without_vllm():
    script = """
import importlib.abc
import sys

class BlockVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm' or fullname.startswith('vllm.'):
            raise ModuleNotFoundError('vllm deliberately unavailable')

sys.meta_path.insert(0, BlockVllm())
from lightllm.common.basemodel.attention.nsa import dsv4_fp8_flashmla_sparse as attention
assert attention.flashmla.__name__ == 'flash_mla'
assert not any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules)
"""
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _pool(page_size, pages):
    stride = ((page_size * 584 + 575) // 576) * 576
    buffer = torch.zeros((pages, stride), dtype=torch.uint8, device="cuda")
    rows = torch.randn((pages, page_size, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    nope = rows[..., :448].to(torch.float8_e4m3fn)
    data = buffer[:, : page_size * 576].view(pages, page_size, 576)
    data[..., :448] = nope.view(torch.uint8)
    data[..., 448:] = rows[..., 448:].contiguous().view(torch.uint8)
    buffer[:, page_size * 576 : page_size * 584].view(pages, page_size, 8)[..., :7] = 127
    decoded = torch.cat((nope.float(), rows[..., 448:].float()), dim=-1).view(-1, 512)
    return buffer, decoded


def _case(heads, ratio):
    torch.manual_seed(6014)
    tokens = 5
    model = type("TestModel", (), {})()
    model.config = {"num_attention_heads": heads, "compress_ratios": [0, 4, 128]}
    model.tp_world_size_ = 1
    backend = attention.DeepseekV4FlashMlaFp8SparseAttBackend(model=model)
    q = torch.randn((tokens, backend.padded_q_head_num, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    q[:, heads:] = 0
    pool, decoded = _pool(128, 2)
    extra, extra_decoded = _pool(64 if ratio == 4 else 2, 3) if ratio else (None, None)
    indices = torch.full((tokens, 1, 128), -1, dtype=torch.int32, device="cuda")
    indices[:, 0, :4] = torch.tensor([0, 3, -1, 7], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([1, 2, 3, 4, 4], dtype=torch.int32, device="cuda")
    extra_indices = indices.clone() if ratio else None
    if ratio:
        extra_indices[:, 0, :4] = torch.tensor([0, 1, -1, 5], dtype=torch.int32, device="cuda")
    params = dict(
        layer_index=0,
        compress_ratio=ratio,
        head_dim_v=512,
        softmax_scale=512 ** -0.5,
        swa_indices=indices,
        swa_lengths=lengths,
        attn_sink=torch.linspace(-0.5, 0.5, heads, dtype=torch.float32, device="cuda"),
        extra_indices=extra_indices,
        extra_lengths=lengths.clone() if ratio else None,
    )
    state_info = SimpleNamespace(mem_manager=SimpleNamespace(get_compressed_kv_buffer=lambda layer: extra))
    return backend, state_info, q, pool, decoded, extra_decoded, params


def _reference(q, decoded, extra_decoded, params, heads):
    outputs = []
    for row in range(q.shape[0]):
        indexes = params["swa_indices"][row, 0, : params["swa_lengths"][row].item()].long()
        selected = [decoded[indexes[indexes >= 0]]]
        if extra_decoded is not None:
            indexes = params["extra_indices"][row, 0, : params["extra_lengths"][row].item()].long()
            selected.append(extra_decoded[indexes[indexes >= 0]])
        kv = torch.cat(selected)
        logits = q[row, :heads].float() @ kv.T * params["softmax_scale"]
        logits = torch.cat((logits, params["attn_sink"][:heads, None]), dim=-1)
        weights = logits.softmax(-1)[:, :-1]
        outputs.append((weights @ kv).to(q.dtype))
    return torch.stack(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [16, 64, 128])
@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("provide_out", [False, True])
def test_prefill_matches_reference_and_preserves_out(heads, ratio, provide_out):
    backend, state_info, q, pool, decoded, extra_decoded, params = _case(heads, ratio)
    state = backend.create_att_prefill_state(state_info)
    state.init_state()
    out = torch.full((q.shape[0], heads, 512), float("nan"), dtype=q.dtype, device=q.device) if provide_out else None
    actual = state.prefill_att(
        q,
        pool,
        None,
        att_control=attention.AttControl(nsa_prefill=True, nsa_prefill_dict=params),
        out=out,
    )
    assert params["attn_sink"].shape == (heads,)
    adapted_sink = backend.prepare_attn_sink(params["attn_sink"])
    assert adapted_sink.shape == (backend.padded_q_head_num,)
    torch.testing.assert_close(adapted_sink[:heads], params["attn_sink"])
    if heads < backend.padded_q_head_num:
        assert torch.count_nonzero(adapted_sink[heads:]) == 0
        assert backend.prepare_attn_sink(params["attn_sink"]).data_ptr() == adapted_sink.data_ptr()
    if provide_out:
        assert actual is out
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, _reference(q, decoded, extra_decoded, params, heads), rtol=0.02, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_decode_cuda_graph_reads_updated_query_and_lengths(heads, ratio):
    backend, state_info, q, pool, decoded, extra_decoded, params = _case(heads, ratio)
    state = backend.create_att_decode_state(state_info)
    state.init_state()
    control = attention.AttControl(nsa_decode=True, nsa_decode_dict=params)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        state.decode_att(q, pool, None, att_control=control)
    torch.cuda.current_stream().wait_stream(stream)
    # Match CudaGraph.capture_decode: rebuild scheduling on each graph replay.
    state.reset_sched_meta_for_capture()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = state.decode_att(q, pool, None, att_control=control)
    q.mul_(1.25)
    params["swa_lengths"].fill_(1)
    if ratio:
        params["extra_lengths"].fill_(1)
    graph.replay()
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, _reference(q, decoded, extra_decoded, params, heads), rtol=0.02, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_upstream_without_out_argument_remains_usable(monkeypatch):
    kernel = attention.flashmla.flash_mla_with_kvcache

    def upstream(**kwargs):
        assert "out" not in kwargs
        return kernel(**kwargs)

    monkeypatch.setattr(attention.flashmla, "flash_mla_with_kvcache", upstream)
    backend, state_info, q, pool, decoded, extra_decoded, params = _case(64, 4)
    assert not backend.supports_out
    state = backend.create_att_prefill_state(state_info)
    state.init_state()
    out = torch.empty((q.shape[0], 64, 512), dtype=q.dtype, device=q.device)
    actual = state.prefill_att(
        q, pool, None, out=out, att_control=attention.AttControl(nsa_prefill=True, nsa_prefill_dict=params)
    )
    assert actual is out
    torch.testing.assert_close(actual, _reference(q, decoded, extra_decoded, params, 64), rtol=0.02, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_out_supports_strided_scratch_buffer():
    backend, state_info, q, pool, decoded, extra_decoded, params = _case(64, 4)
    storage = torch.full((q.shape[0], 64, 1024), float("nan"), dtype=q.dtype, device=q.device)
    out = storage[..., :512]
    state = backend.create_att_prefill_state(state_info)
    state.init_state()
    actual = state.prefill_att(
        q, pool, None, out=out, att_control=attention.AttControl(nsa_prefill=True, nsa_prefill_dict=params)
    )
    assert actual is out
    assert storage[..., 512:].isnan().all()
    torch.testing.assert_close(actual, _reference(q, decoded, extra_decoded, params, 64), rtol=0.02, atol=5e-4)
