import gc
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

flashmla = pytest.importorskip("flash_mla")

from lightllm.common.basemodel.layer_weights.meta_weights import TpAttSinkWeight
from lightllm.models.deepseek_v4.attention import sparse_mla as attention
from unit_tests.common.attention.helpers import (
    backend_context,
    make_backend,
    make_batch,
    make_context,
    run_dense,
    run_sparse,
    update_context,
)


@pytest.fixture(autouse=True)
def _tp_env(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    yield
    # Collect infer-state cycles here so old graphs are not destroyed during the next capture.
    gc.collect()


def test_backend_import_without_vllm():
    script = """
import importlib.abc
import sys

class BlockVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm' or fullname.startswith('vllm.'):
            raise ModuleNotFoundError('vllm deliberately unavailable')

sys.meta_path.insert(0, BlockVllm())
from lightllm.models.deepseek_v4.attention import sparse_mla as attention
import flash_mla as flashmla
assert flashmla.__name__ == 'flash_mla'
assert not any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules)
"""
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("run_mode", ["normal", "prefill", "decode"])
def test_swa_workspace_includes_decode_padding_prefill(monkeypatch: pytest.MonkeyPatch, run_mode: str) -> None:
    backend_class = attention.DeepseekV4FlashMlaFp8SparseAttBackend
    model = SimpleNamespace(
        args=SimpleNamespace(run_mode=run_mode, enable_prefill_microbatch_overlap=False),
        config=dict(
            compress_ratios=[0, 4, 128],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        ),
        layers_infer=[SimpleNamespace(layer_num_=layer) for layer in range(3)],
        tp_rank_=0,
        tp_world_size_=4,
        batch_max_tokens=4096,
        max_req_num=8,
        att_state_role="main",
    )
    workspace, capacity = backend_class.get_swa_prefill_workspace(model=model)
    # D 节点仍执行 HOLD 请求的单 token prefill，但无需保留普通请求的长 prefill 临时区。
    assert workspace is not None
    if run_mode == "decode":
        assert capacity == 0
        assert workspace.shape == (3, 128, 585)
    else:
        assert capacity >= model.batch_max_tokens
        assert workspace.shape == (3, capacity + 128, 585)


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
    from unit_tests.common.attention.helpers import model_stub

    model = model_stub(page_size=256)
    model.config = {"num_attention_heads": heads, "compress_ratios": [0, 4, 128]}
    model.config.update(head_dim=512, qk_rope_head_dim=64, index_head_dim=128, sliding_window=128)
    model.layers_infer = [SimpleNamespace(layer_num_=i) for i in range(3)]
    model.tp_rank_ = 0
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
    attn_sink = TpAttSinkWeight(
        all_q_head_num=heads, weight_name="attn_sink", data_type=torch.float32, head_padding_multiple=64
    )
    attn_sink.load_hf_weights({"attn_sink": torch.linspace(-0.5, 0.5, heads, dtype=torch.float32)})
    params = dict(
        compress_ratio=ratio,
        head_dim_v=512,
        softmax_scale=512 ** -0.5,
        swa_indices=indices,
        swa_lengths=lengths,
        attn_sink=attn_sink.weight,
        extra_indices=extra_indices,
        extra_lengths=lengths.clone() if ratio else None,
    )
    # The attention backend receives all cache inputs through k, without a memory manager.
    state_info = SimpleNamespace()
    return backend, state_info, q, (pool, extra), decoded, extra_decoded, params


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
def test_prefill_matches_reference_and_preserves_out(monkeypatch, heads, ratio, provide_out):
    backend, state_info, q, pool, decoded, extra_decoded, params = _case(heads, ratio)
    state = backend_context(type(backend), "prefill", settings=backend)
    update_context(state, state_info)
    allocations = []

    def alloc_output(shape, **kwargs):
        buffer = torch.empty(shape, **kwargs)
        allocations.append(buffer)
        return buffer

    kernel = flashmla.flash_mla_with_kvcache

    def flashmla_with_out(**kwargs):
        assert kwargs["out"] is not None
        return kernel(**kwargs)

    monkeypatch.setattr(flashmla, "flash_mla_with_kvcache", flashmla_with_out)
    out = torch.full_like(q, float("nan")) if provide_out else None
    actual = run_sparse(state, state_info, q, pool, params, alloc_func=alloc_output, out=out)
    assert params["attn_sink"].shape == (backend.padded_q_head_num,)
    assert torch.count_nonzero(params["attn_sink"][heads:]) == 0
    assert len(allocations) == (0 if provide_out else 1)
    output_buffer = out if provide_out else allocations[0]
    assert actual.data_ptr() == output_buffer.data_ptr()
    assert actual.stride() == output_buffer.stride()
    assert actual.shape == (q.shape[0], heads, 512)
    torch.testing.assert_close(actual, _reference(q, decoded, extra_decoded, params, heads), rtol=0.02, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("overlap", [False, True])
def test_decode_cuda_graph_reads_updated_query_and_lengths(monkeypatch, heads, ratio, overlap):
    from lightllm.common.basemodel import cuda_graph as cuda_graph_module

    args = SimpleNamespace(
        enable_decode_microbatch_overlap=overlap, enable_tpsp_mix_mode=False, enable_torch_memory_saver=False
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: args)
    runner = cuda_graph_module.CudaGraph(1, 2, 1, max_batch_size=5, max_len_in_batch=16384)
    cases = []
    for _ in range(2 if overlap else 1):
        backend, state_info, q, packed_kv, decoded, extra_decoded, params = _case(heads, ratio)
        indices = torch.arange(128, dtype=torch.int32, device=q.device)
        params["swa_indices"].copy_(indices)
        params["swa_lengths"].fill_(1)
        if ratio:
            page_size = 64 if ratio == 4 else 2
            extra, extra_decoded = _pool(page_size, 128 // page_size)
            packed_kv = (packed_kv[0], extra)
            params["extra_indices"].copy_(indices)
            params["extra_lengths"].fill_(1)
        state_info.input_ids = torch.zeros(q.shape[0], dtype=torch.int64, device=q.device)
        state_info.q = q
        state_info.packed_kv = packed_kv
        state_info.att_control = params
        state_info.attention_contexts = {0: backend_context(type(backend), "decode", settings=backend)}
        state_info.attention_groups = (SimpleNamespace(backend_for=lambda phase, backend=backend: backend),)
        update_context(state_info.attention_contexts[0], state_info)
        state_info.b_req_idx = torch.arange(q.shape[0], dtype=torch.int32, device=q.device)
        from lightllm.common.attention_state import AttentionStateManager

        state_info.att_state_manager = AttentionStateManager(request_capacity=q.shape[0])
        state_info.att_state_manager.finish_state_registration()
        state_info.att_state_role = "main"
        cases.append((state_info, decoded, extra_decoded, params))

    def decode_func(*states):
        outputs = tuple(
            run_sparse(state.attention_contexts[0], state, state.q, state.packed_kv, state.att_control)
            for state in states
        )
        return outputs if overlap else outputs[0]

    # Exercise the framework's warmup + capture directly, with no operator-specific reset.
    outputs = runner.capture_decode(decode_func, *(case[0] for case in cases))
    outputs = outputs if overlap else (outputs,)
    for state_info, _, _, _ in cases:
        state = state_info.attention_contexts[0]
        eager_meta = state.sched_meta_by_ratio[ratio]
        capture_meta = state.capture_sched_meta_by_ratio[ratio]
        assert eager_meta.tile_scheduler_metadata.data_ptr() != capture_meta.tile_scheduler_metadata.data_ptr()

    # Reuse the same graph as lengths grow, shrink, and vary within the batch.
    for lengths in ([128] * 5, [64] * 5, [1] * 5, [1, 64, 65, 127, 128]):
        for index, (state_info, _, _, params) in enumerate(cases):
            state_info.q.mul_(1.25 + index * 0.1)
            new_lengths = torch.tensor(lengths, dtype=torch.int32, device=state_info.q.device)
            if index:
                new_lengths = new_lengths.flip(0)
            params["swa_lengths"].copy_(new_lengths)
            if ratio:
                params["extra_lengths"].copy_(new_lengths)
        runner.graph[5].replay()

        for actual, (state_info, decoded, extra_decoded, params) in zip(outputs, cases):
            assert actual.shape == (state_info.q.shape[0], heads, 512)
            assert actual.stride(0) == backend.padded_q_head_num * 512
            torch.testing.assert_close(
                actual, _reference(state_info.q, decoded, extra_decoded, params, heads), rtol=0.02, atol=5e-4
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [16, 64])
def test_prefill_out_supports_strided_scratch_buffer(heads):
    backend, state_info, q, pool, decoded, extra_decoded, params = _case(heads, 4)
    storage = torch.full((q.shape[0], backend.padded_q_head_num, 1024), float("nan"), dtype=q.dtype, device=q.device)
    out = storage[..., :512]
    state = backend_context(type(backend), "prefill", settings=backend)
    update_context(state, state_info)
    actual = run_sparse(state, state_info, q, pool, params, out=out)
    assert actual.data_ptr() == out.data_ptr()
    assert actual.stride() == out.stride()
    assert storage[..., 512:].isnan().all()
    torch.testing.assert_close(actual, _reference(q, decoded, extra_decoded, params, heads), rtol=0.02, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_prefill_cuda_graph_reads_new_request_state(monkeypatch, heads, ratio):
    from lightllm.common.basemodel import prefill_cuda_graph as cuda_graph_module
    from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
    from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
    from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer

    args = SimpleNamespace(
        enable_prefill_microbatch_overlap=False,
        prefill_cudagraph_max_handle_token=8,
        batch_max_tokens=8,
        enable_tpsp_mix_mode=False,
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: args)
    runner = cuda_graph_module.PrefillCudaGraph(decode_cuda_graph=None, tp_world_size=1)
    from lightllm.common.attention_state import AttentionStateManager

    manager = AttentionStateManager(request_capacity=5)
    backend, _, q, packed_kv, decoded, extra_decoded, params = _case(heads, ratio)
    indices = torch.arange(128, dtype=torch.int32, device=q.device)
    params["swa_indices"].copy_(indices)
    if ratio:
        page_size = 64 if ratio == 4 else 2
        extra, extra_decoded = _pool(page_size, 128 // page_size)
        packed_kv = (packed_kv[0], extra)
        params["extra_indices"].copy_(indices)

    def new_request(length):
        infer_state = DeepseekV4InferStateInfo()
        infer_state.att_state_manager = manager
        infer_state.input_ids = torch.zeros(q.shape[0], dtype=torch.int64, device=q.device)
        infer_state.hidden_collector = NoopHiddenCollector()
        infer_state.attn_params = dict(params, swa_lengths=torch.full_like(params["swa_lengths"], length))
        if ratio:
            infer_state.attn_params["extra_lengths"] = torch.full_like(params["extra_lengths"], length)
        infer_state.attention_contexts = {0: backend_context(type(backend), "prefill", settings=backend)}
        update_context(infer_state.attention_contexts[0], infer_state)
        return infer_state

    allocations = []
    callback_states = []
    layer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
    layer.tp_q_head_num_ = heads
    layer.head_dim_ = 512
    layer.v_head_dim = 512
    layer.compress_ratio = ratio
    layer.softmax_scale = params["softmax_scale"]
    layer.layer_num_ = 0
    layer_weight = SimpleNamespace(attn_sink_=SimpleNamespace(weight=params["attn_sink"]))

    def alloc_output(shape, **kwargs):
        buffer = torch.empty(shape, **kwargs)
        allocations.append(buffer)
        return buffer

    def compress_and_index(q_lora, x, infer_state, layer_weight):
        assert not torch.cuda.is_current_stream_capturing()
        callback_states.append(infer_state)
        return infer_state.attn_params

    layer.alloc_tensor = alloc_output
    from dataclasses import replace

    from lightllm.models.deepseek_v4.attention.sparse_mla import DeepseekV4Params
    from unit_tests.common.attention.helpers import make_layer_attention

    attention_params = DeepseekV4Params(
        packed_kv[0], packed_kv[0], packed_kv[1], ratio, 512, params["softmax_scale"], params["attn_sink"]
    )
    attention_context = backend_context(type(backend), "prefill", settings=backend)
    layer.prefill_attention = replace(
        make_layer_attention(backend, attention_context, attention_params), alloc_tensor=alloc_output
    )
    layer._compress_and_index = compress_and_index
    q_lora = torch.empty((q.shape[0], 4), dtype=q.dtype, device=q.device)
    x = torch.empty((q.shape[0], 8), dtype=q.dtype, device=q.device)

    def prefill_func(inputs, infer_state):
        graph_q = inputs[0] * 1.0
        att_out = layer._context_attention_wrapper_run(graph_q, inputs[1], inputs[2], infer_state, layer_weight)
        assert att_out.shape == (q.shape[0], heads, 512)
        assert att_out.data_ptr() == allocations[-1].data_ptr()
        assert att_out.stride(0) == backend.padded_q_head_num * 512
        # A graph segment after attention must read the output filled by the CPU callback.
        return [att_out + 0.125]

    infer_state = new_request(1)
    layer._context_attention_wrapper_run(q, q_lora, x, infer_state, layer_weight)
    outputs = runner.capture_prefill(prefill_func, [q, q_lora, x], infer_state)
    torch.testing.assert_close(
        outputs[0],
        _reference(q, decoded, extra_decoded, infer_state.attn_params, heads) + 0.125,
        rtol=0.02,
        atol=5e-4,
    )
    assert len(infer_state.prefill_cuda_graph_exe_list) == 3
    previous_meta = infer_state.attention_contexts[0].sched_meta_by_ratio[ratio]
    for length in (128, 64, 1, 64):
        q.mul_(1.25)
        infer_state = new_request(length)
        current_meta = infer_state.attention_contexts[0].sched_meta_by_ratio[ratio]
        assert current_meta.tile_scheduler_metadata is None
        actual = runner.replay([q, q_lora, x], infer_state)[0]
        assert callback_states[-1] is infer_state
        assert current_meta.tile_scheduler_metadata.data_ptr() != previous_meta.tile_scheduler_metadata.data_ptr()
        torch.testing.assert_close(
            actual,
            _reference(q, decoded, extra_decoded, infer_state.attn_params, heads) + 0.125,
            rtol=0.02,
            atol=5e-4,
        )
        previous_meta = current_meta
