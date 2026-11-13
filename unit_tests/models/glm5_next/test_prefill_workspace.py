from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.attention.nsa import fp8_glm5_next, nvfp4_glm5_next
from lightllm.models.glm5_next.model import Glm5NextTpPartModel
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from unit_tests.models.glm5_next.test_fp8_kv import _PrefillCase, _make_workspace_backend, _capture_cuda_graph


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def current_cuda_device(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("run_mode", ["normal", "decode"])
@pytest.mark.parametrize("kv_type", ["fp8", "nvfp4"])
def test_prefill_workspace_is_reserved_once_and_reused_at_runtime(monkeypatch, overlap, run_mode, kv_type):
    model = SimpleNamespace(
        args=SimpleNamespace(enable_prefill_microbatch_overlap=overlap, run_mode=run_mode),
        batch_max_tokens=8,
        max_req_num=2,
        max_seq_length=13,
        config={"index_topk": 2048},
        att_state_role=f"workspace_test_{run_mode}_{overlap}",
    )
    backend_class = (
        fp8_glm5_next.Fp8Glm5NextSparseAttBackend if kv_type == "fp8" else nvfp4_glm5_next.NVFP4Glm5NextSparseAttBackend
    )
    reserved = backend_class.allocate_prefill_workspaces(model)
    pointers = [{key: value.data_ptr() for key, value in buffers.items()} for buffers in reserved]
    backend = object.__new__(backend_class)
    backend.__init__(model)
    assert [
        {key: value.data_ptr() for key, value in buffers.items()} for buffers in backend._prefill_workspaces
    ] == pointers
    assert len(reserved) == 1 + int(overlap)
    capacity = 8 if run_mode == "decode" else 34
    if kv_type == "nvfp4":
        capacity = max(capacity, 8 * 2176)
    assert reserved[0]["gathered_kv"].shape == (capacity, 1, 512)
    assert reserved[0]["chunk_topk_indices"].shape == (8, 2176)

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime must reuse the workspace reserved before KV capacity planning")

    monkeypatch.setattr(backend, "allocate_prefill_workspaces", forbidden)
    monkeypatch.setattr(backend, "get_gpu_workspace_buffer", forbidden)
    for slot in range(1 + int(overlap)):
        for q_tokens in (1, 8, 3):
            topk = torch.empty((q_tokens, 2176), dtype=torch.int32, device="cuda")
            views = backend.get_prefill_workspace(SimpleNamespace(microbatch_index=slot), topk)
            assert {key: value.data_ptr() for key, value in views.items()} == pointers[slot]
            torch.testing.assert_close(
                views["cu_seqlens_q"], torch.arange(q_tokens + 1, dtype=torch.int32, device="cuda")
            )


def test_prefill_microbatches_use_independent_scratch_during_graph_replay(monkeypatch):
    monkeypatch.setattr(fp8_glm5_next, "KV_GATHER_CHUNK_SIZE", 7)
    backend = _make_workspace_backend(8, overlap=True)
    first = _PrefillCase([15], [3], 8, backend=backend)
    second = _PrefillCase([17], [3], 8, backend=backend)
    second.infer_state.microbatch_index = 1
    first.check(first.run())
    second.check(second.run())
    pointers = [{key: value.data_ptr() for key, value in buffers.items()} for buffers in backend._prefill_workspaces]
    assert all(pointers[0][key] != pointers[1][key] for key in pointers[0])

    def forbidden(*args, **kwargs):
        raise AssertionError("Attention must not allocate a new prefill workspace")

    monkeypatch.setattr(backend, "allocate_prefill_workspaces", forbidden)
    first_graph, first_output = _capture_cuda_graph(first.run)
    second_graph, second_output = _capture_cuda_graph(second.run)
    first_stream, second_stream = torch.cuda.Stream(), torch.cuda.Stream()
    first.mutate()
    second.mutate()
    for _ in range(3):
        first_stream.wait_stream(torch.cuda.current_stream())
        second_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(first_stream):
            first_graph.replay()
        with torch.cuda.stream(second_stream):
            second_graph.replay()
        torch.cuda.current_stream().wait_stream(first_stream)
        torch.cuda.current_stream().wait_stream(second_stream)
        first.check(first_output)
        second.check(second_output)
    assert [
        {key: value.data_ptr() for key, value in buffers.items()} for buffers in backend._prefill_workspaces
    ] == pointers


@pytest.mark.parametrize("rows", [3, 17])
def test_nvfp4_prefill_microbatches_replay_with_independent_scratch(monkeypatch, rows):
    from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4_mla import quantize_nvfp4_mla
    from lightllm.models.glm5_next.triton_kernel.nvfp4_prefill import nvfp4_cached_prefill
    from unit_tests.common.att_resources.test_nvfp4_mla import dequantize, reference

    torch.manual_seed(5300 + rows)
    model = SimpleNamespace(
        args=SimpleNamespace(enable_prefill_microbatch_overlap=True, run_mode="normal"),
        batch_max_tokens=17,
        max_req_num=1,
        max_seq_length=1024,
        config={"index_topk": 2048},
        att_state_role=f"nvfp4_overlap_{rows}",
    )
    backend = object.__new__(nvfp4_glm5_next.NVFP4Glm5NextSparseAttBackend)
    backend.__init__(model)
    pointers = [{key: value.data_ptr() for key, value in buffers.items()} for buffers in backend._prefill_workspaces]
    assert all(pointers[0][key] != pointers[1][key] for key in pointers[0])
    cases = []
    for slot in range(2):
        raw = torch.randn(1024, 1, 512, dtype=torch.bfloat16, device="cuda")
        packed = torch.empty(1024, 1, 288, dtype=torch.uint8, device="cuda")
        quantize_nvfp4_mla(raw, torch.arange(1024, device="cuda", dtype=torch.int32), packed)
        ragged = torch.randperm(1024, device="cuda", dtype=torch.int32)[:769]
        q = torch.randn(rows, 8, 512, dtype=torch.bfloat16, device="cuda")
        relative = torch.randint(0, 769, (rows, 2176), dtype=torch.int32, device="cuda")
        relative[0] = -1
        current_slots = ragged[-7:].long()
        current = raw[current_slots].clone().add_(0.125)
        row_table = torch.full((1024,), -1, dtype=torch.int32, device="cuda")
        row_table[current_slots] = torch.arange(7, device="cuda", dtype=torch.int32)
        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        workspace = backend.get_prefill_workspace(SimpleNamespace(microbatch_index=slot), relative)

        def run(
            q=q,
            packed=packed,
            ragged=ragged,
            relative=relative,
            current=current,
            row_table=row_table,
            starts=starts,
            workspace=workspace,
        ):
            physical = ragged[relative.clamp_min(0).long()].masked_fill(relative < 0, -1)
            return nvfp4_cached_prefill(
                q, packed, ragged, starts, relative, physical, 0.0625, current, row_table, workspace=workspace
            )

        graph, output = _capture_cuda_graph(run)
        # Keep every captured input alive, including starts and row_table held
        # by run's defaults; CUDAGraph retains addresses rather than tensors.
        cases.append((q, packed, ragged, relative, current_slots, current, graph, output, torch.cuda.Stream(), run))

    def forbidden(*args, **kwargs):
        raise AssertionError("Graph replay must reuse the reserved per-microbatch workspace")

    monkeypatch.setattr(backend, "allocate_prefill_workspaces", forbidden)
    monkeypatch.setattr(backend, "get_gpu_workspace_buffer", forbidden)
    for _ in range(3):
        for q, packed, ragged, relative, current_slots, current, graph, output, stream, _run in cases:
            q.normal_()
            current.normal_()
            relative[1:].random_(0, 769)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        for q, packed, ragged, relative, current_slots, current, graph, output, stream, _run in cases:
            torch.cuda.current_stream().wait_stream(stream)
            decoded = dequantize(packed)
            decoded[current_slots] = current
            physical = ragged[relative.clamp_min(0).long()].masked_fill(relative < 0, -1)
            torch.testing.assert_close(output, reference(q, decoded, physical, 0.0625), rtol=0.02, atol=0.016)
    assert [
        {key: value.data_ptr() for key, value in buffers.items()} for buffers in backend._prefill_workspaces
    ] == pointers


@pytest.mark.parametrize("model_class", [Glm5NextTpPartModel, Glm5NextMTPModel])
@pytest.mark.parametrize("kv_type", ["None", "fp8kv_dsa", "nvfp4kv"])
def test_glm_reserves_prefill_workspace_during_resource_preparation(monkeypatch, model_class, kv_type):
    model = object.__new__(model_class)
    model.args = SimpleNamespace(llm_kv_type=kv_type)
    model.mtp_previous_draft_models = []
    # Whole-state queries are invalid while the shared manager is registering.
    model.att_state_manager = SimpleNamespace(has_full_attention_states=lambda role: True)
    calls = []

    def prepare(self):
        assert self is model
        calls.append("prepare_states")

    def reserve(cls, prepared_model):
        assert prepared_model is model
        calls.append("reserve_workspace")

    monkeypatch.setattr(TpPartBaseModel, "prepare_attention_states", prepare)
    monkeypatch.setattr(fp8_glm5_next.Fp8Glm5NextSparseAttBackend, "allocate_prefill_workspaces", classmethod(reserve))
    monkeypatch.setattr(
        nvfp4_glm5_next.NVFP4Glm5NextSparseAttBackend, "allocate_prefill_workspaces", classmethod(reserve)
    )
    model.prepare_attention_states()
    assert calls == (
        ["prepare_states", "reserve_workspace"] if kv_type in ("fp8kv_dsa", "nvfp4kv") else ["prepare_states"]
    )


def test_full_state_presence_is_available_during_registration():
    from lightllm.common.attention_state.manager import AttentionStateManager
    from lightllm.common.attention_state.full_att import PackedMLALogicalConfig, create_packed_mla_state

    manager = AttentionStateManager(request_capacity=3)
    assert not manager.has_full_attention_states("main")
    state = create_packed_mla_state(
        config=PackedMLALogicalConfig(
            logical_layer_groups=((0,),), latent_dim=512, format="packed_nvfp4", rope_dim=0, indexer_pool_size=4
        ),
        page_size=16,
        device="meta",
    )
    manager.register_state("main", state)
    assert manager.has_full_attention_states("main")
    assert not manager.has_full_attention_states("draft0")
    with pytest.raises(AssertionError, match="registration must finish"):
        manager.get_full_attention_states("main")
    manager.finish_state_registration()
    assert manager.get_full_attention_states("main") == (state,)
