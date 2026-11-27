from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from lightllm.models.glm5_next import prefill_workspace
from lightllm.models.glm5_next.model import Glm5NextTpPartModel
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from unit_tests.models.glm5_next.test_fp8_kv import _PrefillCase, _make_workspace_backend, _capture_cuda_graph


@pytest.fixture
def current_cuda_device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("run_mode", ["normal", "decode"])
def test_prefill_workspace_reservation(current_cuda_device, overlap, run_mode):
    model = SimpleNamespace(
        args=SimpleNamespace(enable_prefill_microbatch_overlap=overlap, run_mode=run_mode),
        batch_max_tokens=8,
        max_req_num=2,
        max_seq_length=13,
        config={"index_topk": 2048, "index_kpool": 4},
        att_state_role=f"workspace_test_{run_mode}_{overlap}",
    )
    reserved = prefill_workspace.reserve_fp8_sparse_prefill_workspaces(model)
    pointers = [{key: value.data_ptr() for key, value in buffers.items()} for buffers in reserved]
    # Reserving again, as a second model role sharing the name would, returns the same storage.
    again = prefill_workspace.reserve_fp8_sparse_prefill_workspaces(model)
    assert [{key: value.data_ptr() for key, value in buffers.items()} for buffers in again] == pointers
    assert len(reserved) == 1 + int(overlap)
    capacity = 8 if run_mode == "decode" else 34
    assert reserved[0]["gathered_kv"].shape == (capacity, 1, 512)
    assert reserved[0]["chunk_topk_indices"].shape == (8, 2176)

    for views in reserved:
        torch.testing.assert_close(views["cu_seqlens_q"], torch.arange(9, dtype=torch.int32, device="cuda"))


def test_draft_models_reuse_the_main_models_prefill_regions(current_cuda_device, monkeypatch):
    from lightllm.utils import workspace_utils

    monkeypatch.setattr(workspace_utils, "_workspace_buffers", {})

    def model(role, batch_max_tokens):
        return SimpleNamespace(
            args=SimpleNamespace(enable_prefill_microbatch_overlap=True, run_mode="normal"),
            batch_max_tokens=batch_max_tokens,
            max_req_num=2,
            max_seq_length=13,
            config={"index_topk": 2048, "index_kpool": 4},
            att_state_role=role,
        )

    main = prefill_workspace.reserve_fp8_sparse_prefill_workspaces(model("main", 8))
    draft = prefill_workspace.reserve_fp8_sparse_prefill_workspaces(model("draft0", 4))
    for main_region, draft_region in zip(main, draft):
        for key in main_region:
            # Each microbatch region of the draft is a prefix of the same region of the main model.
            assert draft_region[key].data_ptr() == main_region[key].data_ptr(), key
    assert main[0]["gathered_kv"].data_ptr() != main[1]["gathered_kv"].data_ptr()
    assert draft[0]["chunk_topk_indices"].shape == (4, 2176)
    torch.testing.assert_close(main[1]["cu_seqlens_q"], torch.arange(9, dtype=torch.int32, device="cuda"))


def test_prefill_microbatches_use_independent_scratch_during_graph_replay(monkeypatch, current_cuda_device):
    monkeypatch.setattr(prefill_workspace, "KV_GATHER_CHUNK_SIZE", 7)
    backend = _make_workspace_backend(8, overlap=True)
    first = _PrefillCase([15], [3], 8, backend=backend)
    second = _PrefillCase([17], [3], 8, backend=backend)
    second.infer_state.microbatch_index = 1
    _, second.att_state = second.new_state()
    first.check(first.run())
    second.check(second.run())
    pointers = [{key: value.data_ptr() for key, value in buffers.items()} for buffers in backend._prefill_workspaces]
    # Each microbatch writes its own scratch; the read-only query offsets are one shared arange.
    assert all(pointers[0][key] != pointers[1][key] for key in pointers[0] if key != "cu_seqlens_q")
    assert pointers[0]["cu_seqlens_q"] == pointers[1]["cu_seqlens_q"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Attention must not allocate a new prefill workspace")

    monkeypatch.setattr(prefill_workspace, "reserve_fp8_sparse_prefill_workspaces", forbidden)
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


@pytest.mark.parametrize("model_class", [Glm5NextTpPartModel, Glm5NextMTPModel])
@pytest.mark.parametrize("kv_type", ["None", "fp8kv_dsa"])
def test_glm_reserves_prefill_workspace_during_resource_preparation(monkeypatch, model_class, kv_type):
    model = object.__new__(model_class)
    model.args = SimpleNamespace(llm_kv_type=kv_type)
    model.mtp_previous_draft_models = []
    sparse = SimpleNamespace(is_linear_attention_layer=False)
    linear = SimpleNamespace(is_linear_attention_layer=True)
    model.layers_infer = [linear, sparse]
    model.att_state_manager = SimpleNamespace(get_full_attention_states=lambda role: [object()])
    calls = []

    def prepare(self):
        assert self is model
        calls.append("prepare_states")

    workspaces = [{"gathered_kv": object()}]

    def reserve(prepared_model):
        assert prepared_model is model
        calls.append("reserve_workspace")
        return workspaces

    monkeypatch.setattr(TpPartBaseModel, "prepare_attention_states", prepare)
    monkeypatch.setattr(prefill_workspace, "reserve_fp8_sparse_prefill_workspaces", reserve)
    model.prepare_attention_states()
    assert calls == (["prepare_states", "reserve_workspace"] if kv_type == "fp8kv_dsa" else ["prepare_states"])
    assert getattr(sparse, "fp8_sparse_prefill_workspaces", None) is (workspaces if kv_type == "fp8kv_dsa" else None)
    assert not hasattr(linear, "fp8_sparse_prefill_workspaces")


def test_sparse_layer_passes_each_microbatch_its_reserved_workspace():
    class CapturePrefill:
        state_key = object()

        def forward(self, att_state, inputs, *, out):
            return inputs.fp8_workspace

    backend = CapturePrefill()
    regions = [{"gathered_kv": torch.empty(1)} for _ in range(2)]
    layer = object.__new__(Glm5NextTransformerLayerInfer)
    layer.is_linear_attention_layer = False
    layer.tp_q_head_num_ = layer.kv_lora_rank = 1
    layer.softmax_scale = 1.0
    layer.network_config_ = {"index_topk": 1, "index_kpool": 1}
    layer.fp8_sparse_prefill_workspaces = regions
    layer.layer_num_ = 0
    query = torch.zeros(1, 1, 1)
    layer.alloc_tensor = torch.empty
    layer.indexer = SimpleNamespace(_get_indices=lambda **kwargs: (None, None))
    layer._sparse_kv_cache = lambda state: query
    weights = SimpleNamespace(k_b_proj_=SimpleNamespace(bmm=lambda query, out: out))
    for slot in (0, 1, 0):
        state = SimpleNamespace(
            microbatch_index=slot,
            att_state_role="main",
            attention_backend_manager=SimpleNamespace(get_backend=lambda role, phase, layer_id: backend),
            att_states={backend.state_key: None},
            nsa_step=SimpleNamespace(q_kv_start_loc=None),
            get_topk_indices_params={"hidden_states": query, "q_lora": query},
        )
        assert layer._context_attention_kernel(query, query, state, weights) is regions[slot]
