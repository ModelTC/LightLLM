from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.manager import AttentionStateManager
from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.models import get_model_class
from lightllm.models.draft_registry import get_draft_model_class
from lightllm.models.qwen4_exp.model import Qwen4ExpTpPartModel
from lightllm.models.qwen4_exp.mtp.model import Qwen4ExpMTPModel
from lightllm.models.qwen4_exp.attention_state import Qwen4ExpAttentionState
from lightllm.models.qwen4_exp.lexical_state import Qwen4ExpLexicalState, LexicalStateConfig
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.attention_state.bounded_att.gdn.gdn_state import GDNState
from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import ReplayGDNState


@pytest.fixture
def lexical(monkeypatch):
    import lightllm.models.qwen4_exp.lexical_state as module

    def copy(*, copies):
        for source, destination in copies:
            destination.copy_(source)

    monkeypatch.setattr(module, "copy_bounded_att_tensors", copy)
    state = Qwen4ExpLexicalState(LexicalStateConfig((1, 5), 2, 6, 8, 63, torch.bfloat16), 4, 2, "cpu")
    state.state_name = "main_qwen4_lexical_state"
    state.initialize_page_pools(4, 2, pin_memory=False)
    return state


def test_lazy_registration_and_resource_factories(config):
    args = get_env_start_args()
    args.data_type, args.linear_att_ssm_data_type = "bf16", "float32"
    args.cpu_cache_token_page_size = 8
    args.pd_kv_page_size = 4
    for model_type in ("qwen4_exp", "qwen4_exp_text"):
        assert get_model_class({"model_type": model_type}) is Qwen4ExpTpPartModel
        assert get_draft_model_class({"model_type": model_type}, "eagle_with_att") is Qwen4ExpMTPModel
    manager = AttentionStateManager(request_capacity=4)
    for cls, role, layers in ((Qwen4ExpTpPartModel, "main", (0, 1, 2, 3)), (Qwen4ExpMTPModel, "draft0", (4,))):
        full = cls.create_full_att_states(config, args, layers)
        bounded = cls.create_bounded_att_states(config, args, layers, 4)
        for state in (*full, *bounded):
            manager.register_state(role, state)
    manager.finish_state_registration()
    state = manager.get_state("main", "", 3, Qwen4ExpAttentionState)
    assert isinstance(state, Qwen4ExpAttentionState)
    assert manager.get_state("draft0", "qwen4_gqa", 4) is not state
    assert len(manager.get_bounded_attention_states("main")) == 2
    assert not manager.get_bounded_attention_states("draft0")
    for state in manager.get_full_attention_states():
        regions = state.get_cpu_transfer_regions()
        assert len(regions) == 6
        assert all(left.byte_end <= right.byte_start for left, right in zip(regions, regions[1:]))
        assert {region.field for region in regions} == {
            "k",
            "v",
            "indexer.k",
            "indexer.v",
            "positions.k",
            "positions.v",
        }
        assert state.get_token_bytes() > 0


@pytest.mark.parametrize("mode,state_type", [("native", GDNState), ("replay", ReplayGDNState)])
def test_mtp_state_modes_keep_qwen4_resources(config, mode, state_type):
    args = get_env_start_args()
    args.data_type = "bfloat16"
    args.mtp_step = 3
    args.ssm_state_mode = mode
    args.ssm_replay_config = {"cache_len": 16, "compute_mode": "precompute"} if mode == "replay" else {}
    args.linear_att_ssm_data_type = "float32"
    layers = (0, 1, 2, 3)
    bounded = Qwen4ExpTpPartModel.create_bounded_att_states(config, args, layers, 4, device="cpu")
    manager = AttentionStateManager(request_capacity=4)
    for state in (*Qwen4ExpTpPartModel.create_full_att_states(config, args, layers), *bounded):
        manager.register_state("main", state)
    manager.finish_state_registration()
    state = manager.get_state("main", "gdn", 1, state_type)
    assert state.config.logical_layer_ids == (0, 1, 2)
    assert state.get_layer_view(1)[1].dtype == torch.float32
    assert isinstance(manager.get_state("main", "qwen4_gqa", 3), Qwen4ExpAttentionState)
    assert any(isinstance(state, Qwen4ExpLexicalState) for state in bounded)
    assert not Qwen4ExpMTPModel.create_bounded_att_states(config, args, (4,), 4, device="cpu")


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_replicated_indexer_and_positions_layout(config, tp):
    args = get_env_start_args()
    args.data_type = "bf16"
    for rank in range(tp):
        (state,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3), rank, tp)
        regions = state._get_transfer_regions(8)
        for region in regions[2:]:
            assert region.write_enabled == (rank == 0)
            assert region.shape[2] == 1
        assert sum(region.byte_end - region.byte_start for region in regions) == regions[-1].byte_end


@pytest.mark.parametrize("kv_type", ["None", "fp8kv_spt"])
def test_state_storage_accounts_for_all_fields_and_restores_positions(config, kv_type):
    args = get_env_start_args()
    args.data_type = "bf16"
    args.llm_kv_type = kv_type
    (state,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
    state.device = torch.device("cpu")
    state.allocate_storage(8)
    tensors = (
        state.k_storage,
        state.v_storage,
        state.indexer.k_storage,
        state.indexer.v_storage,
        state.positions.k_storage,
        state.positions.v_storage,
    )
    assert sum(t.numel() * t.element_size() for t in tensors) == (8 + state.page_size) * state.get_token_bytes()
    indices = torch.tensor([1, 4])
    cos = torch.arange(8).view(2, 4).bfloat16()
    state.store_positions(indices, cos, -cos)
    view = state.qsa_view(3)
    assert torch.equal(view.key_cos[indices], cos)
    assert torch.equal(view.key_sin[indices], -cos)
    assert view.raw_keys.data_ptr() == state.indexer.k_storage.data_ptr()
    assert view.raw_keys.dtype == torch.bfloat16
    if kv_type == "fp8kv_spt":
        assert view.pooled_keys.dtype == state.k_storage.dtype == torch.float8_e4m3fn
        fields = {r.field: r for r in state._get_transfer_regions(8)}
        assert fields["indexer.k"].dtype == torch.bfloat16
        assert fields["indexer.v"].dtype == torch.float8_e4m3fn
        assert fields["positions.k"].dtype == torch.bfloat16


def test_fp8_rejects_calibration_files_and_mixed_encoding_dp_copy(config):
    args = get_env_start_args()
    args.data_type = "bfloat16"
    args.llm_kv_type = "fp8kv_spt"
    args.kv_quant_calibration_config_path = "calibration.json"
    with pytest.raises(ValueError, match="unit scales"):
        Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
    args.kv_quant_calibration_config_path = None
    (fp8,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
    args.llm_kv_type = "None"
    (bf16,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
    with pytest.raises(AssertionError, match="same KV and indexer encoding"):
        fp8.copy_tokens_from_dp(None, None, bf16)


def qsa_tensors(state):
    return (
        state.k_storage,
        state.v_storage,
        state.indexer.k_storage,
        state.indexer.v_storage,
        state.positions.k_storage,
        state.positions.v_storage,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("page_device", ["cpu", "cuda"])
def test_qsa_kv_write_copy_and_page_round_trip(config, fp8, page_device):
    args = get_env_start_args()
    args.data_type = "bfloat16"
    args.cpu_cache_token_page_size = args.pd_kv_page_size = 4
    args.llm_kv_type = "fp8kv_spt" if fp8 else "None"
    states = []
    for _ in range(2):
        (state,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
        state.device = torch.device("cuda")
        state.allocate_storage(16)
        states.append(state)
    source, target = states
    for tensor in qsa_tensors(source):
        tensor.view(torch.uint8).random_(0, 256)
    for tensor in qsa_tensors(target):
        tensor.view(torch.uint8).fill_(173)
    src = torch.tensor([1, 4, 7], device="cuda", dtype=torch.int32)
    dst = torch.tensor([2, 6, 8], device="cuda", dtype=torch.int64)
    kv = torch.randn((3, 4, 8), device="cuda", dtype=torch.bfloat16)
    source.write_full_att_kv(3, src, kv)
    k, v = source.get_layer_view(3)
    torch.testing.assert_close(k.float()[src], kv[:, :2].to(k.dtype).float(), rtol=0, atol=0)
    torch.testing.assert_close(v.float()[src], kv[:, 2:].to(v.dtype).float(), rtol=0, atol=0)
    expected = [tensor.view(torch.uint8)[:, src].clone() for tensor in qsa_tensors(source)]
    source.copy_full_att_tokens(src, dst)
    for tensor, reference in zip(qsa_tensors(source), expected):
        assert torch.equal(tensor.view(torch.uint8)[:, dst], reference)
    target.copy_tokens_from_dp(src, dst, source)
    for tensor, reference in zip(qsa_tensors(target), expected):
        assert torch.equal(tensor.view(torch.uint8)[:, dst], reference)
        tensor.view(torch.uint8).fill_(173)
    regions = source.get_cpu_transfer_regions() if page_device == "cpu" else source.get_pd_transfer_regions()
    layout = StateTransferLayout(
        source.state_name,
        4,
        16,
        16 + regions[-1].byte_end,
        tuple(replace(r, byte_start=r.byte_start + 16, byte_end=r.byte_end + 16) for r in regions),
    )
    pages = torch.full(
        (3, layout.byte_end + 16), 91, dtype=torch.uint8, device=page_device, pin_memory=page_device == "cpu"
    )
    if page_device == "cpu":
        source.write_to_cpu(src, pages, [1], layout)
        target.read_from_cpu(pages, [1], dst, layout)
    else:
        source.write_to_pd_page(src, pages, 1, layout)
        target.read_from_pd_page(pages, dst, 1, layout)
    torch.cuda.synchronize()
    for tensor, reference in zip(qsa_tensors(target), expected):
        assert torch.equal(tensor.view(torch.uint8)[:, dst], reference)
        untouched = [i for i in range(tensor.shape[1]) if i not in (2, 6, 8)]
        assert (tensor.view(torch.uint8)[:, untouched] == 173).all()
    assert (pages[[0, 2]] == 91).all()
    assert (pages[:, :16] == 91).all() and (pages[:, layout.byte_end :] == 91).all()
    for region in layout.regions:
        tail = pages[1, region.byte_start : region.byte_end].view(region.dtype).view(region.shape)
        assert (tail[:, 3:].contiguous().view(torch.uint8) == 91).all()


@pytest.mark.parametrize("page_kind", ["big", "small"])
def test_lexical_prefix_round_trip_is_independent_of_runtime(lexical, page_kind):
    lexical.ngram[:, 1].fill_(17)
    lexical.convolution[:, 1].fill_(3)
    lexical.save_bounded_att_runtime_to_pages([1], [128], page_kind, [0])
    lexical.reset_bounded_att_state([1])
    assert bool((lexical.ngram[:, 1] == 63).all())
    lexical.restore_bounded_att_runtime_from_pages([2], [128], page_kind, [0])
    assert bool((lexical.ngram[:, 2] == 17).all())
    assert bool((lexical.convolution[:, 2] == 3).all())
    lexical.reset_bounded_att_state([2])
    assert bool((lexical._pool(page_kind).ngram[0] == 17).all())


def layout_for(state):
    regions = state.get_cpu_transfer_regions()
    return StateTransferLayout(
        state.state_name,
        None,
        16,
        16 + regions[-1].byte_end,
        tuple(replace(region, byte_start=region.byte_start + 16, byte_end=region.byte_end + 16) for region in regions),
    )


def test_cpu_and_pd_lexical_transfers_preserve_unrelated_bytes(lexical):
    layout = layout_for(lexical)
    pages = torch.full((3, layout.byte_end + 16), 0x5A, dtype=torch.uint8)
    lexical.ngram[:, 1].fill_(7)
    lexical.convolution[:, 1].fill_(11)
    lexical.save_bounded_att_runtime_to_pages([1], [7], "big", [0])
    lexical.write_state_pages_to_cpu("big", [0, 0], pages, [1, 2], layout, [False, True])
    assert bool((pages[2] == 0x5A).all())
    assert bool((pages[:, :16] == 0x5A).all())
    assert bool((pages[:, layout.byte_end :] == 0x5A).all())
    lexical.read_state_pages_from_cpu(pages, [1], [2], layout)
    lexical.restore_bounded_att_runtime_from_pages([2], [7], "big", [2])
    assert torch.equal(lexical.ngram[:, 2], lexical.ngram[:, 1])
    assert torch.equal(lexical.convolution[:, 2], lexical.convolution[:, 1])
    lexical.write_to_pd_page(1, pages, 0, 7, layout)
    lexical.reset_bounded_att_state([2])
    lexical.read_from_pd_page(pages, 2, 0, 7, layout)
    assert torch.equal(lexical.ngram[:, 2], lexical.ngram[:, 1])
    assert torch.equal(lexical.convolution[:, 2], lexical.convolution[:, 1])


def test_dp_copy_includes_runtime_and_prefix(lexical):
    other = Qwen4ExpLexicalState(lexical.config, 4, 2, "cpu")
    other.initialize_page_pools(4, pin_memory=False)
    other.ngram[:, 0].fill_(29)
    other.convolution[:, 0].fill_(13)
    lexical.copy_runtime_state_from_dp([0], [2], other)
    assert bool((lexical.ngram[:, 2] == 29).all())
    other.save_bounded_att_runtime_to_pages([0], [8], "big", [1])
    lexical.copy_big_pages_from_dp([1], [3], other)
    lexical.restore_bounded_att_runtime_from_pages([1], [8], "big", [3])
    assert bool((lexical.convolution[:, 1] == 13).all())


def test_mtp_commit_maps_request_rows_and_keeps_hold(lexical):
    observed = []
    lexical.speculative_lexical_caches = {1: SimpleNamespace(commit=lambda *args: observed.append(args))}
    requests, accepted = torch.tensor([2, 0, 3]), torch.tensor([1, 3, 1])
    lexical.commit_bounded_att_state(requests, accepted, torch.full_like(accepted, 3))
    req, starts, lengths, hold = observed[0]
    assert req is requests and lengths is accepted
    assert starts.tolist() == [0, 1, 2] and hold == 3


def test_ipc_state_excludes_weights_and_speculative_workspaces(lexical):
    lexical.lexical_caches = {1: SimpleNamespace(layer_weight=object())}
    lexical.speculative_lexical_caches = {1: object()}
    exported = lexical.__getstate__()
    assert not exported["lexical_caches"] and not exported["speculative_lexical_caches"]
    assert exported["ngram"] is lexical.ngram
    assert exported["convolution"] is lexical.convolution
    assert lexical.lexical_caches and lexical.speculative_lexical_caches
