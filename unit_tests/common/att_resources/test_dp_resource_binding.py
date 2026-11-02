from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import (
    GQAAttentionConfig,
    create_gqa_state,
    MLAConfig,
    create_mla_state,
    PackedMLALogicalConfig,
    create_packed_mla_state,
)
from lightllm.common.attention_state.full_att.compressed_att.compressed_att_state import (
    CompressedAttentionConfig,
    CompressedAttentionState,
)


def resources(kv_type, split, device):
    manager = AttentionStateManager(0)
    for role in ("main", "draft0"):
        arenas = [((5,),), ((0, 2),)] if split else [((0, 2), (5,))]
        for groups in arenas:
            if kv_type.startswith("compressed_"):
                ratio = int(kv_type.removeprefix("compressed_c"))
                state = CompressedAttentionState(
                    CompressedAttentionConfig(groups, ratio, 512, 64, 256, indexer_dim=128 if ratio == 4 else 0),
                    page_size=256,
                    device=device,
                )
            elif kv_type == "mla":
                state = create_mla_state(MLAConfig(groups, 576, torch.bfloat16), device=device)
            elif kv_type.startswith("packed_"):
                state = create_packed_mla_state(
                    PackedMLALogicalConfig(
                        groups,
                        512 if kv_type == "packed_group64_e8m0" else 576,
                        kv_type,
                    ),
                    device=device,
                )
            else:
                state = create_gqa_state(
                    GQAAttentionConfig(groups, 4, 16, torch.bfloat16, tp_rank=1, tp_world_size=2),
                    kv_type=kv_type,
                    device=device,
                )
            if kv_type.startswith("fp8"):
                width = 2 if kv_type == "fp8kv_spt" else 2 * state.config.head_num
                state.scales = torch.tensor([[0.1 * (group[0] + 1)] * width for group in groups])
            state.allocate_storage(2048 if kv_type.startswith("compressed_") else 8)
            manager.register_state(role, state)
    manager.finish_state_registration()
    return manager


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "kv_type",
    [
        "None",
        "int4kv",
        "int8kv",
        "fp8kv_spt",
        "fp8kv_sph",
        "mla",
        "packed_bf16",
        "packed_group128_fp8",
        "packed_group64_e8m0",
        "compressed_c4",
        "compressed_c128",
    ],
)
@pytest.mark.parametrize("source_split", [False, True])
def test_dp_copy_binds_logical_rows_across_different_arenas(monkeypatch, device, kv_type, source_split):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source, target = resources(kv_type, source_split, device), resources(kv_type, not source_split, device)
    for manager in (source, target):
        for state in manager.get_full_attention_states():
            for field in state.get_token_copy_fields().values():
                field.tensor.view(torch.uint8).fill_(17)
                if manager is source:
                    for row, group in enumerate(state.config.logical_layer_groups):
                        value = 31 + group[0] + 64 * (state.model_role == "draft0")
                        field.tensor.select(field.axes.index("layer"), row).view(torch.uint8).fill_(value)
    source_ids = torch.tensor([7, 2, 0], device=device)
    target_ids = torch.tensor([3, 6, 5], device=device)
    if kv_type.startswith("compressed_"):
        source_ids = (torch.tensor([1536, 512, 0], device=device)[:, None] + torch.arange(256, device=device)).flatten()
        target_ids = (
            torch.tensor([768, 1280, 1536], device=device)[:, None] + torch.arange(256, device=device)
        ).flatten()
    operator = target.get_dp_shared_operator()
    plan = operator.bind_peer(source)
    assert operator.bind_peer(source) is plan

    assert len(plan.token_plans) == 4
    if kv_type.startswith("compressed_"):
        assert all(p.token_block_size == 256 for p in plan.token_plans)
    with monkeypatch.context() as patch:
        patch.setattr(source, "get_full_attention_states", lambda: pytest.fail("resources must be prebound"))
        patch.setattr(target, "get_full_attention_states", lambda: pytest.fail("resources must be prebound"))
        operator.copy_from_dp(source, source_ids, target_ids, [], [], 0, 0)
    for role in ("main", "draft0"):
        for layer in (0, 2, 5):
            state = target.get_state(role, "", layer)
            row = state.get_storage_layer_index(layer)
            value = 31 + (0 if layer == 2 else layer) + 64 * (role == "draft0")
            for field in state.get_token_copy_fields().values():
                copied = field.tensor[row, target_ids].view(torch.uint8)
                assert (copied == value).all()
                assert (field.tensor[row, 1].view(torch.uint8) == 17).all()


@pytest.mark.parametrize("mismatch", ["calibration", "shard", "sharing", "weight"])
def test_dp_binding_rejects_incompatible_content_before_any_write(mismatch):
    source, target = resources("fp8kv_spt", True, "cpu"), resources("fp8kv_spt", False, "cpu")
    state = source.get_state("draft0", "gqa", 5)
    if mismatch == "calibration":
        state.scales.mul_(2)
    elif mismatch == "shard":
        state.config = replace(state.config, tp_rank=0)
    elif mismatch == "sharing":
        state.config = replace(state.config, logical_layer_groups=((5, 7),))
    else:
        source.model_identities["main"] = "changed-checkpoint"
    for state in target.get_full_attention_states():
        for field in state.get_token_copy_fields().values():
            field.tensor.view(torch.uint8).fill_(17)
    with pytest.raises(ValueError, match="DP"):
        target.get_dp_shared_operator().copy_from_dp(source, torch.tensor([0]), torch.tensor([1]), [], [], 0, 0)
    for state in target.get_full_attention_states():
        for field in state.get_token_copy_fields().values():
            assert (field.tensor.view(torch.uint8) == 17).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA bounded copy kernels")
@pytest.mark.parametrize("family", ["gdn", "kda", "window", "compressor"])
@pytest.mark.parametrize("source_split", [False, True])
def test_dp_bounded_copy_binds_runtime_and_checkpoints_by_logical_resource(family, source_split, monkeypatch):
    from unit_tests.common.att_resources.test_pd_resource_binding import bounded_peers

    source = bounded_peers(family, 1, source_split, 2)[0]
    target = bounded_peers(family, 1, not source_split, 2)[0]
    for manager, original in ((source, True), (target, False)):
        for state in manager.get_bounded_attention_states():
            state.initialize_page_pools(3, 2, pin_memory=True)
            for request in range(3):
                for field in state.get_runtime_copy_fields(request).values():
                    field.tensor.fill_(173)
            for kind, count in (("big", 3), ("small", 2)):
                for page in range(count):
                    for field in state.get_snapshot_fields(kind, page).values():
                        field.tensor.fill_(173)
            if original:
                for row, layer in enumerate(state.config.logical_layer_ids):
                    value = 31 + layer + 64 * (state.model_role == "draft0")
                    for field in state.get_runtime_copy_fields(0).values():
                        field.tensor[row].fill_(value)
                    for field in state.get_snapshot_fields("big", 1).values():
                        field.tensor[row].fill_(value + 11)
    operator = target.get_dp_shared_operator()
    plan = operator.bind_peer(source)
    assert len(plan.bounded_plans) == 4
    empty = torch.empty(0, dtype=torch.int64, device="cuda")
    with monkeypatch.context() as patch:
        for manager in (source, target):
            for state in manager.get_bounded_attention_states():
                for name in (
                    "get_runtime_transfer_view",
                    "get_snapshot_transfer_view",
                    "get_runtime_copy_fields",
                    "get_snapshot_fields",
                ):
                    patch.setattr(state, name, lambda *args: pytest.fail("native fields must be prebound"))
            patch.setattr(manager, "get_bounded_attention_states", lambda: pytest.fail("resources must be prebound"))
        operator.copy_from_dp(source, empty, empty, [1], [2], 0, 1)
    torch.cuda.synchronize()
    for state in target.get_bounded_attention_states():
        for row, layer in enumerate(state.config.logical_layer_ids):
            value = 31 + layer + 64 * (state.model_role == "draft0")
            for field in state.get_runtime_copy_fields(1).values():
                assert (field.tensor[row] == value).all()
            for field in state.get_runtime_copy_fields(0).values():
                assert (field.tensor[row] == 173).all()
            for field in state.get_snapshot_fields("big", 2).values():
                assert (field.tensor[row] == value + 11).all()
            for field in state.get_snapshot_fields("big", 0).values():
                assert (field.tensor[row] == 173).all()
            for field in state.get_snapshot_fields("small", 0).values():
                assert (field.tensor[row] == 173).all()
