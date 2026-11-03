from lightllm.common.attention_state.storage import allocate_runtime_storage

from lightllm.common.attention_state.storage import allocate_state_storage
from dataclasses import replace
import json

import pytest
import torch

from lightllm.common.attention_state.copy_plan import CopyPlan
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, create_gqa_state
from lightllm.common.attention_state.manifest import build_state_manifest, validate_state_manifest
from unit_tests.common.att_resources.test_dp_resource_binding import resources


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
def test_pd_wire_identity_and_copy_do_not_depend_on_local_arenas(pd_args, device, kv_type, source_split):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    compressed = kv_type.startswith("compressed_")
    pd_args.pd_kv_page_size = 512 if compressed else 8
    source, target = resources(kv_type, source_split, device), resources(kv_type, not source_split, device)
    for manager in (source, target):
        for state in manager.get_full_attention_states():
            for field in state.get_transfer_fields().values():
                field.tensor.view(torch.uint8).fill_(173)
                if manager is source:
                    for row, group in enumerate(state.config.logical_layer_groups):
                        field.tensor[row].view(torch.uint8).fill_(31 + group[0] + 64 * (state.model_role == "draft0"))
    validate_state_manifest(build_state_manifest(source), build_state_manifest(target))
    page_bytes = source.get_pd_operator().get_page_bytes()
    assert page_bytes == target.get_pd_operator().get_page_bytes()
    pages = torch.full((3, page_bytes), 173, dtype=torch.uint8, device=device)
    src, dst = torch.tensor([7, 2, 0], device=device), torch.tensor([3, 6, 5], device=device)
    if compressed:
        src = torch.cat((torch.arange(256, 512, device=device), torch.arange(9, device=device)))
        dst = torch.cat((torch.arange(768, 1024, device=device), torch.arange(1536, 1545, device=device)))
    CopyPlan.for_pd([source], pages, True).run(pd_page_index=1, token_indices=src)
    CopyPlan.for_pd([target], pages, False).run(pd_page_index=1, token_indices=dst)
    if compressed:
        dst = torch.cat((dst, torch.arange(1545, 1792, device=device)))
    for role in ("main", "draft0"):
        for layer in (0, 2, 5):
            state = target.get_state(role, "", layer)
            row = state.get_storage_layer_index(layer)
            value = 31 + (0 if layer == 2 else layer) + 64 * (role == "draft0")
            for field in state.get_transfer_fields().values():
                assert (field.tensor[row, dst].contiguous().view(torch.uint8) == value).all()
                assert (field.tensor[row, 1].contiguous().view(torch.uint8) == 173).all()
    assert (pages[[0, 2]] == 173).all()


@pytest.mark.parametrize("mismatch", ["sharing", "calibration", "weights"])
def test_regrouped_pd_peer_still_rejects_different_content(pd_args, mismatch):
    source, target = resources("fp8kv_spt", True, "cpu"), resources("fp8kv_spt", False, "cpu")
    state = target.get_state("draft0", "", 0)
    if mismatch == "sharing":
        state.config = replace(state.config, logical_layer_groups=((0, 3), (5,)))
    elif mismatch == "calibration":
        state.scales[0].mul_(2)
    else:
        target.model_identities["main"] = "other-weights"
    with pytest.raises(ValueError, match="incompatible|model or weight"):
        validate_state_manifest(build_state_manifest(source), build_state_manifest(target))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kv_type", ["None", "int4kv", "fp8kv_sph"])
@pytest.mark.parametrize("source_tp,target_tp", [(1, 4), (4, 1), (2, 4)])
def test_pd_regrouping_and_resharding_preserve_global_heads_and_calibration(
    pd_args, tmp_path, device, kv_type, source_tp, target_tp
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    calibration = dict(
        version=2,
        quant_type="per_head",
        layers={
            role: {str(layer): dict(k=[1 + layer, 2 + layer], v=[3 + layer, 4 + layer]) for layer in (0, 5)}
            for role in ("main", "draft0")
        },
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    pd_args.kv_quant_calibration_config_path = str(path)

    def peers(tp, split):
        managers = []
        for rank in range(tp):
            manager = AttentionStateManager(0)
            for role in ("main", "draft0"):
                for groups in [((5,),), ((0, 2),)] if split else [((0, 2), (5,))]:
                    config = GQAAttentionConfig(groups, 2, 16, torch.bfloat16, rank, tp)
                    manager.register_state(role, create_gqa_state(config, kv_type=kv_type, device=device))
            manager.finish_state_registration()
            manager.load_full_att_calibration(pd_args)
            for state in manager.get_full_attention_states():
                allocate_state_storage(state, 8)
                for field in state.get_transfer_fields().values():
                    field.tensor.view(torch.uint8).fill_(173)
            managers.append(manager)
        return managers

    source, target = peers(source_tp, True), peers(target_tp, False)
    reference = build_state_manifest(source[0])
    for manager in (*source, *target):
        validate_state_manifest(reference, build_state_manifest(manager))
    for rank, manager in enumerate(source):
        for state in manager.get_full_attention_states():
            start = rank * state.config.head_num if source_tp <= 2 else rank // (source_tp // 2)
            for row, group in enumerate(state.config.logical_layer_groups):
                for i, field in enumerate(state.get_transfer_fields().values()):
                    for head in range(state.config.head_num):
                        value = 31 + group[0] + start + head + 7 * i + 64 * (state.model_role == "draft0")
                        field.tensor[row, :, head].view(torch.uint8).fill_(value)
    pages = torch.full((2, source[0].get_pd_operator().get_page_bytes()), 173, dtype=torch.uint8, device=device)
    src, dst = torch.tensor([7, 2, 0], device=device), torch.tensor([3, 6, 5], device=device)
    CopyPlan.for_pd(source, pages, True).run(pd_page_index=1, token_indices=src)
    CopyPlan.for_pd(target, pages, False).run(pd_page_index=1, token_indices=dst)
    for rank, manager in enumerate(target):
        for state in manager.get_full_attention_states():
            start = rank * state.config.head_num if target_tp <= 2 else rank // (target_tp // 2)
            for row, group in enumerate(state.config.logical_layer_groups):
                for i, field in enumerate(state.get_transfer_fields().values()):
                    for head in range(state.config.head_num):
                        value = 31 + group[0] + start + head + 7 * i + 64 * (state.model_role == "draft0")
                        assert (field.tensor[row, dst, head].contiguous().view(torch.uint8) == value).all()
                        assert (field.tensor[row, 1, head].contiguous().view(torch.uint8) == 173).all()
    assert (pages[0] == 173).all()


def bounded_peers(family, tp, split, mtp_step):
    from lightllm.common.attention_state.bounded_att import (
        ConvSSMStateConfig,
        GDNState,
        KDAState,
        FP8E8M0SlidingWindowConfig,
        FP8E8M0SlidingWindowState,
        WindowCompressorStateConfig,
        WindowCompressorState,
    )

    managers = []
    for rank in range(tp):
        manager = AttentionStateManager(3)
        for role in ("main", "draft0"):
            for layers in ((1,), (5,)) if split else ((5, 1),):
                if family in ("gdn", "kda"):
                    config = ConvSSMStateConfig(layers, 2, 2, 8, 8, 4, torch.bfloat16, torch.float32, rank, tp)
                    state = allocate_runtime_storage(
                        (GDNState if family == "gdn" else KDAState)(config, 3, mtp_step), "cuda"
                    )
                elif family == "window":
                    config = FP8E8M0SlidingWindowConfig(layers, 16, 128, 64, tp_rank=rank, tp_world_size=tp)
                    state = allocate_runtime_storage(FP8E8M0SlidingWindowState(config, 3, mtp_step), "cuda")
                else:
                    config = WindowCompressorStateConfig(layers, 8, 4, 16, tp_rank=rank, tp_world_size=tp)
                    state = allocate_runtime_storage(WindowCompressorState(config, 3, mtp_step), "cuda")
                manager.register_state(role, state)
        manager.finish_state_registration()
        managers.append(manager)
    return managers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA window and recurrent copy kernels")
@pytest.mark.parametrize("family", ["gdn", "kda", "window", "compressor"])
@pytest.mark.parametrize("source_tp,target_tp", [(1, 4), (4, 1), (2, 4)])
@pytest.mark.parametrize("source_split", [False, True])
@pytest.mark.parametrize("length", [2, 511])
def test_bounded_pd_preserves_logical_layers_across_grouping_and_tp(
    pd_args, family, source_tp, target_tp, source_split, length, monkeypatch
):
    from unit_tests.common.att_resources.test_dsv4_state import gather_swa_reference

    source = bounded_peers(family, source_tp, source_split, 3)
    target = bounded_peers(family, target_tp, not source_split, 1)
    reference = build_state_manifest(source[0])
    for manager in (*source, *target):
        validate_state_manifest(reference, build_state_manifest(manager))
    for peers, is_source in ((source, True), (target, False)):
        for rank, manager in enumerate(peers):
            for state in manager.get_bounded_attention_states():
                if family in ("gdn", "kda"):
                    state.conv_state.fill_(-1)
                    state.ssm_state.fill_(-2)
                    state.mtp_state_index.fill_(2 if is_source else 1)
                else:
                    (state.swa_buffer if family == "window" else state.buffer).fill_(173)
                if not is_source:
                    continue
                for row, layer in enumerate(state.config.logical_layer_ids):
                    value = 31 + layer + 64 * (state.model_role == "draft0")
                    if family in ("gdn", "kda"):
                        config = state.config
                        head_start = rank * config.num_k_heads if len(peers) <= 2 else rank // (len(peers) // 2)
                        for segment in range(3):
                            for head in range(config.num_k_heads):
                                channel = (segment * config.num_k_heads + head) * 8
                                state.conv_state[row, 0, channel : channel + 8] = (
                                    value + head_start + head + 10 * segment + torch.arange(7, device="cuda")
                                )
                        for head in range(config.num_v_heads):
                            state.ssm_state[row, :4, head] = (
                                value + 50 + head_start + head + torch.arange(4, device="cuda")
                            )[:, None, None]
                    else:
                        (state.swa_buffer if family == "window" else state.buffer)[row, 0].fill_(value)
    pages = torch.full((3, source[0].get_pd_operator().get_page_bytes()), 173, dtype=torch.uint8, device="cuda")
    writer = CopyPlan.for_pd(source, pages, True)
    reader = CopyPlan.for_pd(target, pages, False)
    with monkeypatch.context() as patch:
        for manager in (*source, *target):
            for state in manager.get_bounded_attention_states():
                for name in ("get_runtime_transfer_view",):
                    patch.setattr(state, name, lambda *args, **kwargs: pytest.fail("native PD fields must be prebound"))
        writer.run(pd_page_index=1, task_kind="att_state", req_idx=0, state_token_len=length)
        reader.run(pd_page_index=1, task_kind="att_state", req_idx=1, state_token_len=length)
    for rank, manager in enumerate(target):
        for state in manager.get_bounded_attention_states():
            for row, layer in enumerate(state.config.logical_layer_ids):
                value = 31 + layer + 64 * (state.model_role == "draft0")
                config = state.config
                if family in ("gdn", "kda"):
                    head_start = rank * config.num_k_heads if target_tp <= 2 else rank // (target_tp // 2)
                    for segment in range(3):
                        for head in range(config.num_k_heads):
                            channel = (segment * config.num_k_heads + head) * 8
                            expected = value + head_start + head + 10 * segment + torch.arange(2, 6, device="cuda")
                            assert (state.conv_state[row, 1, channel : channel + 8, :4] == expected).all()
                    for head in range(config.num_v_heads):
                        assert (state.ssm_state[row, 2, head] == value + 52 + head_start + head).all()
                    assert state.mtp_state_index.tolist() == [1, 0, 1]
                    assert (state.conv_state[row, [0, 2]] == -1).all()
                    assert (state.ssm_state[row, [0, 1, 3, 4, 5]] == -2).all()
                else:
                    if family == "window":
                        window = gather_swa_reference(state, 1, length)[row]
                    else:
                        positions = torch.arange(length - config.window_size, length)
                        window = state.buffer[row, 1, positions.remainder(state.buffer.shape[2]).cuda()].cpu()
                    count = min(length, config.window_size)
                    assert (window[-count:] == value).all()
                    untouched = (state.swa_buffer if family == "window" else state.buffer)[row, [0, 2]]
                    assert (untouched == 173).all()
    assert (pages[[0, 2]] == 173).all()
