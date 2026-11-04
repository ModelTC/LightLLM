import copy
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.core.formats.kv.calibration import calibration_table_identity
from lightllm.common.attention_state.full_att.gqa.static_fp8_state import (
    StaticFP8AttentionConfig,
    StaticFP8FullAttentionState,
)
from lightllm.server.router.model_infer.mode_backend.pd.resource_io import calibration_identities, check_calibration


def table():
    return dict(
        num_layers=2,
        num_head=2,
        scales_shape=[2, 4],
        scales=[[0.25, 0.5, 0.75, 1], [1, 2, 3, 4]],
        quant_type="per_head",
        q_calibration=dict(num_head=2, scales_shape=[2, 2], scales=[[1, 2], [3, 4]]),
    )


def manager_with_calibration(calibration, tp=1, rank=0, replicated=False):
    manager = AttentionStateManager(1)
    state = StaticFP8FullAttentionState(
        StaticFP8AttentionConfig(((0,), (1,)), 4, 64, rank, tp, replicated=replicated, scale_mode="per_head"),
        device="meta",
    )
    manager.register_state("main", state)
    manager.finish_state_registration()
    state.bind_calibration(calibration, 0)
    return manager, state


def test_table_identity_uses_fp32_kv_content_not_metadata_or_q():
    original = table()
    changed = copy.deepcopy(original)
    changed["source"] = "/different/path"
    changed["q_calibration"]["scales"][0][0] = 17
    changed["scales"][0][0] += 1e-10  # Same FP32 value actually used by the writer.
    assert calibration_table_identity(original, "per_head") == calibration_table_identity(changed, "per_head")
    changed["scales"][0][0] = 0.5
    assert calibration_table_identity(original, "per_head") != calibration_table_identity(changed, "per_head")


def test_implicit_and_explicit_unit_scales_have_same_identity():
    explicit = dict(num_layers=3, scales_shape=[3, 2], scales=[[1, 1]] * 3)
    assert calibration_table_identity(None, "per_tensor") == calibration_table_identity(explicit, "per_tensor")


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_scale_cannot_get_a_content_identity(value):
    calibration = table()
    calibration["scales"][0][0] = value
    with pytest.raises(ValueError, match="finite and positive"):
        calibration_table_identity(calibration, "per_head")


@pytest.mark.parametrize("tp", [1, 2, 8])
def test_pd_identity_covers_all_global_heads_before_tp_sharding(tp):
    calibration = table()
    reference, _ = manager_with_calibration(calibration)
    managers = [manager_with_calibration(calibration, tp, rank)[0] for rank in range(tp)]
    expected = calibration_identities([reference])
    assert calibration_identities(managers) == expected

    # An explicitly expanded table and a compact repeated-head table are equivalent.
    expanded = copy.deepcopy(calibration)
    expanded.update(num_head=4, scales_shape=[2, 8])
    expanded["scales"] = torch.tensor(expanded["scales"]).repeat_interleave(2, dim=-1).tolist()
    assert calibration_identities([manager_with_calibration(expanded)[0]]) == expected

    changed = copy.deepcopy(calibration)
    changed["scales"][0][1] = 0.125
    changed_manager, changed_state = manager_with_calibration(changed, tp, 0)
    if tp > 1:
        _, old_state = manager_with_calibration(calibration, tp, 0)
        torch.testing.assert_close(changed_state.scales, old_state.scales)
    with pytest.raises(ValueError, match="mismatch"):
        check_calibration(expected, calibration_identities([changed_manager]))
    with pytest.raises(ValueError, match="disagree"):
        calibration_identities([reference, changed_manager])


def test_pd_identity_ignores_q_and_rejects_missing_peer_identity():
    local, _ = manager_with_calibration(table())
    calibration = table()
    calibration["q_calibration"]["scales"][0][0] = 23
    remote, _ = manager_with_calibration(calibration)
    identity = calibration_identities([local])
    check_calibration(identity, calibration_identities([remote]))
    with pytest.raises(ValueError, match="mismatch"):
        check_calibration(identity, None)
    with pytest.raises(ValueError, match="mismatch"):
        check_calibration({}, identity)
    check_calibration({}, None)


@pytest.mark.parametrize("tp,rank", [(2, 1), (4, 0), (4, 3)])
def test_replicated_state_keeps_all_global_kv_and_q_scales(tp, rank):
    reference, expected = manager_with_calibration(table())
    manager, actual = manager_with_calibration(table(), tp, rank, replicated=True)
    torch.testing.assert_close(actual.scales, expected.scales)
    torch.testing.assert_close(actual.q_scales, expected.q_scales)
    assert calibration_identities([manager]) == calibration_identities([reference])


def test_planning_and_binding_share_one_file_snapshot(tmp_path):
    path = tmp_path / "calibration.json"
    original = table()
    path.write_text(json.dumps(original))
    args = SimpleNamespace(kv_quant_calibration_config_path=str(path))
    manager = AttentionStateManager(1)
    planned = calibration_table_identity(manager.read_full_att_calibration(args), "per_head")
    changed = copy.deepcopy(original)
    changed["scales"][0][0] = 32
    path.write_text(json.dumps(changed))
    state = StaticFP8FullAttentionState(
        StaticFP8AttentionConfig(((0,), (1,)), 2, 64, scale_mode="per_head"), device="meta"
    )
    manager.register_state("main", state)
    manager.finish_state_registration()
    manager.load_full_att_calibration(args)
    assert state.calibration_identity == planned
    torch.testing.assert_close(state.scales, torch.tensor(original["scales"]))
    with pytest.raises(ValueError, match="source cannot change"):
        manager.read_full_att_calibration(SimpleNamespace(kv_quant_calibration_config_path="another-file"))
