from types import SimpleNamespace

import pytest

from lightllm.common.speculative.config import SpeculativeConfig


@pytest.mark.parametrize(
    ("mode", "step", "requested_dynamic", "expected_dynamic", "draft_model_count"),
    [
        (None, 0, False, False, 0),
        ("vanilla_with_att", 3, False, False, 3),
        ("eagle3", 3, True, True, 1),
        ("dspark", 0, False, True, 1),
        ("dflash", 0, True, False, 1),
    ],
)
def test_speculative_config_normalizes_modes(
    mode, step, requested_dynamic, expected_dynamic, draft_model_count
):
    config = SpeculativeConfig.from_args(
        SimpleNamespace(mtp_mode=mode, mtp_step=step, mtp_dynamic_verify=requested_dynamic)
    )

    config.validate()
    assert config.dynamic_verify is expected_dynamic
    assert config.draft_model_count == draft_model_count
    assert config.enabled is (mode is not None)


def test_dynamic_eagle3_uses_unit_graph_granularity():
    config = SpeculativeConfig(mode="eagle3", step=7, dynamic_verify=True)

    assert config.get_decode_graph_mtp_step(model_config={}, is_draft_model=False) == 0
    assert config.get_decode_graph_warmup_mtp_step(model_config={}, is_draft_model=False) == 3


def test_invalid_mode_is_rejected():
    with pytest.raises(AssertionError, match="unsupported speculative mode"):
        SpeculativeConfig(mode="unknown", step=1).validate()
