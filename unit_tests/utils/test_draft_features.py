import pytest

from lightllm.utils.draft_features import get_target_feature_sources


def test_standard_draft_keeps_sorted_layer_features():
    assert get_target_feature_sources({"target_layer_ids": [4, 1]}) == (1, 4)


def test_teacher_export_includes_final_postnorm_in_projection_width():
    config = {
        "dflash_config": {"target_layer_ids": [3, 27, 51, 54]},
        "loop_teacher": {"feature_sources": [3, 27, 51, 54, "final_postnorm"]},
    }
    assert len(get_target_feature_sources(config)) * 4096 == 20480


def test_explicit_feature_contract_supports_nested_config():
    config = {"text_config": {"target_layer_ids": [0, 2], "target_feature_sources": [0, 2, "final_postnorm"]}}
    assert get_target_feature_sources(config) == (0, 2, "final_postnorm")


@pytest.mark.parametrize("layers", [[], [0, 0], [-1], [True], [0.0], ["0"]])
def test_invalid_layer_contract_fails_before_weight_allocation(layers):
    with pytest.raises(ValueError):
        get_target_feature_sources({"target_layer_ids": layers})


@pytest.mark.parametrize("sources", [[2, 0], [0, 1], ["final_postnorm", 0, 2], [False, 2], [0, 2, "final_prenorm"]])
def test_invalid_feature_order_or_semantics_are_rejected(sources):
    with pytest.raises(ValueError):
        get_target_feature_sources({"target_layer_ids": [0, 2], "target_feature_sources": sources})


def test_conflicting_modern_and_legacy_contracts_are_rejected():
    with pytest.raises(ValueError, match="conflicts"):
        get_target_feature_sources(
            {
                "target_layer_ids": [0],
                "target_feature_sources": [0],
                "loop_teacher": {"feature_sources": [0, "final_postnorm"]},
            }
        )
