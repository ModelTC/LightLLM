from lightllm.common.attention_state.storage import allocate_state_storage
import pytest
import torch

from lightllm.common.attention_state.full_att import StaticFP8AttentionConfig, StaticFP8FullAttentionState


def test_shared_kv_keeps_distinct_query_scales_after_repacking_and_sharding():
    calibration = {
        "version": 2,
        "quant_type": "per_head",
        "layers": {
            "main": {
                "3": {"k": [1, 2, 3, 4], "v": [5, 6, 7, 8], "q": [9, 10, 11, 12]},
                "7": {"q": [13, 14, 15, 16]},
                "9": {"k": [17, 18, 19, 20], "v": [21, 22, 23, 24]},
            }
        },
    }
    for groups in (((3, 7), (9,)), ((9,), (3, 7))):
        state = StaticFP8FullAttentionState(
            StaticFP8AttentionConfig(groups, 4, 128, tp_rank=1, tp_world_size=2, scale_mode="per_head"), device="cpu"
        )
        state.bind_logical_calibration(calibration, "main")
        allocate_state_storage(state, 8)
        row = state.get_storage_layer_index(3)
        torch.testing.assert_close(state.scales[row], torch.tensor([3, 4, 7, 8], dtype=torch.float32))
        torch.testing.assert_close(state.get_query_scale(3), torch.tensor([11, 12], dtype=torch.float32))
        torch.testing.assert_close(state.get_query_scale(7), torch.tensor([15, 16], dtype=torch.float32))
        assert state.get_query_scale(9) is None


def test_conflicting_shared_calibration_fails_before_allocation():
    state = StaticFP8FullAttentionState(
        StaticFP8AttentionConfig(((3, 7),), 1, 128, scale_mode="per_head"), device="cpu"
    )
    calibration = {
        "version": 2,
        "quant_type": "per_head",
        "layers": {
            "main": {
                "3": {"k": [1], "v": [2]},
                "7": {"k": [3]},
            }
        },
    }
    with pytest.raises(ValueError, match="shared KV calibration differs"):
        state.bind_logical_calibration(calibration, "main")
    assert state.k_storage is None and state.scales is None
