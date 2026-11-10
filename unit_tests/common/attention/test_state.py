from dataclasses import replace
import pytest
import torch
from lightllm.common.attention_state.commit import BoundedStateCommit
from lightllm.common.attention_state.manager import AttentionStateManager
from lightllm.common.attention_state.bounded_att import GDNState, KDAState, ConvSSMStateConfig


def test_commit_owns_mapping_and_detects_recycled_requests():
    requests = torch.tensor([2, 2, 4], dtype=torch.int32)
    steps = torch.tensor([0, 1, 0], dtype=torch.int32)
    generations = torch.zeros(8, dtype=torch.int64)
    output = BoundedStateCommit.capture("main", requests, steps, generations)
    requests.fill_(6)
    steps.fill_(5)
    assert output.request_indices.tolist() == [2, 2, 4]
    assert output.mtp_indices.tolist() == [0, 1, 0]
    output.validate_generation(generations)
    generations[2] += 1
    with pytest.raises(RuntimeError, match="reused request slot"):
        output.validate_generation(generations)


def test_overlap_commit_preserves_microbatch_order_and_role():
    generation = torch.zeros(8, dtype=torch.int64)
    a = BoundedStateCommit.capture("main", torch.tensor([4, 4]), torch.tensor([0, 1]), generation)
    b = BoundedStateCommit.capture("main", torch.tensor([2]), torch.tensor([0]), generation)
    merged = BoundedStateCommit.concatenate((a, b))
    assert merged.request_indices.tolist() == [4, 4, 2]
    assert merged.mtp_indices.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="same model role"):
        BoundedStateCommit.concatenate((a, replace(b, role="draft0")))


@pytest.mark.parametrize("state_cls", [GDNState, KDAState])
def test_capture_snapshot_restores_only_participating_role_and_requests(state_cls):
    config = ConvSSMStateConfig(
        logical_layer_ids=(0,),
        global_k_heads=1,
        global_v_heads=1,
        key_dim=2,
        value_dim=2,
        conv_width=2,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
    )
    manager = AttentionStateManager(request_capacity=3)
    main = state_cls(config, request_capacity=3, mtp_step=2, device="cpu")
    draft = state_cls(config, request_capacity=3, mtp_step=2, device="cpu")
    manager.register_state("main", main)
    manager.register_state("draft0", draft)
    manager.finish_state_registration()
    main.conv_state.fill_(2)
    main.ssm_state.fill_(3)
    main.mtp_state_index.fill_(1)
    snapshot = manager.snapshot_capture_state(torch.tensor([1, 1]), role="main")
    main.conv_state[:, 1].fill_(8)
    main.ssm_state[:, 3:6].fill_(9)
    main.mtp_state_index[1] = 2
    main.conv_state[:, 2].fill_(6)
    draft.conv_state.fill_(7)
    manager.restore_capture_state(snapshot)
    assert torch.all(main.conv_state[:, 1] == 2)
    assert torch.all(main.ssm_state[:, 3:6] == 3)
    assert main.mtp_state_index[1] == 1
    assert torch.all(main.conv_state[:, 2] == 6)
    assert torch.all(draft.conv_state == 7)
