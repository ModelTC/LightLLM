from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState, KDAState
from lightllm.common.attention_state.bounded_att import conv_ssm_state


@pytest.mark.parametrize("state_cls", [GDNState, KDAState])
@pytest.mark.parametrize("mtp_step,tp", [(0, 1), (2, 2)])
def test_conv_ssm_transfers(monkeypatch, pd_args, state_cls, mtp_step, tp):
    def copy_on_cpu(copies):
        for source, target in copies:
            target.copy_(source)

    monkeypatch.setattr(conv_ssm_state, "copy_bounded_att_tensors", copy_on_cpu)
    config = ConvSSMStateConfig(
        logical_layer_ids=(1, 3),
        global_k_heads=2,
        global_v_heads=4 if state_cls is GDNState else 2,
        key_dim=2,
        value_dim=3 if state_cls is GDNState else 2,
        conv_width=2,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
        tp_world_size=tp,
    )

    def make_state(steps):
        state = state_cls(config, request_capacity=3, mtp_step=steps, device="cpu")
        state.initialize_page_pools(2, 1, pin_memory=False)
        manager = AttentionStateManager(request_capacity=3)
        manager.register_state("main", state)
        manager.finish_state_registration()
        return state, manager

    source, manager = make_state(mtp_step)
    target, _ = make_state(2 if mtp_step == 0 else 0)
    source.conv_state.copy_(torch.arange(source.conv_state.numel()).reshape(source.conv_state.shape))
    source.ssm_state.copy_(torch.arange(source.ssm_state.numel()).reshape(source.ssm_state.shape))
    expected_conv = source.conv_state[:, 0, :, : config.conv_width].clone()
    expected_ssm = source.ssm_state[:, 0].clone()

    def reset_target():
        target.conv_state.fill_(-1)
        target.ssm_state.fill_(-2)
        if target.mtp_state_index is not None:
            target.mtp_state_index.fill_(1)

    def check_target():
        assert torch.equal(target.conv_state[:, 1, :, : config.conv_width], expected_conv)
        assert torch.equal(target.ssm_state[:, target.mtp_step + 1], expected_ssm)
        assert (target.conv_state[:, 0] == -1).all()
        assert (target.conv_state[:, 1, :, config.conv_width :] == -1).all()
        assert (target.ssm_state[:, : target.mtp_step + 1] == -2).all()
        if target.mtp_state_index is not None:
            assert target.mtp_state_index.tolist() == [1, 0, 1]

    source.save_bounded_att_runtime_to_pages([0], [8], "big", [1])
    cpu_layout = manager.get_cpu_cache_operator().get_page_layouts()[source.state_name]
    cpu_buffer = torch.zeros((2, cpu_layout.byte_end), dtype=torch.uint8)
    source.write_state_pages_to_cpu("big", [1], cpu_buffer, [1], cpu_layout)
    target.read_state_pages_from_cpu(cpu_buffer, [1], [0], cpu_layout)
    reset_target()
    target.restore_bounded_att_runtime_from_pages([1], [8], "big", [0])
    check_target()

    pd_layout = manager.get_pd_operator().get_pd_bounded_att_page_layouts()[source.state_name]
    pd_buffer = torch.zeros((2, pd_layout.byte_end), dtype=torch.uint8)
    source.write_to_pd_page(0, pd_buffer, 1, 8, pd_layout)
    reset_target()
    target.read_from_pd_page(pd_buffer, 1, 1, 8, pd_layout)
    check_target()

    reset_target()
    target.copy_runtime_state_from_dp([0], [1], source)
    check_target()
    reset_target()
    target.copy_big_pages_from_dp([1], [0], source)
    target.restore_bounded_att_runtime_from_pages([1], [8], "big", [0])
    check_target()

    other_cls = KDAState if state_cls is GDNState else GDNState
    other = other_cls(config, request_capacity=3, device="cpu")
    with pytest.raises(AssertionError, match="same algorithm"):
        target.copy_runtime_state_from_dp([0], [1], other)
    with pytest.raises(AssertionError, match="same algorithm"):
        target.copy_big_pages_from_dp([0], [0], other)
    if state_cls is KDAState:
        invalid = state_cls(replace(config, value_dim=3), request_capacity=3, device="cpu")
        with pytest.raises(AssertionError, match="equal Q/K/V"):
            invalid.initialize_page_pools(1, pin_memory=False)
