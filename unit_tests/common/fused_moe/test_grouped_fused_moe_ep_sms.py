from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep.common import get_ep_num_sms


def test_get_ep_num_sms_selects_overlap_quota(monkeypatch):
    import lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep.common as common

    monkeypatch.setattr(common.dist_group_manager, "ep_num_sms", 20, raising=False)
    monkeypatch.setattr(common.dist_group_manager, "ep_non_overlap_num_sms", 64, raising=False)

    assert get_ep_num_sms(overlap_with_compute=True) == 20
    assert get_ep_num_sms(overlap_with_compute=False) == 64
