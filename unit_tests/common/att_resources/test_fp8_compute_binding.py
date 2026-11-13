import pytest
import torch

from unit_tests.common.att_resources.model_helpers import bind_test_fp8_calibration


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_num", [1, 3])
@pytest.mark.parametrize("scale_mode", ["per_tensor", "per_head"])
def test_fp8_split_storage_matches_calibrated_values_and_combined_kernel(head_num: int, scale_mode: str):
    from lightllm.common.attention_state.full_att import StaticFP8AttentionConfig, StaticFP8FullAttentionState
    from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8

    scale_count = 2 if scale_mode == "per_tensor" else 2 * head_num
    scales = (
        torch.arange(
            start=1,
            end=scale_count + 1,
            dtype=torch.float32,
            device="cuda",
        ).unsqueeze(0)
        / 4
    )
    state = StaticFP8FullAttentionState(
        layout_config=StaticFP8AttentionConfig(
            logical_layer_groups=((5,),),
            global_head_num=head_num,
            head_dim=128,
            scale_mode=scale_mode,
        ),
        page_size=4,
        device="cuda",
    )
    bind_test_fp8_calibration(
        state=state,
        scales=scales,
        q_scales=torch.ones((1, head_num), dtype=torch.float32, device="cuda") if scale_mode == "per_head" else None,
    )
    state.allocate_storage(8)
    state.k_storage.zero_()
    state.v_storage.zero_()
    indices = torch.tensor([1, 6], dtype=torch.int32, device="cuda")
    kv = torch.randn((2, 2 * head_num, 128), dtype=torch.bfloat16, device="cuda")
    state.write_full_att_kv(logical_layer_id=5, token_indices=indices, kv=kv)
    k, v = state.get_layer_view(5)
    assert k.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
    actual = torch.cat((k[indices], v[indices]), dim=1)
    head_scales = scales[0].repeat_interleave(head_num) if scale_mode == "per_tensor" else scales[0]
    expected = (kv.float() / head_scales[None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(actual, expected.view(torch.uint8))
    assert not k[[0, 2, 3, 4, 5, 7, 8, 9, 10, 11]].any()
    assert not v[[0, 2, 3, 4, 5, 7, 8, 9, 10, 11]].any()

    # 算子的原合并输出入口仍应保持相同的量化结果。
    combined = torch.zeros((12, 2 * head_num, 128), dtype=torch.uint8, device="cuda")
    destindex_copy_kv_fp8(
        KV=kv,
        DestLoc=indices,
        scales=scales[0],
        Out=combined.view(torch.float8_e4m3fn),
        is_per_tensor_quant=scale_mode == "per_tensor",
    )
    assert torch.equal(combined[indices], actual)
