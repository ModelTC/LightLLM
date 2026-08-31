# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightllm.models.deepseek3_2.triton_kernel.extract_indexer_ks import (
    extract_indexer_ks,
    extract_indexer_ks_dynamic,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _indexer_buffer(num_slots: int):
    keys = (
        torch.arange(num_slots * 128, device="cuda", dtype=torch.float32)
        .remainder_(31)
        .sub_(15)
        .to(torch.float8_e4m3fn)
        .view(num_slots, 1, 128)
    )
    scales = torch.arange(1, num_slots + 1, device="cuda", dtype=torch.float32).view(num_slots, 1, 1)
    buffer = torch.empty((num_slots, 1, 132), device="cuda", dtype=torch.uint8)
    buffer[:, :, :128].copy_(keys.view(torch.uint8))
    buffer[:, :, 128:132].copy_(scales.view(torch.uint8).view(num_slots, 1, 4))
    return buffer, keys.view(num_slots, 128), scales.view(num_slots)


def _assert_packed(output_keys, output_scales, source_keys, source_scales, expected_slots):
    expected_slots = torch.tensor(expected_slots, device="cuda", dtype=torch.int64)
    expected_keys = source_keys.index_select(0, expected_slots)
    expected_scales = source_scales.index_select(0, expected_slots)
    torch.testing.assert_close(
        output_keys[: len(expected_slots)].float(),
        expected_keys.float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output_scales[: len(expected_slots)],
        expected_scales,
        rtol=0,
        atol=0,
    )


def test_dynamic_layout_matches_fixed_full_width_layout():
    buffer, source_keys, source_scales = _indexer_buffer(8)
    req_to_token = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    b_req_idx = torch.tensor([0, 0, 0, 1, 1, 1], device="cuda", dtype=torch.int32)
    b_mtp_index = torch.tensor([0, 1, 2, 0, 1, 2], device="cuda", dtype=torch.int32)
    b_seq_len = torch.tensor([2, 3, 4, 1, 2, 3], device="cuda", dtype=torch.int32)

    fixed_keys, fixed_scales = extract_indexer_ks(
        I_buffer=buffer,
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        req_to_token_indexs=req_to_token,
        out_token_num=24,
        max_kv_seq_len=4,
        mtp_step=2,
    )
    dynamic_keys, dynamic_scales = extract_indexer_ks_dynamic(
        I_buffer=buffer,
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        req_to_token_indexs=req_to_token,
        max_kv_seq_len=4,
        max_request_num=2,
    )

    torch.testing.assert_close(dynamic_keys[:7].float(), fixed_keys[:7].float(), rtol=0, atol=0)
    torch.testing.assert_close(dynamic_scales[:7], fixed_scales[:7], rtol=0, atol=0)


def test_dynamic_layout_packs_variable_request_widths():
    buffer, source_keys, source_scales = _indexer_buffer(12)
    req_to_token = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        device="cuda",
        dtype=torch.int32,
    )
    b_req_idx = torch.tensor([0, 0, 0, 1, 2], device="cuda", dtype=torch.int32)
    b_mtp_index = torch.tensor([0, 1, 2, 0, 0], device="cuda", dtype=torch.int32)
    b_seq_len = torch.tensor([2, 3, 4, 3, 2], device="cuda", dtype=torch.int32)

    output_keys, output_scales = extract_indexer_ks_dynamic(
        I_buffer=buffer,
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        req_to_token_indexs=req_to_token,
        max_kv_seq_len=4,
        max_request_num=3,
    )

    _assert_packed(
        output_keys,
        output_scales,
        source_keys,
        source_scales,
        expected_slots=[0, 1, 2, 3, 4, 5, 6, 8, 9],
    )
