import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.utils.envs_utils import _get_mtp_draft_backbone_layer_num
from lightllm.utils.config_utils import get_deepseek_v4_compress_rates


@pytest.mark.parametrize("mtp_step", [1, 4, 5])
def test_deepseek_v4_dspark_runtime_width_follows_mtp_step(monkeypatch, mtp_step):
    from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    monkeypatch.setattr(DeepseekV4TpPartModel, "_verify_params", lambda self: None)
    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    model.args = SimpleNamespace(mtp_step=mtp_step)
    model.config = {"block_size": 5, "dspark_block_size": 5}
    model.enable_tpsp_mix_mode = False

    model._verify_params()

    assert model.config["block_size"] == mtp_step


@pytest.mark.parametrize("mtp_step", [0, 6])
def test_deepseek_v4_dspark_rejects_invalid_runtime_width(monkeypatch, mtp_step):
    from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    monkeypatch.setattr(DeepseekV4TpPartModel, "_verify_params", lambda self: None)
    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    model.args = SimpleNamespace(mtp_step=mtp_step)
    model.config = {"block_size": 5, "dspark_block_size": 5}
    model.enable_tpsp_mix_mode = False

    with pytest.raises(AssertionError, match=r"requires --mtp_step in \[1, 5\]"):
        model._verify_params()


@pytest.mark.parametrize("schema", ["compress_ratios", "layer_types"])
def test_deepseek_v4_dspark_layer_count_uses_trailing_swa_layers(tmp_path, schema):
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "compress_ratios": [4] * 43 + [0, 0, 0],
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [40, 41, 42],
    }
    if schema == "layer_types":
        del config["compress_ratios"]
        config["layer_types"] = ["compressed_attention"] * 43
        config["compress_rates"] = {"compressed_attention": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert _get_mtp_draft_backbone_layer_num(str(tmp_path)) == 3


def test_deepseek_v4_dspark_extends_target_compress_rates_for_draft_layers():
    target_rates = [0, 0] + [value for _ in range(20) for value in (4, 128)] + [4]
    config = {
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "compress_ratios": target_rates + [0],
    }

    rates = get_deepseek_v4_compress_rates(config, layer_num=46)

    assert rates == target_rates + [0, 0, 0]


def test_dspark_cuda_graph_padding_keeps_complete_hold_blocks():
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    block_size = 5
    batch_size = 14 * block_size
    graph_batch_size = 16 * block_size
    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    model.block_size = block_size
    model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=127)
    model.mem_manager = SimpleNamespace(HOLD_TOKEN_MEMINDEX=255)
    model_input = ModelInput(
        batch_size=batch_size,
        total_token_num=batch_size,
        max_q_seq_len=1,
        max_kv_seq_len=8,
        input_ids=torch.ones(batch_size, dtype=torch.int64),
        b_req_idx=torch.arange(batch_size, dtype=torch.int32),
        b_mtp_index=torch.zeros(batch_size, dtype=torch.int32),
        b_seq_len=torch.full((batch_size,), 8, dtype=torch.int32),
        b_position_delta=torch.zeros(batch_size, dtype=torch.int32),
        b_shared_seq_len=torch.zeros(batch_size, dtype=torch.int32),
        b_shared_radix_node_id=torch.full((batch_size,), -1, dtype=torch.int64),
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
    )

    padded_input = model._create_padded_decode_model_input(model_input, graph_batch_size)

    assert padded_input.batch_size == graph_batch_size
    torch.testing.assert_close(padded_input.b_req_idx[:batch_size], model_input.b_req_idx)
    assert padded_input.b_req_idx[batch_size:].eq(127).all()


def test_dspark_empty_decode_padding_builds_one_hold_block():
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    block_size = 5
    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    model.block_size = block_size
    model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=127)
    model.mem_manager = SimpleNamespace(HOLD_TOKEN_MEMINDEX=255)
    model_input = ModelInput(
        batch_size=0,
        total_token_num=0,
        max_q_seq_len=1,
        max_kv_seq_len=0,
        input_ids=torch.empty((0,), dtype=torch.int64),
        b_req_idx=torch.empty((0,), dtype=torch.int32),
        b_mtp_index=torch.empty((0,), dtype=torch.int32),
        b_seq_len=torch.empty((0,), dtype=torch.int32),
        b_position_delta=torch.empty((0,), dtype=torch.int32),
        b_shared_seq_len=torch.empty((0,), dtype=torch.int32),
        b_shared_radix_node_id=torch.empty((0,), dtype=torch.int64),
        is_prefill=False,
        multimodal_params=[],
    )

    padded_input = model._create_padded_decode_model_input(model_input, 1)

    assert model_input.batch_size == 0
    assert padded_input.batch_size == block_size
    assert padded_input.total_token_num == 2 * block_size
    assert padded_input.input_ids.tolist() == [1] * block_size
    assert padded_input.b_req_idx.tolist() == [127] * block_size
    assert padded_input.b_seq_len.tolist() == [2] * block_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_dspark_swa_index_exposes_history_and_complete_block():
    from lightllm.models.deepseek_v4.triton_kernel.build_dspark_swa_index import (
        build_dspark_swa_index,
    )

    block_size = 3
    window = 4
    req_to_swa_pages = torch.tensor([[1], [2], [3]], dtype=torch.int32, device="cuda")
    req_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device="cuda")
    positions = torch.tensor([4, 5, 6, 2, 3, 4], dtype=torch.int32, device="cuda")
    padded_width = 8
    indices = torch.empty((6, padded_width), dtype=torch.int32, device="cuda")
    lengths = torch.empty((6,), dtype=torch.int32, device="cuda")
    write_slots = torch.empty((6,), dtype=torch.int32, device="cuda")

    build_dspark_swa_index(
        req_idx=req_idx,
        positions=positions,
        req_to_swa_pages=req_to_swa_pages,
        swa_index=indices,
        swa_length=lengths,
        swa_write_slots=write_slots,
        window=window,
        block_size=block_size,
        page_size=128,
        hold_req_id=2,
        hold_swa_slot=120,
    )

    expected_indices = torch.tensor(
        [
            [131, 130, 129, 128, 132, 133, 134, -1],
            [131, 130, 129, 128, 132, 133, 134, -1],
            [131, 130, 129, 128, 132, 133, 134, -1],
            [257, 256, 258, 259, 260, -1, -1, -1],
            [257, 256, 258, 259, 260, -1, -1, -1],
            [257, 256, 258, 259, 260, -1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(lengths, torch.tensor([7, 7, 7, 5, 5, 5], dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(
        write_slots,
        torch.tensor([132, 133, 134, 258, 259, 260], dtype=torch.int32, device="cuda"),
    )
