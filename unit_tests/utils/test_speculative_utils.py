import json
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.attention.base_att as base_att_module
import lightllm.common.basemodel.hidden_collector as hidden_collector_module
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.common.basemodel.attention.fa3.fp import Fa3DecodeAttState, Fa3PrefillAttState
from lightllm.common.basemodel.attention.fa3.mla import MlaFa3DecodeAttState, MlaFa3PrefillAttState
from lightllm.models import get_draft_model_class
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight
from lightllm.server.router.model_infer.speculative.proposers.dflash import DFlashProposer
from lightllm.utils import envs_utils


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("lightllm.models.deepseek_mtp.model", "Deepseek3MTPModel"),
        ("lightllm.models.glm4_moe_lite_mtp.model", "Glm4MoeLiteMTPModel"),
        ("lightllm.models.mistral_mtp.model", "MistralMTPModel"),
        ("lightllm.models.qwen3_moe_mtp.model", "Qwen3MOEMTPModel"),
        ("lightllm.models.qwen3_5_mtp.model", "Qwen3_5MTPModel"),
        ("lightllm.models.qwen3_5_moe_mtp.model", "Qwen3_5MoeMTPModel"),
        ("lightllm.models.qwen3_eagle.model", "Qwen3EagleModel"),
        ("lightllm.models.qwen3_dflash.model", "Qwen3DFlashModel"),
        ("lightllm.models.qwen3_5_dflash.model", "Qwen3_5DFlashModel"),
        ("lightllm.models.qwen3_dspark.model", "Qwen3DSparkModel"),
        ("lightllm.models.qwen3_5_dspark.model", "Qwen3_5DSparkModel"),
    ],
)
def test_spec_draft_model_class_is_marked(module_name, class_name):
    model_class = getattr(import_module(module_name), class_name)
    assert model_class.is_mtp_draft_model is True


def test_qwen3_eagle_uses_layers_checkpoint_prefix():
    layer_weight = Qwen3EagleTransformerLayerWeight.__new__(Qwen3EagleTransformerLayerWeight)
    layer_weight.layer_num_ = 2

    layer_weight._init_weight_names()

    assert layer_weight._q_weight_name == "layers.2.self_attn.q_proj.weight"
    assert layer_weight._hidden_norm_weight_name == "layers.2.hidden_norm.weight"


@pytest.mark.parametrize(
    "mtp_mode, is_draft_model, draft_step, dynamic_spec, expected",
    [
        ("dspark", False, 7, True, True),
        ("dspark", True, 7, True, False),
        ("dflash", False, 7, True, True),
        ("dflash", True, 7, True, False),
        ("vanilla_with_att", True, 7, True, False),
        ("vanilla_with_att", True, 0, True, False),
        ("eagle3", True, 0, True, False),
        ("eagle3", False, 7, True, True),
        ("eagle3", False, 7, False, False),
    ],
)
def test_attention_backend_selects_dynamic_spec_layout(
    monkeypatch,
    mtp_mode,
    is_draft_model,
    draft_step,
    dynamic_spec,
    expected,
):
    monkeypatch.setattr(
        base_att_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_mode=mtp_mode, mtp_dynamic_verify=dynamic_spec),
    )
    backend = SimpleNamespace(
        model=SimpleNamespace(
            is_mtp_draft_model=is_draft_model,
            mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: draft_step),
        )
    )

    assert BaseAttBackend.uses_dynamic_spec_verify_layout(backend) is expected


@pytest.mark.parametrize(
    "mtp_mode, is_draft_model, expected",
    [
        (None, False, True),
        ("dflash", False, True),
        ("dflash", True, False),
        ("dspark", True, False),
        ("eagle3", True, True),
        ("vanilla_with_att", True, True),
    ],
)
def test_attention_backend_selects_causality(monkeypatch, mtp_mode, is_draft_model, expected):
    monkeypatch.setattr(
        base_att_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_mode=mtp_mode),
    )
    backend = SimpleNamespace(model=SimpleNamespace(is_mtp_draft_model=is_draft_model))

    assert BaseAttBackend.uses_causal_attention(backend) is expected


@pytest.mark.parametrize("state_class", [Fa3PrefillAttState, MlaFa3PrefillAttState])
def test_fa3_prefill_state_owns_causality(state_class):
    infer_state = SimpleNamespace(
        b1_cu_q_seq_len=torch.tensor([0, 1, 2], dtype=torch.int32),
        b1_cu_kv_seq_len=torch.tensor([0, 3, 7], dtype=torch.int32),
        b_req_idx=torch.tensor([0, 1], dtype=torch.int32),
        batch_size=2,
        max_kv_seq_len=4,
        input_ids=torch.empty(2, dtype=torch.int64),
        req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(8, dtype=torch.int32).reshape(2, 4)),
    )
    state = state_class(
        backend=SimpleNamespace(uses_causal_attention=lambda: False),
        infer_state=infer_state,
    )

    state.init_state()

    assert state.causal is False


@pytest.mark.parametrize("state_class", [Fa3DecodeAttState, MlaFa3DecodeAttState])
def test_fa3_decode_state_owns_causality(state_class):
    infer_state = SimpleNamespace(
        b1_cu_q_seq_len=torch.tensor([0, 1, 2], dtype=torch.int32),
        b1_cu_kv_seq_len=torch.tensor([0, 3, 7], dtype=torch.int32),
        b_req_idx=torch.tensor([0, 1], dtype=torch.int32),
        b_seq_len=torch.tensor([3, 4], dtype=torch.int32),
    )
    model = SimpleNamespace(
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
    )
    state = state_class(
        backend=SimpleNamespace(
            model=model,
            uses_causal_attention=lambda: False,
            uses_dynamic_spec_verify_layout=lambda: False,
        ),
        infer_state=infer_state,
    )
    state._init_page_table = lambda _: None

    state.init_state()

    assert state.causal is False


@pytest.mark.parametrize(
    "model_type, spec_mode, expected_class_name",
    [
        ("deepseek_v3", "vanilla_with_att", "Deepseek3MTPModel"),
        ("deepseek_v3", "eagle_with_att", "Deepseek3MTPModel"),
        ("glm4_moe_lite", "vanilla_with_att", "Glm4MoeLiteMTPModel"),
        ("glm4_moe_lite", "eagle_with_att", "Glm4MoeLiteMTPModel"),
        ("mistral", "vanilla_no_att", "MistralMTPModel"),
        ("mistral", "eagle_no_att", "MistralMTPModel"),
        ("qwen3_moe", "vanilla_no_att", "Qwen3MOEMTPModel"),
        ("qwen3_moe", "eagle_no_att", "Qwen3MOEMTPModel"),
        ("qwen3_5", "vanilla_with_att", "Qwen3_5MTPModel"),
        ("qwen3_5_text", "eagle_with_att", "Qwen3_5MTPModel"),
        ("qwen3_5_moe", "vanilla_with_att", "Qwen3_5MoeMTPModel"),
        ("qwen3_5_moe_text", "eagle_with_att", "Qwen3_5MoeMTPModel"),
        ("qwen3", "dflash", "Qwen3DFlashModel"),
        ("qwen3_5", "dflash", "Qwen3_5DFlashModel"),
        ("qwen3_5_text", "dflash", "Qwen3_5DFlashModel"),
        ("qwen3", "dspark", "Qwen3DSparkModel"),
        ("qwen3_5", "dspark", "Qwen3_5DSparkModel"),
        ("qwen3_5_text", "dspark", "Qwen3_5DSparkModel"),
        ("qwen3", "eagle3", "Qwen3EagleModel"),
    ],
)
def test_draft_model_registry(model_type, spec_mode, expected_class_name):
    model_class = get_draft_model_class(
        model_cfg={"model_type": model_type},
        spec_mode=spec_mode,
    )

    assert model_class.__name__ == expected_class_name


def test_draft_model_registry_rejects_unsupported_model_type():
    with pytest.raises(ValueError, match="Unsupported speculative draft model"):
        get_draft_model_class(
            model_cfg={"model_type": "gemma4"},
            spec_mode="dspark",
        )


@pytest.mark.parametrize(
    "model_type, spec_mode",
    [
        ("deepseek_v3", "eagle3"),
        ("deepseek_v3", "dspark"),
        ("deepseek_v3", "dflash"),
        ("glm4_moe_lite", "eagle3"),
        ("glm4_moe_lite", "dspark"),
        ("glm4_moe_lite", "dflash"),
        ("qwen3_5", "eagle3"),
        ("qwen3_5_moe", "eagle3"),
        ("qwen3_5_moe", "dspark"),
        ("qwen3_5_moe", "dflash"),
        ("qwen3", "eagle_no_att"),
    ],
)
def test_draft_model_registry_rejects_unsupported_mode(model_type, spec_mode):
    with pytest.raises(ValueError, match="Unsupported speculative draft model"):
        get_draft_model_class(
            model_cfg={"model_type": model_type},
            spec_mode=spec_mode,
        )


def test_dflash_dynamic_verify_uses_fixed_block_token_probabilities():
    block_size = 4
    max_draft_step = 3
    verify_row_count = 5
    accepted_tail_rows = torch.tensor([0, 3])
    flat_draft_token_ids = torch.arange(2 * block_size)
    flat_draft_token_probs = torch.arange(2 * block_size, dtype=torch.float32) / 10

    draft_model = SimpleNamespace(
        block_size=block_size,
        forward=lambda _: SimpleNamespace(logits=torch.empty(2 * block_size, 1)),
    )
    backend = SimpleNamespace(
        max_draft_step=max_draft_step,
        draft_models=[draft_model],
        _gen_argmax_token_ids_and_prob=lambda _: (flat_draft_token_ids, flat_draft_token_probs),
    )
    proposer = DFlashProposer(backend=backend, enable_dynmaic_mtp=True)
    proposer.extend_draft_kv_cache = lambda **_: None
    proposer.build_block_draft_input = lambda **_: (SimpleNamespace(), torch.tensor([10, 11]))

    proposal = proposer.propose_next(
        main_model_input=SimpleNamespace(),
        main_model_output=SimpleNamespace(spec_hidden=torch.empty(verify_row_count, 1)),
        next_token_ids=torch.arange(verify_row_count),
        b_req_mtp_start_loc=torch.tensor([0, 3]),
        draft_step=2,
        accept_len=torch.tensor([1, 1]),
    )

    expected_blocks = flat_draft_token_ids.reshape(2, block_size)[:, :2]
    torch.testing.assert_close(proposal.token_ids[accepted_tail_rows, 1:], expected_blocks)
    assert proposal.schedule_scores.shape == (verify_row_count, 2)
    for step in range(2):
        scores = proposal.schedule_scores[:, step]
        expected_scores = torch.zeros(verify_row_count)
        expected_scores[accepted_tail_rows] = flat_draft_token_probs.reshape(2, block_size)[:, step]
        torch.testing.assert_close(scores, expected_scores)


def test_dflash_reuses_decode_input_for_kv_commit():
    forwarded_inputs = []
    draft_model = SimpleNamespace(forward=forwarded_inputs.append)
    proposer = DFlashProposer(
        backend=SimpleNamespace(draft_models=[draft_model]),
        enable_dynmaic_mtp=False,
    )
    input_ids = torch.arange(4)
    multimodal_params = [{"images": [], "audios": []}] * 4
    position_delta = torch.zeros(4, dtype=torch.int32)
    model_input = SimpleNamespace(
        batch_size=4,
        total_token_num=20,
        prefix_total_token_num=None,
        input_ids=input_ids,
        b_seq_len=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        b_position_delta=position_delta,
        multimodal_params=multimodal_params,
        is_prefill=False,
    )
    target_hidden = torch.empty(4, 16)

    proposer.extend_draft_kv_cache(model_input, target_hidden)

    assert len(forwarded_inputs) == 1
    assert forwarded_inputs[0] is model_input
    assert model_input.input_ids is input_ids
    assert model_input.multimodal_params is multimodal_params
    assert model_input.total_token_num == model_input.batch_size
    assert model_input.prefix_total_token_num == 0
    assert model_input.is_prefill
    torch.testing.assert_close(model_input.b_ready_cache_len, model_input.b_seq_len - 1)
    torch.testing.assert_close(model_input.b_prefill_start_loc, torch.arange(4, dtype=torch.int32))
    assert model_input.b_position_delta is position_delta
    assert model_input.mtp_draft_input_hiddens is target_hidden


def test_dflash_expands_position_delta_with_request_block_rows():
    block_size = 3
    proposer = DFlashProposer(
        backend=SimpleNamespace(
            draft_models=[SimpleNamespace(block_size=block_size, mask_token_id=99)],
        ),
        enable_dynmaic_mtp=False,
    )
    proposer.alloc_extra_mem_indexes = lambda token_count: torch.arange(token_count, dtype=torch.int32)
    model_input = SimpleNamespace(
        b_req_idx=torch.tensor([10, 10, 11, 12, 12], dtype=torch.int32),
        b_seq_len=torch.tensor([4, 5, 7, 8, 9], dtype=torch.int32),
        b_position_delta=torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32),
        max_kv_seq_len=9,
    )

    draft_input, _ = proposer.build_block_draft_input(
        main_model_input=model_input,
        next_token_ids=torch.arange(5, dtype=torch.int64),
        accepted_tail_rows=torch.tensor([1, 4]),
        request_count=2,
    )

    assert torch.equal(
        draft_input.b_position_delta,
        torch.tensor([11, 11, 11, 21, 21, 21], dtype=torch.int32),
    )
    assert torch.equal(draft_input.b_seq_len, torch.tensor([6, 7, 8, 10, 11, 12], dtype=torch.int32))


def test_hidden_collector_reads_target_layer_ids(monkeypatch):
    config_reads = []

    def get_config_dict(path):
        config_reads.append(path)
        return {"target_layer_ids": [1, 20, 36]}, {}

    monkeypatch.setattr(
        hidden_collector_module.PretrainedConfig,
        "get_config_dict",
        get_config_dict,
    )
    monkeypatch.setattr(
        hidden_collector_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_draft_model_dir=["/models/dspark"]),
    )
    model = SimpleNamespace(is_mtp_draft_model=False, layers_num=40)

    collector = hidden_collector_module.LayerHiddenCollector(model=model)
    new_collector = collector.new_instance()

    assert collector.layer_ids == frozenset((1, 20, 36))
    assert new_collector.layer_ids == collector.layer_ids
    assert config_reads == ["/models/dspark"]


def test_dflash_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 5}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "dflash",
            "mtp_step": 7,
            "mtp_dynamic_verify": False,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 5


def test_qwen35_dflash_added_kv_layers_come_from_nested_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "num_hidden_layers": 48,
                "dflash_config": {"num_hidden_layers": 5},
            }
        )
    )

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "dflash",
            "mtp_step": 7,
            "mtp_dynamic_verify": False,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 5


def test_dspark_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 6}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "dspark",
            "mtp_step": 7,
            "mtp_dynamic_verify": True,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 6


def test_eagle3_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 2}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "eagle3",
            "mtp_step": 7,
            "mtp_dynamic_verify": False,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 2
