from types import SimpleNamespace

from lightllm.common.speculative.config import SpeculativeConfig
from lightllm.server import api_start


def test_block_mode_derives_mtp_step_from_checkpoint(monkeypatch):
    draft_config = {
        "architectures": ["Qwen3DSparkModel"],
        "block_size": 4,
        "target_layer_ids": [8, 16, 24],
        "mask_token_id": 151665,
        "markov_rank": 0,
        "enable_confidence_head": True,
        "confidence_head_with_markov": False,
    }
    monkeypatch.setattr(
        api_start.PretrainedConfig,
        "get_config_dict",
        lambda _model_dir: (draft_config, {}),
    )
    args = SimpleNamespace(mtp_draft_model_dir=["/models/dspark"], mtp_step=0)
    config = SpeculativeConfig(mode="dspark", step=0, dynamic_verify=True)

    normalized = api_start.normalize_block_mtp_step_from_first_draft_config(args, config)

    assert args.mtp_step == 4
    assert normalized.step == 4
    normalized.validate()


def test_non_block_mode_keeps_explicit_step(monkeypatch):
    monkeypatch.setattr(
        api_start.PretrainedConfig,
        "get_config_dict",
        lambda _model_dir: (_ for _ in ()).throw(AssertionError("must not load config")),
    )
    args = SimpleNamespace(mtp_draft_model_dir=["/models/eagle3"], mtp_step=3)
    config = SpeculativeConfig(mode="eagle3", step=3, dynamic_verify=True)

    assert api_start.normalize_block_mtp_step_from_first_draft_config(args, config) is config
    assert args.mtp_step == 3
