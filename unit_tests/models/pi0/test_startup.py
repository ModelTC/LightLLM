from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.models.registry import get_output_modality
from lightllm.server.generationserver.action import configure_defaults


def test_non_pi_model_does_not_select_pi_backend():
    assert get_output_modality({"model_type": "qwen3_vl"}) is None


def test_pi_models_register_the_action_backend():
    assert get_output_modality({"model_type": "pi0"}) == "action"
    assert get_output_modality({"model_type": "pi05"}) == "action"


def test_pi_defaults_select_the_model_owned_backend(monkeypatch):
    args = StartArgs(model_dir="/unused", generation_backend=None)
    config = {"action_horizon": 50}
    monkeypatch.setattr(
        "lightllm.server.generationserver.action.load_action_config",
        lambda _args: config,
    )

    configure_defaults(args)
    assert args.generation_backend == "action"
    assert args.generation_runtime_config is config
    assert args.disable_dynamic_prompt_cache and args.disable_chunked_prefill
