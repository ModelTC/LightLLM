from dataclasses import fields
from types import SimpleNamespace

import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_sparse_backend_options_preserve_flashmla_only(phase, capsys):
    parser = make_argument_parser()
    option = f"llm_{phase}_att_backend"
    choices = next(field.metadata["choices"] for field in fields(StartArgs) if field.name == option)
    assert "flashmla_sparse" in choices
    assert "flashinfer_sparse" not in choices
    args = parser.parse_args([f"--{option}", "flashmla_sparse"])
    assert getattr(args, option) == "flashmla_sparse"
    with pytest.raises(SystemExit):
        parser.parse_args([f"--{option}", "flashinfer_sparse"])
    assert "invalid choice" in capsys.readouterr().err


BACKEND_OPTIONS = (
    "llm_prefill_att_backend",
    "llm_decode_att_backend",
    "llm_prefill_linear_att_backend",
    "llm_decode_linear_att_backend",
    "vit_att_backend",
)


@pytest.mark.parametrize("option", BACKEND_OPTIONS)
def test_backend_options_have_scalar_defaults_and_matching_choices(option):
    parser = make_argument_parser()
    field = next(field for field in fields(StartArgs) if field.name == option)
    action = next(action for action in parser._actions if action.dest == option)
    assert getattr(parser.parse_args([]), option) == getattr(StartArgs(), option) == "auto"
    assert action.choices == field.metadata["choices"]
    assert action.nargs is None


def test_backend_options_are_independent():
    args = make_argument_parser().parse_args(
        [
            "--llm_prefill_att_backend",
            "fa3",
            "--llm_decode_att_backend",
            "flashinfer",
            "--llm_prefill_linear_att_backend",
            "flashqla",
            "--llm_decode_linear_att_backend",
            "triton",
        ]
    )
    assert [getattr(args, option) for option in BACKEND_OPTIONS] == ["fa3", "flashinfer", "flashqla", "triton", "auto"]


@pytest.mark.parametrize("option", BACKEND_OPTIONS)
def test_backend_options_reject_multiple_values(option, capsys):
    with pytest.raises(SystemExit):
        make_argument_parser().parse_args([f"--{option}", "triton", "triton"])
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option", ["llm_prefill_att_backend", "llm_decode_att_backend", "llm_decode_linear_att_backend"]
)
def test_flashqla_is_only_a_linear_prefill_choice(option, capsys):
    with pytest.raises(SystemExit):
        make_argument_parser().parse_args([f"--{option}", "flashqla"])
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "decode,linear_decode,step,expected",
    [
        ("triton", "auto", 2, True),
        ("fa3", "triton", 2, False),
        ("not_triton", "triton", 2, False),
        ("triton", "triton", 0, False),
    ],
)
def test_mtp_kernel_uses_exact_regular_decode_backend(monkeypatch, decode, linear_decode, step, expected):
    from lightllm.utils import envs_utils

    monkeypatch.setattr(
        envs_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(
            mtp_step=step,
            llm_decode_att_backend=decode,
            llm_decode_linear_att_backend=linear_decode,
        ),
    )
    envs_utils.enable_triton_mtp_kernel.cache_clear()
    try:
        assert envs_utils.enable_triton_mtp_kernel() is expected
    finally:
        envs_utils.enable_triton_mtp_kernel.cache_clear()


@pytest.mark.parametrize("requested", ["auto", "triton", "flashqla"])
def test_kda_option_reaches_the_shared_selector(monkeypatch, requested):
    from lightllm.common.attention import selection
    from lightllm.common.attention.api import AttentionFamily
    from lightllm.common.basemodel import attention
    from lightllm.models.glm5_next import model as glm5
    from unit_tests.common.attention.helpers import make_spec

    calls = []
    sentinel = object()

    def select(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(create=lambda model, spec: sentinel)

    monkeypatch.setattr(selection, "select_attention_backend", select)
    monkeypatch.setattr(
        attention, "describe_attention_layer", lambda model, layer, family, phase: make_spec(family, phase)
    )
    model = glm5.Glm5NextTpPartModel.__new__(glm5.Glm5NextTpPartModel)
    from unit_tests.common.attention.helpers import model_stub

    vars(model).update(vars(model_stub(llm_prefill_linear_att_backend=requested)))
    model.layers_infer = [SimpleNamespace(get_attention_family=lambda: AttentionFamily.KDA)]
    model.prepare_attention_backends()
    assert calls[0]["requested"] == requested
    assert calls[0]["specs"][0].family is AttentionFamily.KDA
    assert model.attention_groups[0].prefill_backend is model.attention_groups[0].decode_backend is sentinel
