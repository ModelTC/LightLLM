import asyncio
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.core.objs import ConstraintOutParams, SamplingParams
from lightllm.server.core.objs.py_sampling_params import SamplingParams as PySamplingParams
from lightllm.server.router.model_infer.structured_output.state import ConstraintState
from lightllm.server.router.model_infer.structured_output.state_factory import OutputConstraintStateFactory


def make_state(compiler, budget, marker=b"]>", grammar=None):
    params = ConstraintOutParams()
    params.initialize(None, "", "", "", list(marker), thinking_token_budget=budget)
    compiled = None
    if grammar is not None:
        payload = asyncio.run(compiler.compile("regex", grammar))
        compiled = compiler.output_constraint_state_factory.get_grammar(payload)
    return ConstraintState.from_grammar(compiled, params)


def next_tokens(state, drafts=()):
    mask = torch.empty((len(drafts) + 1, 9), dtype=torch.int32)
    state.fill_masks(mask, drafts)
    ids = torch.arange(257)
    return ((mask[:, ids // 32] >> (ids % 32)) & 1).bool()


@pytest.mark.parametrize("grammar", [None, "ab"])
@pytest.mark.parametrize("budget", [0, 2])
def test_budget_forces_end_before_grammar(compiler, budget, grammar):
    state = make_state(compiler, budget, grammar=grammar)
    for token in b"xy"[:budget]:
        assert next_tokens(state).all()
        state.commit(token)
    for token in b"]>":
        assert next_tokens(state)[0].nonzero().flatten().tolist() == [token]
        state.commit(token)
        assert state.error is None
    assert not state.in_reasoning
    assert state.reasoning_tokens == budget + 2
    assert not state.is_terminated()
    if grammar is None:
        assert next_tokens(state).all()
        state.commit(ord("!"))
    else:
        for token in b"ab":
            assert next_tokens(state)[0].nonzero().flatten().tolist() == [token]
            state.commit(token)
        state.commit(256)
        assert state.is_terminated()
    assert state.reasoning_tokens == budget + 2
    assert state.error is None


def test_natural_end_and_partial_end_at_budget(compiler):
    state = make_state(compiler, 10, grammar="ab")
    for token in b"x]>":
        state.commit(token)
    assert next_tokens(state)[0].nonzero().flatten().tolist() == [ord("a")]

    state = make_state(compiler, 2, grammar="ab")
    for token in b"x]":
        state.commit(token)
    # Finish a naturally started delimiter instead of emitting a duplicate prefix.
    assert next_tokens(state)[0].nonzero().flatten().tolist() == [ord(">")]


@pytest.mark.parametrize("grammar", [None, "ab"])
def test_mtp_budget_crossing_rolls_back_and_rejects_invalid_draft(compiler, grammar):
    state = make_state(compiler, 2, grammar=grammar)
    state.commit(ord("x"))
    mask = next_tokens(state, b"y]>a")
    assert mask[0].all()
    assert mask[1].nonzero().flatten().tolist() == [ord("]")]
    assert mask[2].nonzero().flatten().tolist() == [ord(">")]
    if grammar is not None:
        assert mask[3].nonzero().flatten().tolist() == [ord("a")]
        assert mask[4].nonzero().flatten().tolist() == [ord("b")]
    else:
        assert mask[3:].all()
    assert state.in_reasoning and state.reasoning_tokens == 1
    assert state.reasoning_tail == (ord("x"),)
    assert state.error is None

    mask = next_tokens(state, b"y!aa")
    assert mask[1].nonzero().flatten().tolist() == [ord("]")]
    assert mask[2:].all()
    assert state.reasoning_tokens == 1 and state.error is None
    for token in b"y]>":
        state.commit(token)
    assert not state.in_reasoning and state.error is None


@pytest.mark.parametrize("budget", [None, -1, 0, 12])
def test_budget_sampling_params_round_trip(budget):
    params = SamplingParams()
    params.init(None, thinking_token_budget=budget, guided_reasoning_end=[248069])
    restored = SamplingParams()
    restored.init(None, **params.to_dict())
    for candidate in (params, restored, SamplingParams.from_buffer_copy(params)):
        assert candidate.constraint_out_params.thinking_token_budget == (-1 if budget is None else budget)
        assert candidate.constraint_out_params.has_constraint() == (budget is not None and budget >= 0)
        assert not candidate.constraint_out_params.has_grammar_constraint()
    py_params = PySamplingParams(
        thinking_token_budget=budget,
        guided_reasoning_end=[248069],
        regular_constraint="",
        guided_grammar="",
        guided_json="",
    )
    restored.init(None, **py_params.to_dict())
    assert restored.constraint_out_params.thinking_token_budget == py_params.thinking_token_budget


@pytest.mark.parametrize("budget", [True, 1.5, "2", -2, 2 ** 31])
def test_invalid_budget_rejected_before_ctypes_conversion(budget):
    params = SamplingParams()
    with pytest.raises(ValueError, match="thinking_token_budget"):
        params.init(None, thinking_token_budget=budget, guided_reasoning_end=[248069])
    with pytest.raises(ValueError, match="thinking_token_budget"):
        PySamplingParams(thinking_token_budget=budget, guided_reasoning_end=[248069]).verify()


def test_budget_requires_reasoning_end():
    params = SamplingParams()
    with pytest.raises(ValueError, match="guided_reasoning_end"):
        params.init(None, thinking_token_budget=2)
    with pytest.raises(ValueError, match="guided_reasoning_end"):
        PySamplingParams(thinking_token_budget=2).verify()


@pytest.mark.parametrize("marker", [[-1], [True], [1.5], ["7"], [2 ** 31], None])
def test_invalid_reasoning_end_ids_rejected_before_ctypes_conversion(marker):
    params = SamplingParams()
    with pytest.raises(ValueError, match="guided_reasoning_end"):
        params.init(None, thinking_token_budget=0, guided_reasoning_end=marker)


@pytest.mark.parametrize("marker", [[31], [255], list(b"aba")])
def test_forced_marker_supports_signed_mask_words_and_overlapping_prefixes(compiler, marker):
    state = make_state(compiler, 0, marker=marker)
    for token in marker:
        assert next_tokens(state)[0].nonzero().flatten().tolist() == [token]
        state.commit(token)
    assert state.error is None and not state.in_reasoning


@pytest.mark.parametrize("history", [b"x", b"xy", b"xy]", b"xy]>a"])
@pytest.mark.parametrize("grammar", [None, "ab"])
def test_pd_replay_restores_budget_and_partial_forced_end(compiler, history, grammar):
    params = SamplingParams()
    params.init(
        None,
        thinking_token_budget=2,
        guided_reasoning_end=list(b"]>"),
        pd_previous_output_len=len(history),
    )
    payload = asyncio.run(compiler.compile("regex", grammar)) if grammar else None
    req = SimpleNamespace(
        input_len=len(history) + 1,
        shm_prompt_ids=SimpleNamespace(arr=[ord("P"), *history]),
        get_compiled_grammar=lambda: payload,
    )
    factory = compiler.output_constraint_state_factory if grammar else OutputConstraintStateFactory(None, 257, [256])
    state = factory.create_state(req, SimpleNamespace(shm_param=params))
    reference = make_state(compiler, 2, grammar=grammar)
    for token in history:
        reference.commit(token)
    assert state.error is None
    assert state.in_reasoning == reference.in_reasoning
    assert state.reasoning_tokens == reference.reasoning_tokens
    torch.testing.assert_close(next_tokens(state), next_tokens(reference))


def test_unlimited_budget_does_not_force_end(compiler):
    state = make_state(compiler, -1)
    for token in b"unlimited thinking":
        state.commit(token)
    assert next_tokens(state).all()
    assert state.reasoning_tokens == len(b"unlimited thinking")
    assert state.error is None and state.in_reasoning


def test_unbudgeted_grammar_preserves_reasoning_and_grammar_progress(compiler):
    state = make_state(compiler, -1, grammar="ab")
    state.commit(ord("x"))
    next_tokens(state, b"]>a")
    assert state.in_reasoning and state.reasoning_tail == (ord("x"),)
    assert state.reasoning_tokens == 1
    assert next_tokens(state).all()
    for token in b"]>ab":
        state.commit(token)
    state.commit(256)
    assert state.error is None and state.is_terminated()


def test_thinking_budget_cli_defaults_to_enabled():
    from lightllm.server.api_cli import make_argument_parser
    from lightllm.server.core.objs import StartArgs

    parser = make_argument_parser()
    assert not StartArgs().disable_thinking_budget
    assert not parser.parse_args([]).disable_thinking_budget
    assert parser.parse_args(["--disable_thinking_budget"]).disable_thinking_budget
    with pytest.raises(SystemExit):
        parser.parse_args(["--enable_thinking_budget"])


def test_budget_rejects_excluded_end_token():
    params = SamplingParams()
    with pytest.raises(ValueError, match="logit_bias"):
        params.init(None, thinking_token_budget=1, guided_reasoning_end=[7], logit_bias={"7": -100})


@pytest.mark.parametrize("grammar_enabled", [False, True])
def test_budget_rejects_disabled_service(monkeypatch, grammar_enabled):
    from lightllm.server.core.objs import sampling_params

    monkeypatch.setattr(
        sampling_params,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_thinking_budget=True),
    )
    params = SamplingParams()
    params.init(None, thinking_token_budget=1, guided_reasoning_end=[7])
    with pytest.raises(ValueError, match="--disable_thinking_budget"):
        asyncio.run(params.verify_cpu_async(object() if grammar_enabled else None))


@pytest.mark.parametrize("parser", ["qwen3", "deepseek-r1", "deepseek-v4", "gemma4"])
def test_budget_request_accepts_reasoning_parsers(monkeypatch, parser, tmp_path):
    from lightllm.server.core.objs import sampling_params
    from lightllm.server.core.objs import StartArgs

    (tmp_path / "config.json").write_text('{"vocab_size": 257}')
    args = StartArgs(reasoning_parser=parser, model_dir=str(tmp_path))
    monkeypatch.setattr(sampling_params, "get_env_start_args", lambda: args)
    params = SamplingParams()
    params.init(None, thinking_token_budget=2, guided_reasoning_end=[7])
    assert asyncio.run(params.get_compiled_grammar(None)) is None


@pytest.mark.parametrize("parser", ["gpt-oss", "GPT-OSS"])
@pytest.mark.parametrize("budget", [0, 2])
@pytest.mark.parametrize("grammar_enabled", [False, True])
def test_gpt_oss_budget_rejected_after_serialization(monkeypatch, parser, budget, grammar_enabled):
    from lightllm.server.core.objs import StartArgs

    args = StartArgs(reasoning_parser=parser)
    monkeypatch.setattr("lightllm.server.core.objs.sampling_params.get_env_start_args", lambda: args)
    params = SamplingParams()
    params.init(None, thinking_token_budget=budget, guided_reasoning_end=[7])
    if grammar_enabled:
        params.constraint_out_params.guided_grammar.initialize("json", None)
    # Both native requests and forwarded P/D structs must fail before compilation or inference.
    restored = SamplingParams()
    restored.init(None, **params.to_dict())
    for candidate in (params, SamplingParams.from_buffer_copy(params), restored):
        with pytest.raises(ValueError, match="thinking_token_budget is not supported by the gpt-oss"):
            asyncio.run(candidate.verify_cpu_async(None))


@pytest.mark.parametrize("grammar_enabled", [False, True])
@pytest.mark.parametrize("budget", [None, 0])
@pytest.mark.parametrize("marker", [[-1], [257], [287], [2 ** 31 - 1], [7, 257]])
def test_reasoning_end_rejects_ids_outside_output_vocab(
    monkeypatch, tmp_path, compiler, grammar_enabled, budget, marker
):
    from lightllm.server.core.objs import StartArgs

    (tmp_path / "config.json").write_text('{"vocab_size": 257}')
    args = StartArgs(reasoning_parser="qwen3", model_dir=str(tmp_path))
    monkeypatch.setattr("lightllm.server.core.objs.sampling_params.get_env_start_args", lambda: args)
    params = SamplingParams()
    params.init(None, thinking_token_budget=budget, guided_reasoning_end=[0] * len(marker))
    # Validate copied/forwarded structs too, rather than relying on init().
    params = SamplingParams.from_buffer_copy(params)
    params.constraint_out_params.guided_reasoning_end.sequence[: len(marker)] = marker
    with pytest.raises(ValueError, match=r"guided_reasoning_end token ID .* must be in \[0, 257\)"):
        asyncio.run(params.verify_cpu_async(compiler if grammar_enabled else None))


@pytest.mark.parametrize("grammar_enabled", [False, True])
@pytest.mark.parametrize("marker", [[0], [256], [31, 32, 255, 256]])
def test_reasoning_end_accepts_output_vocab_boundaries(monkeypatch, tmp_path, compiler, grammar_enabled, marker):
    from lightllm.server.core.objs import StartArgs

    (tmp_path / "config.json").write_text('{"vocab_size": 257}')
    args = StartArgs(reasoning_parser="qwen3", model_dir=str(tmp_path))
    monkeypatch.setattr("lightllm.server.core.objs.sampling_params.get_env_start_args", lambda: args)
    params = SamplingParams()
    # Added tokens can be outside tokenizer.vocab_size while valid for model output.
    params.init(SimpleNamespace(vocab_size=128), thinking_token_budget=0, guided_reasoning_end=marker)
    assert asyncio.run(params.get_compiled_grammar(compiler if grammar_enabled else None)) is None


@pytest.mark.parametrize(
    "parser,candidates",
    [
        (None, None),
        ("qwen3", 2),
    ],
)
def test_budget_request_rejects_unsupported_configuration(monkeypatch, parser, candidates):
    from lightllm.server.core.objs import sampling_params
    from lightllm.server.core.objs import StartArgs

    args = StartArgs(reasoning_parser=parser, target_vocab_topk_sampling=candidates)
    monkeypatch.setattr(sampling_params, "get_env_start_args", lambda: args)
    params = SamplingParams()
    params.init(None, thinking_token_budget=2, guided_reasoning_end=[7])
    with pytest.raises(ValueError, match="thinking_token_budget"):
        asyncio.run(params.get_compiled_grammar(None))


def test_unbudgeted_request_does_not_inspect_budget_configuration(monkeypatch):
    from unittest.mock import Mock

    from lightllm.server.core.objs import sampling_params

    configuration = Mock(side_effect=AssertionError("Unbudgeted requests must not inspect budget configuration"))
    monkeypatch.setattr(sampling_params, "get_env_start_args", configuration)
    params = SamplingParams()
    params.init(None)
    asyncio.run(params.verify_cpu_async(None))
    configuration.assert_not_called()
