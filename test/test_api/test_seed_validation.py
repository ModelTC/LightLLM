import pytest
from pydantic import ValidationError

from lightllm.server.api_models import ChatCompletionRequest, CompletionRequest, MAX_SEED
from lightllm.server.core.objs.py_sampling_params import SamplingParams as PySamplingParams
from lightllm.server.core.objs.sampling_params import SamplingParams


@pytest.mark.parametrize(
    ("request_type", "request_data"),
    [
        (CompletionRequest, {"model": "test", "prompt": "hello"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hello"}]}),
    ],
)
def test_api_request_seed_range(request_type, request_data):
    for seed in (None, -1, 0, MAX_SEED):
        assert request_type(seed=seed, **request_data).seed == seed

    for seed in (-2, MAX_SEED + 1):
        with pytest.raises(ValidationError):
            request_type(seed=seed, **request_data)


@pytest.mark.parametrize(
    ("seed", "expected_seed"),
    [
        (None, -1),
        (-1, -1),
        (0, 0),
        (MAX_SEED, MAX_SEED),
    ],
)
def test_sampling_params_normalizes_and_accepts_seed(seed, expected_seed):
    sampling_params = SamplingParams()
    sampling_params.init(None, seed=seed)
    assert sampling_params.seed == expected_seed

    py_sampling_params = PySamplingParams(seed=seed)
    py_sampling_params.verify()
    assert py_sampling_params.seed == expected_seed


@pytest.mark.parametrize("seed", [-2, MAX_SEED + 1])
def test_sampling_params_rejects_out_of_range_seed(seed):
    sampling_params = SamplingParams()
    with pytest.raises(ValueError, match="seed must be -1"):
        sampling_params.init(None, seed=seed)

    with pytest.raises(ValueError, match="seed must be -1"):
        PySamplingParams(seed=seed)
