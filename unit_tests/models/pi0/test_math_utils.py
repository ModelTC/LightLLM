import pytest
import torch

from lightllm.models.pi0.model import Pi0ActionExpertModel


@pytest.mark.parametrize("steps", [1, 4, 10])
def test_denoise_schedule_has_exact_requested_step_count(steps):
    model = Pi0ActionExpertModel.__new__(Pi0ActionExpertModel)
    model.device = torch.device("cpu")
    model.data_type = torch.float64
    model.vla_config = {
        "expert_hidden_size": 4,
        "min_period": 0.004,
        "max_period": 4.0,
    }
    model._denoise_constants = model._init_denoise_constants(10)
    embeddings, dt = model._get_denoise_constants(steps)
    cached_embeddings, cached_dt = model._get_denoise_constants(steps)
    assert embeddings.shape == (steps, 4)
    assert cached_embeddings is embeddings
    assert cached_dt is dt
    torch.testing.assert_close(dt, torch.tensor(-1.0 / steps, dtype=torch.float32))


def test_denoise_schedule_rejects_steps_not_precomputed_at_startup():
    model = Pi0ActionExpertModel.__new__(Pi0ActionExpertModel)
    model._denoise_constants = {1: (None, None)}
    with pytest.raises(ValueError, match=r"\[1, 1\]"):
        model._get_denoise_constants(2)
