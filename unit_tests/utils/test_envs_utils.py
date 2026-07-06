from lightllm.utils import envs_utils


def test_lightllm_hypercorn_backlog_defaults_above_hypercorn_default(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_HYPERCORN_BACKLOG", raising=False)

    assert envs_utils.get_lightllm_hypercorn_backlog() == 1024


def test_lightllm_hypercorn_backlog_can_be_overridden(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_HYPERCORN_BACKLOG", "2048")

    assert envs_utils.get_lightllm_hypercorn_backlog() == 2048
