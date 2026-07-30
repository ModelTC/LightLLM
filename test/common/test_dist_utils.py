from types import SimpleNamespace

from lightllm.utils import dist_utils, envs_utils


def test_single_node_group_reuses_world_group(monkeypatch):
    world_group = dist_utils.dist.group.WORLD
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: SimpleNamespace(nnodes=1, tp=4))
    monkeypatch.setattr(
        dist_utils.dist,
        "new_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("new_group must not be called")),
    )

    assert dist_utils.create_new_group_for_current_node("nccl") is world_group
