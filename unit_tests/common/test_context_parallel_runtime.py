import pytest
import torch

from lightllm.common import context_parallel_runtime as module


def test_layout_keeps_tp_rank_constant_for_context_groups():
    layout = module.ContextParallelLayout(global_rank=5, dp_size=2, context_size=2, tp_size=2)

    assert (layout.dp_rank, layout.context_rank, layout.tp_rank) == (1, 0, 1)
    assert layout.group_ranks("tp") == (4, 5)
    assert layout.group_ranks("context") == (5, 7)
    assert layout.group_ranks("dp") == (1, 5)


def test_tp_major_layout_matches_contiguous_dcp_context_groups():
    layout = module.ContextParallelLayout(global_rank=3, dp_size=2, context_size=2, tp_size=2, order="tp_major")

    assert (layout.dp_rank, layout.context_rank, layout.tp_rank) == (0, 1, 1)
    assert layout.group_ranks("tp") == (1, 3)
    assert layout.group_ranks("context") == (2, 3)
    assert layout.group_ranks("dp") == (3, 7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_rank": True, "dp_size": 1, "context_size": 1, "tp_size": 1},
        {"global_rank": 0, "dp_size": 1.5, "context_size": 1, "tp_size": 1},
        {"global_rank": 0, "dp_size": 1, "context_size": 0, "tp_size": 1},
        {"global_rank": 4, "dp_size": 1, "context_size": 2, "tp_size": 2},
    ],
)
def test_layout_rejects_invalid_topology(kwargs):
    with pytest.raises(ValueError):
        module.ContextParallelLayout(**kwargs)


def test_group_creation_requires_initialized_world(monkeypatch):
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized torch.distributed"):
        module.create_context_parallel_groups(
            module.ContextParallelLayout(global_rank=0, dp_size=1, context_size=1, tp_size=1)
        )


def test_group_creation_uses_deterministic_specs(monkeypatch):
    layout = module.ContextParallelLayout(global_rank=3, dp_size=2, context_size=2, tp_size=2)
    created = []

    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: layout.world_size)
    monkeypatch.setattr(module.dist, "get_rank", lambda group=None: layout.global_rank)
    monkeypatch.setattr(module.dist, "new_group", lambda ranks, backend=None: created.append((ranks, backend)) or ranks)

    groups = module.create_context_parallel_groups(layout, backend="gloo")

    assert (
        len(created)
        == layout.dp_size * layout.context_size + layout.dp_size * layout.tp_size + layout.context_size * layout.tp_size
    )
    assert created[0] == ([0, 1], "gloo")
    assert groups.group("tp") == [2, 3]
    assert groups.group("context") == [1, 3]
    assert groups.group("dp") == [3, 7]


def test_all_reduce_selects_the_requested_group(monkeypatch):
    layout = module.ContextParallelLayout(global_rank=0, dp_size=1, context_size=2, tp_size=1)
    calls = []
    groups = module.ContextParallelGroups(layout, tp="tp", context="context", dp="dp")
    tensor = torch.ones(1)
    monkeypatch.setattr(module.dist, "all_reduce", lambda value, group: calls.append((value, group)))

    assert groups.all_reduce(tensor, kind="context") is tensor
    assert calls == [(tensor, "context")]
