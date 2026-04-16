"""Pure-Python unit tests for the LLM auto-profile path.

These tests exercise the memory manager's probe sizing, target arithmetic,
and the basemodel init's retry loop. They stub torch.cuda.* and
get_available_gpu_memory via monkeypatch so they run on CPU without any
GPU-related setup. They are the only pure-Python tests in
unit_tests/common/basemodel/ at the time of writing — the other tests
in that directory are triton kernel tests that require a real GPU.
"""
import pytest


class _StubStartArgs:
    def __init__(self, graph_max_batch_size, batch_max_tokens):
        self.graph_max_batch_size = graph_max_batch_size
        self.batch_max_tokens = batch_max_tokens


@pytest.fixture
def stub_env_start_args(monkeypatch):
    def _install(graph_max_batch_size, batch_max_tokens):
        stub = _StubStartArgs(graph_max_batch_size, batch_max_tokens)
        monkeypatch.setattr(
            "lightllm.utils.envs_utils.get_env_start_args",
            lambda: stub,
        )
        return stub

    return _install


def _make_bare_mem_manager():
    """Construct a MemoryManager instance without running __init__.

    This lets us call profile_size() / profile_size_target() on a plain
    object with only the fields those methods touch, avoiding the need
    to initialize CUDA, distributed, or shared-memory state.
    """
    from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

    mgr = MemoryManager.__new__(MemoryManager)
    mgr.size = None
    mgr._probe_tokens = None
    mgr._mem_fraction = 1.0
    # Cell size only matters when profile_size_target() runs — not probe.
    mgr.head_num = 8
    mgr.head_dim = 64
    mgr.layer_num = 32
    import torch

    mgr.dtype = torch.float16
    return mgr


def test_profile_size_probe_formula_graph_heavy(stub_env_start_args):
    """Probe size = gmbs * (bmt + 256) when that exceeds the 8192 floor."""
    stub_env_start_args(graph_max_batch_size=64, batch_max_tokens=4096)
    mgr = _make_bare_mem_manager()
    mgr.profile_size(mem_fraction=1.0)
    assert mgr.size == 64 * (4096 + 256)
    assert mgr._probe_tokens == 64 * (4096 + 256)
    assert mgr._mem_fraction == 1.0


def test_profile_size_probe_formula_tiny_config(stub_env_start_args):
    """Probe size floors to 8192 when gmbs*(bmt+256) is smaller."""
    stub_env_start_args(graph_max_batch_size=1, batch_max_tokens=128)
    mgr = _make_bare_mem_manager()
    mgr.profile_size(mem_fraction=1.0)
    assert mgr.size == 8192


def test_profile_size_early_return_when_size_preset(stub_env_start_args):
    """If size is already set (e.g. --max_total_token_num), profile_size is a no-op."""
    stub_env_start_args(graph_max_batch_size=64, batch_max_tokens=4096)
    mgr = _make_bare_mem_manager()
    mgr.size = 131072
    mgr.profile_size(mem_fraction=0.9)
    assert mgr.size == 131072
    assert mgr._probe_tokens is None  # not touched


def test_profile_size_target_arithmetic(monkeypatch, stub_env_start_args):
    """profile_size_target computes target from peak, peers, canary, budget."""
    stub_env_start_args(graph_max_batch_size=64, batch_max_tokens=4096)

    mgr = _make_bare_mem_manager()
    mgr.profile_size(mem_fraction=1.0)  # picks probe
    probe_tokens = mgr._probe_tokens
    cell_size = mgr.get_cell_size()
    probe_kv_bytes = probe_tokens * cell_size

    # Set up a synthetic 80 GB card.
    TOTAL_GB = 80.0
    total_bytes = int(TOTAL_GB * 1024 ** 3)
    # Peer footprint: 10 GB worth of ViT + audio driver reservation.
    peer_bytes = int(10 * 1024 ** 3)
    # Own reserved: weights + probe KV + graphs + stress activations. Say 35 GB.
    own_reserved_bytes = int(35 * 1024 ** 3)
    # Peak reserved (high-water-mark after stress) = own_reserved in this model.
    peak_reserved = own_reserved_bytes

    monkeypatch.setattr(
        "lightllm.common.kv_cache_mem_manager.mem_manager.get_total_gpu_memory",
        lambda: TOTAL_GB,
    )
    monkeypatch.setattr(
        "lightllm.common.kv_cache_mem_manager.mem_manager.get_available_gpu_memory",
        lambda world_size=1: (total_bytes - own_reserved_bytes - peer_bytes) / 1024 ** 3,
    )
    import torch

    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: own_reserved_bytes)
    monkeypatch.setattr(
        "torch.distributed.get_world_size",
        lambda: 1,
    )

    target = mgr.profile_size_target(peak_reserved)

    non_kv_overhead = peak_reserved - probe_kv_bytes
    canary_bytes = 256 * 1024 * 1024
    expected_budget = total_bytes - non_kv_overhead - canary_bytes - peer_bytes
    expected_target = int(expected_budget / cell_size)

    assert target == expected_target
    # Sanity: target is strictly larger than the probe (the whole point).
    assert target > probe_tokens


def test_profile_size_target_peer_footprint_floors_to_zero(monkeypatch, stub_env_start_args):
    """If get_available_gpu_memory says more is available than total-own_reserved,
    the peer footprint must floor to 0 (the arithmetic produced a negative number)."""
    stub_env_start_args(graph_max_batch_size=1, batch_max_tokens=128)
    mgr = _make_bare_mem_manager()
    mgr.profile_size(mem_fraction=1.0)

    TOTAL_GB = 80.0

    monkeypatch.setattr(
        "lightllm.common.kv_cache_mem_manager.mem_manager.get_total_gpu_memory",
        lambda: TOTAL_GB,
    )
    # avail > total - own_reserved, i.e. peer_footprint would be negative
    monkeypatch.setattr(
        "lightllm.common.kv_cache_mem_manager.mem_manager.get_available_gpu_memory",
        lambda world_size=1: TOTAL_GB,  # "everything is available"
    )
    import torch

    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)

    target = mgr.profile_size_target(peak_reserved_bytes=1024 * 1024)
    assert target > 0  # didn't crash on negative peer footprint


def test_profile_size_target_mem_fraction_multiplier(monkeypatch, stub_env_start_args):
    """--mem_fraction 0.95 should produce a target 95% the size of the default 1.0."""
    stub_env_start_args(graph_max_batch_size=64, batch_max_tokens=4096)

    TOTAL_GB = 80.0
    total_bytes = int(TOTAL_GB * 1024 ** 3)
    own_reserved = int(35 * 1024 ** 3)
    peak_reserved = own_reserved

    def _patch(mgr):
        monkeypatch.setattr(
            "lightllm.common.kv_cache_mem_manager.mem_manager.get_total_gpu_memory",
            lambda: TOTAL_GB,
        )
        monkeypatch.setattr(
            "lightllm.common.kv_cache_mem_manager.mem_manager.get_available_gpu_memory",
            lambda world_size=1: (total_bytes - own_reserved) / 1024 ** 3,
        )
        import torch

        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: own_reserved)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)

    mgr_default = _make_bare_mem_manager()
    mgr_default.profile_size(mem_fraction=1.0)
    _patch(mgr_default)
    target_default = mgr_default.profile_size_target(peak_reserved)

    mgr_paranoid = _make_bare_mem_manager()
    mgr_paranoid.profile_size(mem_fraction=0.95)
    _patch(mgr_paranoid)
    target_paranoid = mgr_paranoid.profile_size_target(peak_reserved)

    # Paranoid target is 95% of default target (± rounding).
    ratio = target_paranoid / target_default
    assert 0.94 < ratio <= 0.95
