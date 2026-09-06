from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import inspect
import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.multi_level_kv_cache import (
    AdaptiveCachePlacementController,
    CacheCapacityConfig,
    CachePlacementController,
    CacheTier,
    GpuOnlyCachePlacementController,
    LegacyCachePlacementController,
    create_cache_placement_controller,
)


def make_controller(
    gpu_tokens: int,
    cpu_tokens: int,
    disk_tokens: int,
    history_size: int = AdaptiveCachePlacementController.MAX_HISTORY_SIZE,
    initial_history_size: int = None,
    enable_cpu_cache: bool = None,
    enable_disk_cache: bool = None,
) -> CachePlacementController:
    if enable_cpu_cache is None:
        enable_cpu_cache = cpu_tokens > 0
    if enable_disk_cache is None:
        enable_disk_cache = disk_tokens > 0
    if initial_history_size is None:
        initial_history_size = min(AdaptiveCachePlacementController.INITIAL_HISTORY_SIZE, history_size)
    return AdaptiveCachePlacementController(
        capacity=CacheCapacityConfig(
            gpu_tokens=gpu_tokens,
            cpu_tokens=cpu_tokens,
            disk_tokens=disk_tokens,
        ),
        args=SimpleNamespace(
            enable_cpu_cache=enable_cpu_cache,
            enable_disk_cache=enable_disk_cache,
        ),
        max_history_size=history_size,
        initial_history_size=initial_history_size,
    )


def make_req(request_id: int, input_len: int, group_req_id: int = None, cache_tiers=(CacheTier.GPU,)):
    if group_req_id is None:
        group_req_id = request_id
    return SimpleNamespace(
        cache_tiers=cache_tiers,
        shm_req=SimpleNamespace(
            request_id=request_id,
            group_req_id=group_req_id,
            input_len=input_len,
        ),
    )


def test_controller_exposes_one_public_operation():
    controller = make_controller(1, 1, 1)

    public_methods = [
        name for name, method in inspect.getmembers(controller, predicate=inspect.ismethod) if not name.startswith("_")
    ]

    assert public_methods == ["set_req_cache_way"]


def test_controller_is_an_interface():
    assert inspect.isabstract(CachePlacementController)


@pytest.mark.parametrize(
    "cache_tiers",
    [
        (CacheTier.DISK,),
        (CacheTier.GPU, CacheTier.DISK),
    ],
)
def test_controller_rejects_illegal_existing_cache_tiers(cache_tiers):
    req = make_req(0, 100, cache_tiers=cache_tiers)

    with pytest.raises(AssertionError):
        make_controller(1, 1, 1).set_req_cache_way([req])


def test_gpu_only_controller_always_selects_gpu():
    reqs = [make_req(0, 100), make_req(1, 1000)]

    GpuOnlyCachePlacementController().set_req_cache_way(reqs)

    assert tuple(req.cache_tiers for req in reqs) == ((CacheTier.GPU,), (CacheTier.GPU,))


def test_legacy_controller_selects_all_enabled_cache_levels():
    req = make_req(0, 100)

    LegacyCachePlacementController(enable_cpu_cache=True, enable_disk_cache=True).set_req_cache_way([req])

    assert req.cache_tiers == (CacheTier.GPU, CacheTier.CPU, CacheTier.DISK)


def test_legacy_controller_does_not_select_disabled_disk_cache():
    req = make_req(0, 100)

    LegacyCachePlacementController(enable_cpu_cache=True, enable_disk_cache=False).set_req_cache_way([req])

    assert req.cache_tiers == (CacheTier.GPU, CacheTier.CPU)


def test_legacy_controller_keeps_diverse_slave_request_on_gpu():
    req = make_req(request_id=2, group_req_id=1, input_len=100)

    LegacyCachePlacementController(enable_cpu_cache=True, enable_disk_cache=True).set_req_cache_way([req])

    assert req.cache_tiers == (CacheTier.GPU,)


def test_cache_placement_strategy_cli_defaults_to_adaptive_and_accepts_legacy():
    parser = make_argument_parser()

    assert parser.parse_args([]).cache_placement_strategy == "adaptive"
    assert parser.parse_args(["--cache_placement_strategy", "legacy"]).cache_placement_strategy == "legacy"


def test_factory_creates_gpu_only_controller_when_cpu_cache_is_disabled():
    args = SimpleNamespace(enable_cpu_cache=False)

    controller = create_cache_placement_controller(args=args, radix_cache=None)

    assert isinstance(controller, GpuOnlyCachePlacementController)


def test_factory_creates_legacy_controller_for_legacy_strategy():
    args = SimpleNamespace(
        enable_cpu_cache=True,
        enable_disk_cache=True,
        cache_placement_strategy="legacy",
    )

    controller = create_cache_placement_controller(args=args, radix_cache=None)

    assert isinstance(controller, LegacyCachePlacementController)


def test_factory_calculates_adaptive_cache_capacities_across_local_dp_replicas_and_applies_gpu_ratio(monkeypatch):
    from lightllm.utils import envs_utils, kv_cache_utils

    cpu_cache_meta = SimpleNamespace(page_num=10, calcu_one_page_size=lambda: 64)
    monkeypatch.setattr(kv_cache_utils, "calcu_cpu_cache_meta", lambda: cpu_cache_meta)
    monkeypatch.setattr(envs_utils, "get_cache_placement_gpu_capacity_ratio", lambda: 0.5)
    args = SimpleNamespace(
        enable_cpu_cache=True,
        enable_disk_cache=True,
        cache_placement_strategy="adaptive",
        cpu_cache_token_page_size=16,
        disk_cache_storage_size=1,
        dp=4,
        nnodes=2,
    )

    controller = create_cache_placement_controller(
        args=args,
        radix_cache=SimpleNamespace(total_token_num=80),
    )

    assert isinstance(controller, AdaptiveCachePlacementController)
    assert controller._capacity == CacheCapacityConfig(
        gpu_tokens=80,
        cpu_tokens=160,
        disk_tokens=int(1024 ** 3) // 64 * 16,
    )


@pytest.mark.parametrize(
    ("configured_ratio", "expected_ratio"),
    [
        (None, 0.8),
        ("0.5", 0.5),
        ("1", 1.0),
    ],
)
def test_cache_placement_gpu_capacity_ratio_from_environment(monkeypatch, configured_ratio, expected_ratio):
    from lightllm.utils.envs_utils import get_cache_placement_gpu_capacity_ratio

    if configured_ratio is None:
        monkeypatch.delenv("LIGHTLLM_CACHE_PLACEMENT_GPU_CAPACITY_RATIO", raising=False)
    else:
        monkeypatch.setenv("LIGHTLLM_CACHE_PLACEMENT_GPU_CAPACITY_RATIO", configured_ratio)

    assert get_cache_placement_gpu_capacity_ratio() == expected_ratio


@pytest.mark.parametrize("configured_ratio", ["0", "-0.1", "1.1", "nan"])
def test_cache_placement_gpu_capacity_ratio_rejects_out_of_range_values(monkeypatch, configured_ratio):
    from lightllm.utils.envs_utils import get_cache_placement_gpu_capacity_ratio

    monkeypatch.setenv("LIGHTLLM_CACHE_PLACEMENT_GPU_CAPACITY_RATIO", configured_ratio)

    with pytest.raises(AssertionError):
        get_cache_placement_gpu_capacity_ratio()


def test_set_req_cache_way_uses_recent_input_lengths_and_capacity_ratio():
    controller = make_controller(gpu_tokens=1, cpu_tokens=1, disk_tokens=2, history_size=4)
    warmup_reqs = [
        make_req(0, 400),
        make_req(1, 100),
        make_req(2, 300),
        make_req(3, 200),
    ]
    controller.set_req_cache_way(warmup_reqs)
    reqs = [
        make_req(4, 400),
        make_req(5, 100),
        make_req(6, 300),
        make_req(7, 200),
    ]

    result = controller.set_req_cache_way(reqs)

    assert result is None
    assert tuple(req.cache_tiers for req in warmup_reqs) == ((CacheTier.GPU, CacheTier.CPU, CacheTier.DISK),) * 4
    assert tuple(req.cache_tiers for req in reqs) == (
        (CacheTier.CPU, CacheTier.DISK),
        (CacheTier.GPU,),
        (CacheTier.GPU,),
        (CacheTier.GPU,),
    )


def test_adaptive_controller_uses_enable_flags_instead_of_capacity_to_select_disk_path():
    controller = make_controller(
        gpu_tokens=1,
        cpu_tokens=1,
        disk_tokens=0,
        enable_cpu_cache=True,
        enable_disk_cache=True,
    )
    warmup_req = make_req(0, 100)

    controller.set_req_cache_way([warmup_req])

    assert warmup_req.cache_tiers == (CacheTier.GPU, CacheTier.CPU, CacheTier.DISK)


def test_length_boundary_uses_max_cpu_and_disk_capacity():
    controller = make_controller(gpu_tokens=1, cpu_tokens=1, disk_tokens=3, history_size=4)
    controller.set_req_cache_way([make_req(0, 100), make_req(1, 200), make_req(2, 300), make_req(3, 400)])
    assert controller._gpu_max_input_len == 200
    req = make_req(4, 150)

    controller.set_req_cache_way([req])

    assert req.cache_tiers == (CacheTier.GPU,)


def test_zero_gpu_capacity_does_not_place_requests_in_gpu_after_warmup():
    controller = make_controller(gpu_tokens=0, cpu_tokens=1, disk_tokens=0, history_size=2)
    controller.set_req_cache_way([make_req(0, 100), make_req(1, 200)])
    req = make_req(2, 100)

    controller.set_req_cache_way([req])

    assert controller._gpu_max_input_len == 0
    assert req.cache_tiers == (CacheTier.CPU,)


def test_set_req_cache_way_does_not_use_disabled_tiers():
    controller = make_controller(gpu_tokens=1, cpu_tokens=3, disk_tokens=0, history_size=4)
    warmup_reqs = [make_req(index, length) for index, length in enumerate((100, 200, 300, 400))]
    controller.set_req_cache_way(warmup_reqs)
    reqs = [make_req(index + 4, length) for index, length in enumerate((100, 200, 300, 400))]

    controller.set_req_cache_way(reqs)

    assert tuple(req.cache_tiers for req in warmup_reqs) == ((CacheTier.GPU, CacheTier.CPU),) * 4
    assert tuple(req.cache_tiers for req in reqs) == (
        (CacheTier.GPU,),
        (CacheTier.GPU,),
        (CacheTier.CPU,),
        (CacheTier.CPU,),
    )


def test_length_boundary_keeps_history_after_initial_update():
    controller = make_controller(1, 1, 1, history_size=3)

    controller.set_req_cache_way([make_req(0, 100), make_req(1, 200)])
    assert controller._gpu_max_input_len is None
    assert list(controller._recent_input_lengths) == [100, 200]

    controller.set_req_cache_way([make_req(2, 300)])
    assert controller._gpu_max_input_len == 300
    assert list(controller._recent_input_lengths) == [100, 200, 300]

    latest_req = make_req(3, 1000)
    controller.set_req_cache_way([latest_req])

    assert controller._gpu_max_input_len == 300
    assert list(controller._recent_input_lengths) == [200, 300, 1000]
    assert latest_req.cache_tiers == (CacheTier.CPU, CacheTier.DISK)


def test_length_boundary_updates_every_36_steps_after_initial_update():
    controller = make_controller(1, 1, 1, history_size=4, initial_history_size=2)

    controller.set_req_cache_way([make_req(0, 100), make_req(1, 200)])
    assert controller._gpu_max_input_len == 200
    assert controller._steps_since_last_update == 0
    assert list(controller._recent_input_lengths) == [100, 200]

    for step in range(35):
        controller.set_req_cache_way([make_req(step + 2, 1000)])
    assert controller._gpu_max_input_len == 200
    assert controller._steps_since_last_update == 35
    assert list(controller._recent_input_lengths) == [1000, 1000, 1000, 1000]

    controller.set_req_cache_way([make_req(37, 1000)])
    assert controller._gpu_max_input_len == 1000
    assert controller._steps_since_last_update == 0
    assert list(controller._recent_input_lengths) == [1000, 1000, 1000, 1000]


def test_history_window_keeps_latest_requests_when_single_extend_exceeds_limit():
    controller = make_controller(1, 1, 1, history_size=3)
    reqs = [make_req(index, length) for index, length in enumerate((100, 200, 300, 400, 500))]

    controller.set_req_cache_way(reqs)

    assert controller._gpu_max_input_len == 400
    assert list(controller._recent_input_lengths) == [300, 400, 500]


def test_repeated_assignment_is_rejected():
    controller = make_controller(1, 1, 1)
    req = make_req(0, 100)
    controller.set_req_cache_way([req])

    req.shm_req.input_len = 10000
    with pytest.raises(AssertionError):
        controller.set_req_cache_way([req])

    assert list(controller._recent_input_lengths) == [100]


def test_concurrent_reassignment_is_rejected():
    controller = make_controller(1, 1, 1)
    req = make_req(0, 100)

    with pytest.raises(AssertionError):
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: controller.set_req_cache_way([req]), range(2)))

    assert list(controller._recent_input_lengths) == [100]


def test_diverse_slave_request_stays_on_gpu_and_is_not_recorded():
    controller = make_controller(0, 1, 1)
    slave_req = make_req(request_id=2, group_req_id=1, input_len=1000)

    controller.set_req_cache_way([slave_req])

    assert slave_req.cache_tiers == (CacheTier.GPU,)
    assert list(controller._recent_input_lengths) == []
