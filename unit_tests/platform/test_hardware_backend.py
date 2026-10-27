"""Platform registry tests that need no device.

The cuda backend must forward to exactly the torch.cuda calls the code used before the platform layer,
these tests record the torch.cuda calls instead of running them so they also run on non-NVIDIA hosts.
"""
import json
import types

import pytest
import torch

import lightllm.platform as platform
from lightllm.platform.base.registry import get_platform_spec


def _reset_platform():
    platform._backend = None
    for fn in (platform.get_platform_name, platform.get_device_type, platform.get_device_module):
        fn.cache_clear()


@pytest.fixture
def use_platform(monkeypatch):
    def _use(name):
        monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps({"hardware_platform": name}))
        from lightllm.utils.envs_utils import get_env_start_args

        get_env_start_args.cache_clear()
        _reset_platform()

    yield _use
    from lightllm.utils.envs_utils import get_env_start_args

    get_env_start_args.cache_clear()
    _reset_platform()


@pytest.mark.parametrize(
    "name, backend_cls, device_type, dist_backend",
    [
        ("cuda", "CudaBackend", "cuda", "nccl"),
        ("musa", "MusaBackend", "cuda", "nccl"),
        ("xpu", "XpuBackend", "xpu", "xccl"),
    ],
)
def test_platform_resolution(use_platform, name, backend_cls, device_type, dist_backend):
    use_platform(name)
    assert platform.get_platform_name() == name
    assert type(platform._get_backend()).__name__ == backend_cls
    assert platform.get_device_type() == device_type
    assert platform.get_dist_backend() == dist_backend
    assert platform.get_device_module() is getattr(torch, device_type)


def test_cuda_like_runtime_has_no_backend_restriction(use_platform):
    use_platform("cuda")
    assert platform._get_backend().runtime.supported_att_backends is None
    use_platform("xpu")
    assert platform._get_backend().runtime.supported_att_backends == ("triton", "sdpa")


def test_ops_are_built_lazily(use_platform):
    use_platform("cuda")
    backend = platform._get_backend()
    assert not isinstance(backend._ops, type(None)) and callable(backend._ops)
    ops = backend.ops
    from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward

    assert ops.norm.rms_norm is rmsnorm_forward
    assert backend.ops is ops


@pytest.mark.parametrize(
    "versions, expected",
    [
        ({"cuda": "12.8", "hip": None, "xpu": None}, "cuda"),  # NVIDIA build
        ({"cuda": None, "hip": "7.0", "xpu": None}, "cuda"),  # ROCm build exposes the torch.cuda api
        ({"cuda": None, "hip": None, "xpu": None, "musa": "4.0"}, "cuda"),  # MUSA runs the cuda path via torchada
        ({"cuda": None, "hip": None, "xpu": "20260000"}, "xpu"),  # Intel XPU build
        ({"cuda": None, "hip": None, "xpu": None}, "cuda"),  # cpu build keeps the old default
    ],
)
def test_detect_platform_from_torch_build(monkeypatch, versions, expected):
    for name, value in versions.items():
        monkeypatch.setattr(torch.version, name, value, raising=False)
    assert platform.detect_platform_name() == expected
    assert platform.resolve_platform_name("auto") == expected
    assert platform.resolve_platform_name(None) == expected
    # an explicit platform always wins
    assert platform.resolve_platform_name("cuda") == "cuda"
    assert platform.resolve_platform_name("musa") == "musa"


def test_auto_in_start_args_resolves_to_detected(use_platform):
    use_platform("auto")
    assert platform.get_platform_name() == platform.detect_platform_name()


def test_start_args_default_is_auto():
    from lightllm.server.core.objs.start_args_type import StartArgs

    assert StartArgs().hardware_platform == "auto"


def test_platform_detected_without_start_args(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_START_ARGS", raising=False)
    _reset_platform()
    try:
        assert platform.get_platform_name() == platform.detect_platform_name()
    finally:
        _reset_platform()


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, name, ret=None):
        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return ret

        return fn


def test_cuda_graph_backend_forwards_to_torch_cuda(use_platform, monkeypatch):
    use_platform("cuda")
    rec = _Recorder()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", rec("CUDAGraph", ret="graph"))
    monkeypatch.setattr(torch.cuda, "graph", rec("graph", ret="ctx"))
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", rec("graph_pool_handle", ret="pool"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", rec("is_current_stream_capturing", ret=False))

    graph = platform._get_backend().graph
    assert graph.create_graph() == "graph"
    assert graph.graph("graph", pool="pool") == "ctx"
    assert graph.graph("graph") == "ctx"
    assert graph.graph("graph", stream="s") == "ctx"
    assert graph.graph_pool_handle() == "pool"
    assert graph.is_capturing() is False
    # torch.cuda.graph(g, pool=p) == torch.cuda.graph(g, pool=p, stream=None): the defaults of torch.cuda.graph.
    assert rec.calls == [
        ("CUDAGraph", (), {}),
        ("graph", ("graph",), {"pool": "pool", "stream": None}),
        ("graph", ("graph",), {"pool": None, "stream": None}),
        ("graph", ("graph",), {"pool": None, "stream": "s"}),
        ("graph_pool_handle", (), {}),
        ("is_current_stream_capturing", (), {}),
    ]


def test_cuda_runtime_mem_get_info_forwards_to_torch_cuda(use_platform, monkeypatch):
    use_platform("cuda")
    rec = _Recorder()
    monkeypatch.setattr(torch.cuda, "mem_get_info", rec("mem_get_info", ret=(1, 2)))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    runtime = platform._get_backend().runtime
    assert runtime.mem_get_info(3) == (1, 2)
    assert rec.calls == [("mem_get_info", (torch.device("cuda", 3),), {})]


def test_quant_platform(use_platform, monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    # Import order as in the server: lightllm.common.quantization alone has a pre-existing import cycle.
    import lightllm.common.basemodel  # noqa: F401
    from lightllm.common.quantization import _get_quant_platform
    from lightllm.common.quantization.registry import QUANTMETHODS
    from lightllm.common.quantization.no_quant import NoQuantization

    for name, expected in [("cuda", "cuda"), ("musa", "cuda"), ("xpu", "xpu")]:
        use_platform(name)
        assert _get_quant_platform() == expected
        assert isinstance(QUANTMETHODS.get("none", platform=_get_quant_platform()), NoQuantization)

    use_platform("xpu")
    with pytest.raises(ValueError, match="platform 'xpu' not supported"):
        QUANTMETHODS.get("awq", platform=_get_quant_platform())


def test_backend_availability_is_checked_at_device_binding(use_platform, monkeypatch):
    use_platform("cuda")
    runtime = platform.get_hardware_backend().runtime
    monkeypatch.setattr(runtime, "is_available", lambda: False)
    # Plain accessors work without a device (tests mocking torch.cuda), the check fails explicitly.
    assert platform.get_hardware_backend().graph is not None
    with pytest.raises(RuntimeError, match="hardware_platform matches"):
        platform.check_hardware_backend_available()
    monkeypatch.setattr(runtime, "is_available", lambda: True)
    platform.check_hardware_backend_available()


def test_registered_platforms():
    for name in ["cuda", "musa", "ascend", "xpu"]:
        assert get_platform_spec(name).name == name


def _xpu_args(**overrides):
    args = dict(
        enable_cpu_cache=False,
        enable_disk_cache=False,
        enable_dp_prompt_cache_fetch=False,
        enable_ep_moe=False,
        enable_torch_memory_saver=False,
        enable_prefill_cudagraph=False,
        enable_mps=False,
        run_mode="normal",
        mtp_mode=None,
        sampling_backend="triton",
        enable_profiling=None,
        disable_cudagraph=False,
    )
    args.update(overrides)
    return types.SimpleNamespace(**args)


def test_xpu_start_args_default_to_eager(monkeypatch):
    from lightllm.platform.backends.xpu.start_args import check_and_set_xpu_args

    monkeypatch.delenv("LIGHTLLM_XPU_ENABLE_GRAPH", raising=False)
    args = _xpu_args()
    check_and_set_xpu_args(args)
    assert args.disable_cudagraph is True

    monkeypatch.setenv("LIGHTLLM_XPU_ENABLE_GRAPH", "1")
    args = _xpu_args()
    check_and_set_xpu_args(args)
    assert args.disable_cudagraph is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_cpu_cache": True},
        {"enable_dp_prompt_cache_fetch": True},
        {"enable_ep_moe": True},
        {"enable_torch_memory_saver": True},
        {"enable_prefill_cudagraph": True},
        {"enable_mps": True},
        {"run_mode": "prefill"},
        {"run_mode": "decode"},
        {"mtp_mode": "vanilla_with_att"},
        {"sampling_backend": "flashinfer"},
        {"enable_profiling": "nvtx"},
    ],
)
def test_xpu_start_args_reject_cuda_only_features(overrides):
    from lightllm.platform.backends.xpu.start_args import check_and_set_xpu_args

    with pytest.raises(ValueError, match="hardware_platform xpu"):
        check_and_set_xpu_args(_xpu_args(**overrides))


def test_attention_backend_filter(use_platform, monkeypatch):
    from lightllm.common.basemodel.attention import create_utils
    from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend
    from lightllm.common.basemodel.attention.fa3.fp import Fa3AttBackend

    monkeypatch.setattr(create_utils, "get_env_start_args", lambda: types.SimpleNamespace(enable_ep_moe=False))

    use_platform("xpu")
    monkeypatch.setattr(platform._get_backend().runtime, "is_available", lambda: True)
    # xpu never runs the cuda subprocess validation.
    monkeypatch.setattr(create_utils, "validate", lambda name: pytest.fail("validate called on xpu"))
    assert create_utils._auto_select_backend("None", create_utils.data_type_to_backend) is TritonAttBackend
    assert create_utils._get_explicit_backend(create_utils.data_type_to_backend, "None", "triton") is TritonAttBackend
    with pytest.raises(ValueError, match="not supported on platform xpu"):
        create_utils._get_explicit_backend(create_utils.data_type_to_backend, "None", "fa3")
    with pytest.raises(ValueError, match="No attention backend"):
        create_utils._auto_select_backend("fp8kv_sph", create_utils.data_type_to_backend, priority_list=["fa3"])

    use_platform("cuda")
    monkeypatch.setattr(platform._get_backend().runtime, "is_available", lambda: True)
    # cuda keeps the validated priority order.
    monkeypatch.setattr(create_utils, "validate", lambda name: name == "fa3")
    assert create_utils._auto_select_backend("None", create_utils.data_type_to_backend) is Fa3AttBackend
    assert create_utils._get_explicit_backend(create_utils.data_type_to_backend, "None", "fa3") is Fa3AttBackend
