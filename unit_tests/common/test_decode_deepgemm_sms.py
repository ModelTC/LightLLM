from types import SimpleNamespace

import pytest

from lightllm.distributed import communication_op as comm


@pytest.mark.parametrize(
    "mode,low_latency,expected",
    [
        ("decode", True, 132),
        ("prefill", True, 118),
        ("normal", True, 118),
        ("decode", False, 118),
    ],
)
def test_decode_compute_pool_preserves_communication_budget(
    monkeypatch, mode, low_latency, expected
):
    import deep_gemm

    compute, communication = [], []
    # Both old and new DeepGEMM import paths must observe the same test hook.
    monkeypatch.setattr(deep_gemm, "set_num_sms", compute.append)
    try:
        import deep_gemm.jit_kernels.utils as old_utils
    except ImportError:
        pass
    else:
        monkeypatch.setattr(old_utils, "set_num_sms", compute.append)
    monkeypatch.setattr(comm, "get_device_sm_count", lambda: 132)
    monkeypatch.setattr(
        comm, "get_env_start_args", lambda: SimpleNamespace(run_mode=mode)
    )
    monkeypatch.setattr(
        comm,
        "deep_ep",
        SimpleNamespace(Buffer=SimpleNamespace(set_num_sms=communication.append)),
    )
    manager = comm.DistributeGroupManager()
    manager.ep_low_latency_buffer = object() if low_latency else None
    manager._set_num_sms_for_deep_gemm(14)
    assert compute == [expected]
    assert communication == ([14] if low_latency else [])
    assert manager.ep_num_sms == 14
