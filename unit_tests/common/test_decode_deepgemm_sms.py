from types import SimpleNamespace
import pytest

from lightllm.distributed import communication_op as comm


@pytest.mark.parametrize(
    "mode,has_low_latency,expected_sms",
    [("decode", True, 132), ("normal", True, 118), ("prefill", False, 118), ("decode", False, 118)],
)
def test_decode_compute_pool_preserves_communication_budget(monkeypatch, mode, has_low_latency, expected_sms):
    import deep_gemm

    compute_sms, communication_sms = [], []
    monkeypatch.setattr(deep_gemm, "set_num_sms", compute_sms.append)
    monkeypatch.setattr(comm, "get_device_sm_count", lambda: 132)
    monkeypatch.setattr(comm, "get_env_start_args", lambda: SimpleNamespace(run_mode=mode))
    monkeypatch.setattr(comm, "deep_ep", SimpleNamespace(Buffer=SimpleNamespace(set_num_sms=communication_sms.append)))
    manager = comm.DistributeGroupManager()
    manager.ep_low_latency_buffer = object() if has_low_latency else None
    manager._set_num_sms_for_deep_gemm(14, 15)
    assert compute_sms == [expected_sms]
    assert communication_sms == ([14] if has_low_latency else [])
    assert manager.ep_num_sms == 14
