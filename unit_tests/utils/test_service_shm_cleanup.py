import json

from lightllm.utils import service_shm_cleanup


def test_cleanup_service_shm_only_removes_matching_service(monkeypatch, tmp_path):
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    matching_names = ["service_0_req_pool", "service_0_token_load"]
    for name in [*matching_names, "service_1_req_pool", "other_service_0_value"]:
        (shm_dir / name).touch()

    removed_system_v_keys = []
    monkeypatch.setattr(service_shm_cleanup, "SHM_DIR", shm_dir)
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_system_v_shm",
        staticmethod(lambda keys: removed_system_v_keys.extend(keys) or len(keys)),
    )

    service_shm_cleanup.ServiceShmCleanup.cleanup_posix_shm("service_0")
    service_shm_cleanup.ServiceShmCleanup.cleanup_system_v_shm([101, 102])

    assert all(not (shm_dir / name).exists() for name in matching_names)
    assert (shm_dir / "service_1_req_pool").exists()
    assert (shm_dir / "other_service_0_value").exists()
    assert removed_system_v_keys == [101, 102]


def test_system_v_shm_keys_follow_feature_switches(monkeypatch):
    start_args = {
        "run_mode": "prefill",
        "enable_cpu_cache": True,
        "enable_multimodal": False,
        "cpu_kv_cache_shm_id": 21,
        "multi_modal_cache_shm_id": 22,
    }
    removed_system_v_keys = []
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(start_args))
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_posix_shm",
        staticmethod(lambda service_name: 0),
    )
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_system_v_shm",
        staticmethod(lambda keys: removed_system_v_keys.extend(keys) or len(keys)),
    )

    service_shm_cleanup.ServiceShmCleanup("current_service_0").cleanup_service_resources()

    assert removed_system_v_keys == [21]


def test_non_inference_mode_skips_system_v_shm_cleanup(monkeypatch):
    start_args = {
        "run_mode": "visual_only",
        "enable_cpu_cache": True,
        "enable_multimodal": True,
        "cpu_kv_cache_shm_id": 21,
        "multi_modal_cache_shm_id": 22,
    }
    system_v_cleanup_calls = []
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(start_args))
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_posix_shm",
        staticmethod(lambda service_name: 0),
    )
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_system_v_shm",
        staticmethod(lambda keys: system_v_cleanup_calls.append(keys) or 0),
    )

    service_shm_cleanup.ServiceShmCleanup("current_service_0").cleanup_service_resources()

    assert system_v_cleanup_calls == []


def test_register_launcher_cleanup_uses_current_start_args(monkeypatch):
    cleanup_calls = []
    atexit_callbacks = []
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_posix_shm",
        staticmethod(lambda service_name: cleanup_calls.append(("posix", service_name)) or 0),
    )
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_system_v_shm",
        staticmethod(lambda keys: cleanup_calls.append(("system_v", keys)) or 0),
    )
    monkeypatch.setattr(service_shm_cleanup.atexit, "register", atexit_callbacks.append)
    start_args = {
        "run_mode": "normal",
        "model_dir": "/models/test",
        "tp": 2,
        "enable_cpu_cache": True,
        "enable_multimodal": True,
        "cpu_kv_cache_shm_id": 21,
        "multi_modal_cache_shm_id": 22,
    }
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(start_args),
    )

    cleanup = service_shm_cleanup.register_launcher_shm_cleanup("current_service_0")

    assert cleanup_calls == []
    assert atexit_callbacks == [cleanup]

    cleanup()
    cleanup()
    assert cleanup_calls == [
        ("posix", "current_service_0"),
        ("system_v", [21, 22]),
        ("posix", "current_service_0"),
        ("system_v", [21, 22]),
    ]
