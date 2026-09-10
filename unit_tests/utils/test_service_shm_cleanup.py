import json
import os

from lightllm.utils import service_shm_cleanup


def test_cleanup_service_shm_only_removes_matching_service(monkeypatch, tmp_path):
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    matching_names = ["service_0_req_pool", "service_0_token_load"]
    for name in [*matching_names, "service_1_req_pool", "other_service_0_value"]:
        (shm_dir / name).touch()

    removed_system_v_keys = []
    monkeypatch.setattr(service_shm_cleanup, "SHM_DIR", shm_dir)
    monkeypatch.setattr(service_shm_cleanup, "_unlink_posix_shm", lambda name: (shm_dir / name).unlink())
    monkeypatch.setattr(
        service_shm_cleanup,
        "_remove_system_v_shm",
        lambda key: removed_system_v_keys.append(key) or True,
    )

    service_shm_cleanup.cleanup_service_shm("service_0", [101, 102])

    assert all(not (shm_dir / name).exists() for name in matching_names)
    assert (shm_dir / "service_1_req_pool").exists()
    assert (shm_dir / "other_service_0_value").exists()
    assert removed_system_v_keys == [101, 102]


def test_register_launcher_cleanup_recovers_dead_owner_and_records_current_service(monkeypatch, tmp_path):
    owner_dir = tmp_path / "owners"
    owner_dir.mkdir()
    old_owner_path = owner_dir / "old_service_0.json"
    old_owner_path.write_text(
        json.dumps(
            {
                "service_name": "old_service_0",
                "pid": 999999999,
                "create_time": 1.0,
                "system_v_shm_keys": [11],
            }
        ),
        encoding="utf-8",
    )

    cleanup_calls = []
    atexit_callbacks = []
    monkeypatch.setattr(service_shm_cleanup, "OWNER_DIR", owner_dir)
    monkeypatch.setattr(service_shm_cleanup, "OWNER_LOCK_PATH", tmp_path / "owners.lock")
    monkeypatch.setattr(service_shm_cleanup, "cleanup_service_shm", lambda *args: cleanup_calls.append(args))
    monkeypatch.setattr(service_shm_cleanup.atexit, "register", atexit_callbacks.append)
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps({"cpu_kv_cache_shm_id": 21, "multi_modal_cache_shm_id": 22}),
    )
    service_shm_cleanup._registered_cleanups.clear()

    cleanup = service_shm_cleanup.register_launcher_shm_cleanup("current_service_0")

    assert cleanup_calls == [("old_service_0", [11])]
    assert not old_owner_path.exists()
    current_owner_path = owner_dir / "current_service_0.json"
    current_owner = json.loads(current_owner_path.read_text(encoding="utf-8"))
    assert current_owner["pid"] == os.getpid()
    assert current_owner["system_v_shm_keys"] == [21, 22]
    assert atexit_callbacks == [cleanup]

    cleanup()
    cleanup()
    assert cleanup_calls[-1] == ("current_service_0", [21, 22])
    assert cleanup_calls.count(("current_service_0", [21, 22])) == 1
    assert not current_owner_path.exists()
