from lightllm.utils import shm_utils


def test_get_service_shm_name_adds_prefix_once(monkeypatch):
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: "service_uuid_0")

    assert shm_utils.get_service_shm_name("req_pool") == "service_uuid_0_req_pool"
    assert shm_utils.get_service_shm_name("service_uuid_0_req_pool") == "service_uuid_0_req_pool"


def test_create_or_link_shm_passes_scoped_name_to_shared_memory_layer(monkeypatch):
    created_names = []
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: "service_uuid_1")
    monkeypatch.setattr(
        shm_utils,
        "_force_create_shm",
        lambda name, expected_size: created_names.append((name, expected_size)) or object(),
    )

    shm_utils.create_or_link_shm("token_load", 128, force_mode="create")

    assert created_names == [("service_uuid_1_token_load", 128)]
