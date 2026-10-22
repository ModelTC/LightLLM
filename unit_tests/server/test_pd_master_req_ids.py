"""Request IDs are owned by one RPyC service instead of shared memory."""

import socket
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rpyc.utils.server import ThreadedServer

from lightllm.server.httpserver import req_id_generator_client
from lightllm.server.httpserver.req_id_generator_server import (
    CLIENT_REQUEST_ID_RANGE_SIZE,
    CONFIG_SERVER_REQUEST_TIMEOUT_SECONDS,
    MAX_BEST_OF,
    get_req_id_generator_socket_address,
)
from lightllm.utils.net_utils import get_abstract_unix_socket_address
from lightllm.server.httpserver.req_id_generator_client import ReqIDGenerator
from lightllm.server.httpserver.req_id_generator_server import (
    ConfigServerReqIDGeneratorState,
    StandaloneReqIDGeneratorState,
    _create_req_id_generator_state,
)


def _args(**overrides):
    values = {
        "run_mode": "normal",
        "config_server_host": None,
        "config_server_port": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _remote_client(ipc_name):
    socket_address = get_abstract_unix_socket_address("req_id_generator", ipc_name)
    with patch.object(
        req_id_generator_client,
        "get_req_id_generator_socket_address",
        return_value=socket_address,
    ):
        return ReqIDGenerator()


def test_server_allocates_aligned_request_id_ranges():
    state = StandaloneReqIDGeneratorState()

    first = state.allocate_id_range(4)
    second = state.allocate_id_range(2)

    assert first == (8, 40)
    assert second == (40, 56)


def test_client_generates_locally_until_its_range_is_exhausted(monkeypatch):
    connection = Mock()
    connection.root.allocate_id_range.side_effect = [(8, 8200), (8200, 16392)]
    monkeypatch.setattr(req_id_generator_client, "unix_connect", Mock(return_value=connection))
    client = _remote_client("request-id-client")

    first_range = [client.generate_id() for _ in range(CLIENT_REQUEST_ID_RANGE_SIZE)]

    assert first_range == list(range(8, 8200, MAX_BEST_OF))
    connection.root.allocate_id_range.assert_called_once_with(CLIENT_REQUEST_ID_RANGE_SIZE)
    assert client.generate_id() == 8200
    assert connection.root.allocate_id_range.call_count == 2


def test_config_server_range_is_used_and_renewed(monkeypatch):
    responses = [
        Mock(status_code=200, json=Mock(return_value={"start_id": 0, "end_id": 800000})),
        Mock(
            status_code=200,
            json=Mock(return_value={"start_id": 800000, "end_id": 1600000}),
        ),
    ]
    get = Mock(side_effect=responses)
    monkeypatch.setattr("lightllm.server.httpserver.req_id_generator_server.requests.get", get)
    state = ConfigServerReqIDGeneratorState("10.0.0.1", 9000)

    assert state.allocate_id_range(1) == (8, 16)
    state.next_id = state.range_end - MAX_BEST_OF
    assert state.allocate_id_range(1) == (799992, 800000)
    assert state.allocate_id_range(2) == (800000, 800016)
    assert [call.args for call in get.call_args_list] == [
        ("http://10.0.0.1:9000/allocate_global_unique_id_range",),
        ("http://10.0.0.1:9000/allocate_global_unique_id_range",),
    ]
    assert [call.kwargs for call in get.call_args_list] == [
        {"timeout": CONFIG_SERVER_REQUEST_TIMEOUT_SECONDS},
        {"timeout": CONFIG_SERVER_REQUEST_TIMEOUT_SECONDS},
    ]


def test_request_id_socket_uses_configured_ipc_name():
    with patch(
        "lightllm.server.httpserver.req_id_generator_server.get_pd_master_unix_socket_address",
        return_value=get_abstract_unix_socket_address("req_id_generator", "launcher-name"),
    ):
        assert get_req_id_generator_socket_address().endswith(b"launcher-name_req_id_generator")


def test_state_type_is_selected_once_during_server_startup():
    assert isinstance(_create_req_id_generator_state(_args()), StandaloneReqIDGeneratorState)
    assert isinstance(
        _create_req_id_generator_state(_args(config_server_host="127.0.0.1")),
        StandaloneReqIDGeneratorState,
    )
    assert isinstance(
        _create_req_id_generator_state(_args(config_server_port=9000)),
        StandaloneReqIDGeneratorState,
    )
    assert isinstance(
        _create_req_id_generator_state(
            _args(
                run_mode="normal",
                config_server_host="127.0.0.1",
                config_server_port=9000,
            )
        ),
        ConfigServerReqIDGeneratorState,
    )


def test_real_rpc_clients_share_one_sequence():
    ipc_name = uuid.uuid4().hex
    state = StandaloneReqIDGeneratorState()
    server = ThreadedServer(
        state,
        socket_path=get_abstract_unix_socket_address("req_id_generator", ipc_name),
        protocol_config={"allow_pickle": True},
    )
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    clients = [_remote_client(ipc_name) for _ in range(4)]
    try:
        ids = [clients[index % len(clients)].generate_id() for index in range(200)]
        assert len(set(ids)) == len(ids)
        assert all(request_id % MAX_BEST_OF == 0 for request_id in ids)
        assert state.next_id == 8 + 4 * CLIENT_REQUEST_ID_RANGE_SIZE * MAX_BEST_OF
        assert clients[0].connection._channel.stream.sock.family == socket.AF_UNIX
    finally:
        for client in clients:
            client.connection.close()
        server.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
