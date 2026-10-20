"""HTTP worker client for the process-owned request ID generator."""

import threading

from rpyc.utils.classic import obtain
from rpyc.utils.factory import unix_connect

from .req_id_generator_server import (
    CLIENT_REQUEST_ID_RANGE_SIZE,
    MAX_BEST_OF,
    get_req_id_generator_socket_address,
)

RPC_TIMEOUT_SECONDS = 5


class ReqIDGenerator:
    """Generate request IDs through the launcher-owned RPyC service."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_id = 0
        self.range_end = 0
        self.connection = unix_connect(
            get_req_id_generator_socket_address(),
            config={
                "sync_request_timeout": RPC_TIMEOUT_SECONDS,
                "allow_pickle": True,
            },
        )

    def generate_id(self) -> int:
        with self.lock:
            if self.next_id >= self.range_end:
                self.next_id, self.range_end = obtain(
                    self.connection.root.allocate_id_range(CLIENT_REQUEST_ID_RANGE_SIZE)
                )
            request_id = self.next_id
            self.next_id += MAX_BEST_OF
            return request_id
