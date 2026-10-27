"""Process-owned request ID generator and its RPyC service."""

import threading
import time

import requests
import rpyc
from rpyc.utils.classic import obtain

from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_pd_master_unix_socket_address

logger = init_logger(__name__)

# group_req_id 按 MAX_BEST_OF 对齐，使 sub_req_id 可以直接映射回 group_req_id。
MAX_BEST_OF = 8
# 每个 HTTP worker 本地缓存 1024 个 group request ID，减少逐请求 RPC。
CLIENT_REQUEST_ID_RANGE_SIZE = 1024
CONFIG_SERVER_REQUEST_TIMEOUT_SECONDS = 10


def convert_sub_id_to_group_id(sub_req_id: int) -> int:
    return (sub_req_id // MAX_BEST_OF) * MAX_BEST_OF


def get_req_id_generator_socket_address() -> bytes:
    return get_pd_master_unix_socket_address("req_id_generator")


@rpyc.service
class StandaloneReqIDGeneratorState(rpyc.Service):
    """Allocate request IDs from this server's independent ID space."""

    def __init__(self):
        self.next_id = MAX_BEST_OF
        self.range_end = 1 << 63
        self.lock = threading.Lock()

    @rpyc.exposed
    def allocate_id_range(self, request_count: int) -> tuple[int, int]:
        request_count = int(obtain(request_count))
        if request_count <= 0:
            raise ValueError("request_count must be positive")
        with self.lock:
            required_range_size = request_count * MAX_BEST_OF
            self._check_id_range(required_range_size)
            allocated_range_start = self.next_id
            allocated_range_end = allocated_range_start + required_range_size
            self.next_id = allocated_range_end
            return allocated_range_start, allocated_range_end

    def _check_id_range(self, required_range_size: int) -> None:
        if self.next_id + required_range_size > self.range_end:
            raise OverflowError("request ID space is exhausted")


@rpyc.service
class ConfigServerReqIDGeneratorState(rpyc.Service):
    """Allocate request IDs from large ranges supplied by a config server."""

    def __init__(self, config_server_host: str, config_server_port: int):
        self.config_server_url = f"http://{config_server_host}:{config_server_port}" "/allocate_global_unique_id_range"
        self.next_id = 0
        self.range_end = 0
        self.lock = threading.Lock()

    @rpyc.exposed
    def allocate_id_range(self, request_count: int) -> tuple[int, int]:
        request_count = int(obtain(request_count))
        if request_count <= 0:
            raise ValueError("request_count must be positive")
        with self.lock:
            required_range_size = request_count * MAX_BEST_OF
            self._check_and_set_new_id_range(required_range_size)
            allocated_range_start = self.next_id
            allocated_range_end = allocated_range_start + required_range_size
            self.next_id = allocated_range_end
            return allocated_range_start, allocated_range_end

    def _check_and_set_new_id_range(self, required_range_size: int) -> None:
        if self.next_id + required_range_size > self.range_end:
            self._request_id_range(required_range_size)

    def _request_id_range(self, required_range_size: int) -> None:
        while True:
            try:
                response = requests.get(
                    self.config_server_url,
                    timeout=CONFIG_SERVER_REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to fetch ID range from config server: {response.status_code}")
                id_range = response.json()
                logger.info("get new id range %s", id_range)
                range_start = id_range["start_id"]
                self.next_id = max(
                    MAX_BEST_OF,
                    (range_start + MAX_BEST_OF - 1) // MAX_BEST_OF * MAX_BEST_OF,
                )
                self.range_end = id_range["end_id"]
                assert self.next_id + required_range_size <= self.range_end, (
                    f"get id range error {self.next_id} {self.range_end}, " f"required_range_size={required_range_size}"
                )
                return
            except BaseException as exc:
                logger.exception(str(exc))
                time.sleep(3)


def start_req_id_generator_server(args, pipe_writer) -> None:
    import setproctitle
    from rpyc.utils.server import ThreadedServer

    from lightllm.utils.graceful_utils import graceful_registry
    from lightllm.utils.process_check import start_parent_check_thread

    start_parent_check_thread()
    graceful_registry("start_req_id_generator_server")
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::req_id_generator_server")
    server = ThreadedServer(
        _create_req_id_generator_state(args),
        socket_path=get_req_id_generator_socket_address(),
        protocol_config={"allow_pickle": True},
    )
    pipe_writer.send("init ok")
    server.start()


def _create_req_id_generator_state(args):
    if args.config_server_host is not None and args.config_server_port is not None:
        return ConfigServerReqIDGeneratorState(args.config_server_host, args.config_server_port)
    return StandaloneReqIDGeneratorState()
