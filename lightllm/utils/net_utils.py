import socket
import subprocess
import ipaddress
import os
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

DEFAULT_BASE_PORT = 10000
PORTS_PER_INSTANCE = 1000
MAX_INSTANCE_ID = 7


def _is_port_available(port: int) -> bool:
    """Check if a port is available by attempting to bind it."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            return True
    except OSError:
        return False


def alloc_can_use_network_port(num=3, used_nccl_ports=None, instance_id=0):
    """
    Allocate available network ports within an instance-specific range.

    Each instance gets a dedicated 1000-port range starting from BASE_PORT
    (default 10000, override via LIGHTLLM_BASE_PORT env var).
    Instance 0: 10000-10999, Instance 1: 11000-11999, etc.
    """
    if instance_id < 0 or instance_id > MAX_INSTANCE_ID:
        raise ValueError(f"instance_id must be in range [0, {MAX_INSTANCE_ID}], got {instance_id}")

    base_port = int(os.environ.get("LIGHTLLM_BASE_PORT", DEFAULT_BASE_PORT))
    range_start = base_port + instance_id * PORTS_PER_INSTANCE
    range_end = range_start + PORTS_PER_INSTANCE
    used_set = set(used_nccl_ports) if used_nccl_ports else set()

    port_list = []
    for port in range(range_start, range_end):
        if len(port_list) >= num:
            break
        if port in used_set:
            continue
        if _is_port_available(port):
            port_list.append(port)
            used_set.add(port)

    if len(port_list) >= num:
        logger.info(f"Instance {instance_id}: allocated {len(port_list)} ports in [{range_start}, {range_end}): {port_list}")
        return port_list

    raise RuntimeError(
        f"Failed to allocate {num} ports for instance {instance_id} in range [{range_start}, {range_end}). "
        f"Only found {len(port_list)} available. Try a different instance_id or set LIGHTLLM_BASE_PORT."
    )


def alloc_can_use_port(min_port, max_port):
    port_list = []
    for port in range(min_port, max_port):
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = test_socket.connect_ex(("localhost", port))
            test_socket.close()

            if result != 0:
                port_list.append(port)
        except Exception:
            continue
    return port_list


def find_available_port(start_port, end_port):
    for port in range(start_port, end_port + 1):
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = test_socket.connect_ex(("localhost", port))
            test_socket.close()

            if result != 0:
                return port
        except Exception:
            continue
    return None


def get_hostname_ip():
    try:
        result = subprocess.run(["hostname", "-i"], capture_output=True, text=True, check=True)
        # 兼容 hostname -i 命令输出多个 ip 的情况
        result = result.stdout.strip().split(" ")[0]
        logger.info(f"get hostname ip {result}")
        return result
    except subprocess.CalledProcessError as e:
        logger.exception(f"Error executing command: {e}")
        return None


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


class PortLocker:
    def __init__(self, ports):
        self.ports = ports
        self.sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(len(self.ports))]
        for _socket in self.sockets:
            _socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def lock_port(self):
        for _socket, _port in zip(self.sockets, self.ports):
            try:
                _socket.bind(("", _port))
                _socket.listen(1)
            except Exception as e:
                logger.error(f"port {_port} has been used")
                raise e

    def release_port(self):
        for _socket in self.sockets:
            _socket.close()
