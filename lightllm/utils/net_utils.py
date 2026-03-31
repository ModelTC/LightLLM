import socket
import subprocess
import ipaddress
import random
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

BASE_PORT = 10000
PORTS_PER_INSTANCE = 1000
MAX_INSTANCE_ID = 7


def _is_port_available(port: int) -> bool:
    """Check if a port is available by attempting to bind it."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", port))
        s.close()
        return True
    except OSError:
        return False


def alloc_can_use_network_port(num=3, used_nccl_ports=None, instance_id=0):
    if used_nccl_ports is None:
        used_nccl_ports = []

    range_start = BASE_PORT + instance_id * PORTS_PER_INSTANCE
    range_end = range_start + PORTS_PER_INSTANCE
    used_set = set(used_nccl_ports)

    port_list = []
    for port in range(range_start, range_end):
        if len(port_list) >= num:
            break
        if port in used_set:
            continue
        if _is_port_available(port):
            port_list.append(port)

    if len(port_list) < num:
        raise RuntimeError(
            f"Failed to allocate {num} ports for instance {instance_id} "
            f"in range [{range_start}, {range_end}), only got {len(port_list)}"
        )

    logger.info(f"Instance {instance_id}: allocated {len(port_list)} ports in [{range_start}, {range_end}): {port_list}")
    return port_list


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
