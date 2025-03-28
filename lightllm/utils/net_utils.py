import socket
import subprocess
import ipaddress
import random
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def alloc_can_use_network_port(num=3, used_nccl_ports=None, from_port_num=10000):
    port_list = []
    for port in range(from_port_num, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(("localhost", port))
            if result != 0 and port not in used_nccl_ports:
                port_list.append(port)
            if len(port_list) > num * 30:
                break

    if len(port_list) < num:
        return None

    random.shuffle(port_list)
    return port_list[0:num]


def alloc_can_use_port(min_port, max_port):
    port_list = []
    for port in range(min_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(("localhost", port))
            if result != 0:
                port_list.append(port)
    return port_list


def find_available_port(start_port, end_port):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(("localhost", port))
            if result != 0:
                return port
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
