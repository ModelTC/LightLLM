import socket
import subprocess
import ipaddress
import random
import os
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

DEFAULT_BASE_PORT = 10000
PORTS_PER_INSTANCE = 1000
MAX_INSTANCE_ID = 7


def alloc_can_use_network_port(num=3, used_ports=None, from_port_num=DEFAULT_BASE_PORT, instance_id=0):
    if instance_id < 0 or instance_id > MAX_INSTANCE_ID:
        raise ValueError(f"instance_id must be in range [0, {MAX_INSTANCE_ID}], got {instance_id}")

    base_port = int(os.environ.get("LIGHTLLM_BASE_PORT", from_port_num))
    # Keep independent launchers away from the same free-port window, especially for NCCL TCPStore ports.
    range_start = base_port + instance_id * PORTS_PER_INSTANCE
    range_end = range_start + PORTS_PER_INSTANCE
    used_ports = used_ports or []

    port_list = []
    for port in range(range_start, range_end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(("localhost", port))
            if result != 0 and port not in used_ports:
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
