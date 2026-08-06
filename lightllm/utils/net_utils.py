import socket
import subprocess
import ipaddress
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


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


def validate_ports(ports: list):
    """校验端口列表：列表内不重复，且当前均可 bind。"""
    if len(ports) != len(set(ports)):
        raise RuntimeError(f"conflicting ports in list: {ports}")

    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
            except OSError as e:
                raise RuntimeError(f"port {port} has been used") from e
    logger.info(f"validated ports: {ports}")
