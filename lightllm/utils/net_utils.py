import socket
import subprocess
import ipaddress
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_abstract_unix_socket_address(service_name: str, ipc_name: str) -> bytes:
    """Build a Linux IPC address that the kernel releases when its sockets close."""
    return f"\0lightllm_{ipc_name}_{service_name}".encode()


def get_pd_master_unix_socket_address(service_name: str) -> bytes:
    """Build a shared PD master IPC address from the service name."""
    from lightllm.utils.envs_utils import get_unique_server_name

    unique_server_name = get_unique_server_name()
    if unique_server_name is None:
        raise RuntimeError("Unique server name is not initialized")
    return get_abstract_unix_socket_address(service_name, unique_server_name)


def format_host_port(host: str, port: int) -> str:
    """Format an HTTP/WebSocket authority, including bracketed IPv6 hosts."""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{port}"


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
