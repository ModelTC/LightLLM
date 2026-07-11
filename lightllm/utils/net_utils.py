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


class PortManager:
    def __init__(self, args, ports=()):
        self.args = args
        self.port_to_socket = {}
        self.lock_ports(ports)

    @classmethod
    def for_model_server(cls, args):
        ports = [args.port, args.rl_rpyc_port]
        if args.node_rank == 0:
            ports.append(args.nccl_port)
        if not args.disable_vision and not args.visual_use_proxy_mode:
            ports.extend((args.visual_nccl_ports or [])[: args.visual_dp])
        if not args.disable_audio:
            ports.extend((args.audio_nccl_ports or [])[: args.audio_dp])
        return cls(args, ports)

    def lock_ports(self, ports):
        for port in dict.fromkeys(port for port in ports if port is not None):
            if port in self.port_to_socket:
                continue
            port_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            port_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                port_socket.bind(("", port))
                port_socket.listen(1)
            except Exception:
                port_socket.close()
                logger.error(f"port {port} has been used")
                raise
            self.port_to_socket[port] = port_socket

    def allocate_ports(self, num, excluded_ports=()):
        used_ports = set(self.port_to_socket)
        used_ports.update(port for port in excluded_ports if port is not None)
        ports = alloc_can_use_network_port(
            num=num,
            instance_id=self.args.lightllm_instance_id,
            used_ports=used_ports,
        )
        if ports is None:
            raise RuntimeError(f"failed to allocate {num} network ports")
        self.lock_ports(ports)
        return ports

    def release_unused_ports(self):
        args = self.args
        active_ports = {
            args.port,
            args.router_port,
            args.detokenization_port,
            args.http_server_port,
            args.metric_port,
        }
        if args.enable_profiling:
            active_ports.add(args.router_profiler_port)
        if args.node_rank == 0:
            active_ports.update([args.nccl_port, args.rl_rpyc_port])
        if args.enable_multimodal:
            active_ports.add(args.cache_port)
        if not args.disable_vision:
            active_ports.add(args.visual_port)
            if not args.visual_use_proxy_mode:
                active_ports.update(args.visual_nccl_ports or [])
        if not args.disable_audio:
            active_ports.add(args.audio_port)
            active_ports.update(args.audio_nccl_ports or [])
        if args.enable_cpu_cache:
            active_ports.add(args.multi_level_kv_cache_port)
        self.release_ports(set(self.port_to_socket) - active_ports)

    def release_ports(self, ports):
        for port in ports:
            _socket = self.port_to_socket.pop(port, None)
            if _socket is not None:
                _socket.close()

    def release_all(self):
        for _socket in self.port_to_socket.values():
            _socket.close()
        self.port_to_socket.clear()
