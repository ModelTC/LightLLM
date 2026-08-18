import ipaddress
import zmq
import socket
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_port_args import get_shm_port_args

logger = init_logger(__name__)

# 子节点上报的载荷只是一个 ip 字符串，正常不会超过 45 字节（ipv6 最长形式），
# 这里留一些余量，避免异常端或恶意端发送超大报文。
_MAX_CHILD_IP_BYTES = 64


def _decode_child_ip(raw: bytes) -> str:
    """把子节点上报的原始字节解析为 ip 字符串。

    这里刻意不使用 recv_pyobj：该端口必须绑定在所有网卡上（子节点要跨机连接过来），
    而 recv_pyobj 等价于对网络数据直接做 pickle.loads，任何能访问该端口的主机都可以
    构造恶意 pickle 在主节点上执行任意代码。载荷本身只是一个 ip，用 utf-8 解码加
    ipaddress 校验就足够，且能彻底消除反序列化执行路径。

    Raises:
        ValueError: 载荷过大、不是合法 utf-8 或不是合法 ip 时抛出，让启动直接失败，
            而不是把非法值写进 args.child_ips 后在后续建连时才报错。
    """
    if len(raw) > _MAX_CHILD_IP_BYTES:
        raise ValueError(f"child ip payload too large: {len(raw)} bytes > {_MAX_CHILD_IP_BYTES}")

    ip_str = raw.decode("utf-8").strip()
    ipaddress.ip_address(ip_str)  # 非法 ip 会抛 ValueError
    return ip_str


def send_and_receive_node_ip(args):
    # 在多节点tp的部署形式中，0 号节点作为主节点，其他节点作为
    # 从节点，0 号节点需要知道所有从节点的ip信息，这样才能构建
    # 一些通信组件转发请求信息给从节点。
    is_multinode_tp = args.dp == 1 and args.nnodes > 1
    if is_multinode_tp:
        base_port = get_shm_port_args().multinode_httpmanager_port
        if args.node_rank == 0:
            args.child_ips = None
            args.child_ips = []
            for i in range(1, args.nnodes):
                context = zmq.Context(2)
                comm_socket = context.socket(zmq.PULL)
                comm_socket.bind(f"tcp://*:{base_port + i + 100}")
                logger.info(f"binding port {base_port + i + 100}")
                try:
                    args.child_ips.append(_decode_child_ip(comm_socket.recv()))
                finally:
                    comm_socket.close()
            logger.info(f"Received child IPs: {args.child_ips}")
        else:
            local_ip = socket.gethostbyname(socket.gethostname())
            context = zmq.Context(2)
            comm_socket = context.socket(zmq.PUSH)
            comm_socket.connect(f"tcp://{args.nccl_host}:{base_port + args.node_rank + 100}")
            logger.info(f"connecting to {args.nccl_host}:{base_port + args.node_rank + 100}")
            comm_socket.send(local_ip.encode("utf-8"))
            comm_socket.close()
