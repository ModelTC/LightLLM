import zmq
import socket
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def send_and_receive_node_ip(args):
    # 传输子node的ip
    if args.nnodes > 1:

        if args.node_rank == 0:
            args.child_ips = None
            args.child_ips = []
            for i in range(1, args.nnodes):
                context = zmq.Context(2)
                comm_socket = context.socket(zmq.PULL)
                comm_socket.bind(f"tcp://*:{args.multinode_httpmanager_port + i + 100}")
                logger.info(f"binding port {args.multinode_httpmanager_port + i + 100}")
                args.child_ips.append(comm_socket.recv_pyobj())
                comm_socket.close()
            logger.info(f"Received child IPs: {args.child_ips}")
        else:
            local_ip = socket.gethostbyname(socket.gethostname())
            context = zmq.Context(2)
            comm_socket = context.socket(zmq.PUSH)
            comm_socket.connect(f"tcp://{args.nccl_host}:{args.multinode_httpmanager_port + args.node_rank + 100}")
            logger.info(f"connecting to {args.nccl_host}:{args.multinode_httpmanager_port + args.node_rank + 100}")
            comm_socket.send_pyobj(local_ip)
            comm_socket.close()
