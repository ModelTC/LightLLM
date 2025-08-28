from .roundrobin import RoundRobinDpBalancer
from typing import List
from lightllm.server.router.req_queue.base_queue import BaseQueue
from .bs import DpBsBalancer


def get_dp_balancer(args, dp_size_in_node: int, inner_queues: List[BaseQueue]):
    if args.dp_balancer == "round_robin":
        return RoundRobinDpBalancer(dp_size_in_node, inner_queues)
    elif args.dp_balancer == "bs_balancer":
        return DpBsBalancer(dp_size_in_node, inner_queues)
    else:
        raise ValueError(f"Invalid dp balancer: {args.dp_balancer}")
