from .dp_base_balancer import RoundRobinDpBalancer
from typing import List
from lightllm.server.router.req_queue.base_queue import BaseQueue
from .dp_balancer_for_pd import DpBalancerForPd


def get_dp_balancer(args, dp_size_in_node: int, inner_queues: List[BaseQueue]):
    if args.dp_balancer == "round_robin":
        return DpBalancerForPd(dp_size_in_node, inner_queues)
    if args.run_mode in ["prefill", "decode"]:
        return DpBalancerForPd(dp_size_in_node, inner_queues)
    else:
        raise ValueError(f"Invalid dp balancer: {args.dp_balancer}")
