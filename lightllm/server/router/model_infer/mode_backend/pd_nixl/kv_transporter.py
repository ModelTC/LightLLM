import os

from torch import Tensor

from lightllm.server.core.objs import StartArgs
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_hostname_ip

logger = init_logger(__name__)


def create_kv_transporter(args: StartArgs, node_id: int, tp_idx: int, kv_move_buffer: Tensor):
    backend = os.getenv("LIGHTLLM_PD_KV_TRANSPORT_BACKEND", "nixl").lower()
    if backend == "nixl":
        from .nixl_kv_transporter import NixlKVTransporter

        return NixlKVTransporter(node_id=node_id, tp_idx=tp_idx, kv_move_buffer=kv_move_buffer)

    if backend == "nccl":
        from .nccl_kv_transporter import NcclKVTransporter

        logger.info("Use NCCL as pd_nixl KV transporter backend")
        port_min = args.pd_p_allowed_port_min + tp_idx * 100
        port_max = min(args.pd_p_allowed_port_max, port_min + 99)
        if port_min > args.pd_p_allowed_port_max:
            port_min = args.pd_p_allowed_port_min
            port_max = args.pd_p_allowed_port_max
        return NcclKVTransporter(
            node_id=node_id,
            tp_idx=tp_idx,
            kv_move_buffer=kv_move_buffer,
            host_ip=get_hostname_ip() or args.host,
            store_port_min=port_min,
            store_port_max=port_max,
        )

    raise ValueError(f"unsupported LIGHTLLM_PD_KV_TRANSPORT_BACKEND={backend}")
