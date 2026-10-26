from dataclasses import dataclass
from lightllm.utils.log_utils import init_logger
from .rpc_payload import MAX_RPC_FRAME_BYTES

logger = init_logger(__name__)
VISUAL_ONLY_RPC_HOST = "127.0.0.1"

rpyc_config = {
    "allow_pickle": True,
    "allow_all_attrs": True,
    "allow_getattr": True,
    "allow_setattr": True,
}

visual_only_rpyc_config = {
    "allow_pickle": False,
    "allow_all_attrs": False,
    "allow_getattr": True,
    "allow_setattr": False,
    "allow_delattr": False,
    "allow_public_attrs": False,
    "allow_exposed_attrs": True,
    "max_message_size": MAX_RPC_FRAME_BYTES,
}


@dataclass
class VIT_Obj:
    node_id: int
    host_ip: str
    port: int

    def to_log_str(self):
        return f"VIT host_ip_port: {self.host_ip}:{self.port}, node_id: {self.node_id}"
