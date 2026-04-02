from dataclasses import dataclass
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

rpyc_config = {
    "allow_pickle": True,
    "allow_all_attrs": True,
    "allow_getattr": True,
    "allow_setattr": True,
}


@dataclass
class VIT_Obj:
    node_id: int
    host_ip: str
    port: int

    def to_log_str(self):
        return f"VIT host_ip_port: {self.host_ip}:{self.port}, node_id: {self.node_id}"
