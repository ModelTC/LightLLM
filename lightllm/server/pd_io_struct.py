import enum
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from fastapi import WebSocket

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# 节点的行为
class NodeRole(enum.Enum):
    P = "prefill"
    D = "decode"

    NP = "nixl_prefill"
    ND = "nixl_decode"

    NORMAL = "normal"
    PD_MASTER = "pd_master"

    def is_D(self):
        return self == NodeRole.D or self == NodeRole.ND

    def is_P(self):
        return self == NodeRole.P or self == NodeRole.NP

    def is_normal(self):
        return self == NodeRole.NORMAL

    def is_P_or_NORMAL(self):
        return self.is_P() or self.is_normal()

    def is_P_or_D(self):
        return self.is_P() or self.is_D()

    def is_NP_or_ND(self):
        return self == NodeRole.NP or self == NodeRole.ND


class ObjType(enum.Enum):
    ABORT = 1
    REQ = 2
    TOKEN_PACKS = 3


@dataclass
class _PD_Client_RunStatus:
    total_token_usage_rate: float = 0.0  # pd 节点上的 token 使用率


@dataclass
class PD_Client_Obj:
    node_id: int
    client_ip_port: str
    mode: str  # 只能是 prefill 或者 decode 节点
    start_args: object  # 节点的启动参数信息，用于做匹配性的校验，防止运行过程中出现问题。
    websocket: WebSocket = None  # 用于通信的 websocket 连接对象
    run_status: _PD_Client_RunStatus = field(default_factory=_PD_Client_RunStatus)

    def __post_init__(self):
        if self.mode not in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
            error_info = f"""mode must in ["prefill", "decode", "nixl_prefill", "nixl_decode"], but get {self.mode}"""
            logger.error(error_info)
            raise ValueError(error_info)
        return

    def to_llm_url(self):
        return f"http://{self.client_ip_port}/pd_generate_stream"


@dataclass
class PD_Master_Obj:
    node_id: int
    host_ip_port: str

    def to_log_str(self):
        return f"PD_MASTER host_ip_port: {self.host_ip_port} node_id: {self.node_id}"


@dataclass
class UpKVStatus:
    group_request_id: int
    #  The identifier of the pd_master node handling the request.
    pd_master_node_id: int
    # decode node dp_index to handle this request
    dp_index: int

    def __post_init__(self):
        if not isinstance(self.group_request_id, int):
            error_info = "group_request_id only can be int"
            logger.error(error_info)
            raise ValueError(error_info)

        if not isinstance(self.pd_master_node_id, int):
            error_info = "pd_master_node_id only can be int"
            logger.error(error_info)
            raise ValueError(error_info)
        return


@dataclass
class DecodeNodeInfo:
    node_id: int
    ip: str
    rpyc_port: str
    max_new_tokens: int
    pd_master_node_id: int


@dataclass
class PDTransJoinInfo:
    decode_id: int
    decode_device_id: int
    prefill_id: int
    prefill_device_id: int
    pd_prefill_nccl_ip: str
    pd_prefill_nccl_port: int
    # 用于标识一次唯一的连接，prefill_id 和 decode_id 相同时，可能因为网络原因重连，为了更好的区分
    # 一次连接，使用一个 uuid 为其标识
    connect_id: str


@dataclass
class PDTransLeaveInfo:
    decode_id: int
    prefill_id: int
    # 用于标识一次唯一的连接，prefill_id 和 decode_id 相同时，可能因为网络原因重连，为了更好的区分
    # 一次连接，使用一个 uuid 为其标识
    connect_id: str


@dataclass
class KVMoveTask:
    group_request_id: int
    input_tokens: List[int]  # 代表输入的token_id 序列
    prefill_token_indexes: List[int]  # 在prefill节点上 mem manager kv buffer中的token index
    # 在decode节点上 mem manager kv buffer中的token index, 其代表的是真实占用的额外token，并不与prefill_token_indexes 一样长
    decode_token_indexes: List[int]
    move_kv_len: int  # 因为 prompt cache 的原因，当prefill节点和decode节点沟通后，传输的kv的数量可能少于 prefill_value 的长度
    prefill_node_id: int
    decode_node: DecodeNodeInfo
    # 保存prefill 和 decode 节点对应处理的dp_index, 如果是普通tp模式，这个值一定是0,
    # 如果是deepseekv2的tp dp 混合模式, 才有真正的意义。
    prefill_dp_index: int
    decode_dp_index: int
    mark_start_time: float = None
    # 标记任务使用某个连接id进行传输
    connect_id: str = None

    def __post_init__(self):
        if len(self.input_tokens) <= 0:
            error_info = "key must len >= 1"
            logger.error(error_info)
            raise ValueError(error_info)

    def to_prefill_log_info(self):
        v_len = None if self.prefill_token_indexes is None else len(self.prefill_token_indexes)
        d_i = self.prefill_dp_index
        id = self.group_request_id
        log = f"id: {id} in_len:{len(self.input_tokens)} v_len: {v_len} move_len: {self.move_kv_len} dp_index:{d_i}"
        return log + f" connect_id: {self.connect_id}"

    def to_decode_log_info(self):
        v_len = None if self.decode_token_indexes is None else len(self.decode_token_indexes)
        d_i = self.decode_dp_index
        id = self.group_request_id
        log = f"id: {id} in_len:{len(self.input_tokens)} v_len: {v_len} move_len: {self.move_kv_len} dp_index:{d_i}"
        return log + f" connect_id: {self.connect_id}"

    def id(self):
        return self.group_request_id

    def get_cost_time(self):
        if self.mark_start_time is not None:
            return time.time() - self.mark_start_time
        else:
            return 100000000000


@dataclass
class KVMoveTaskGroup:
    tasks: List[KVMoveTask]
    connect_id: str


####### 下边是 NIXL模式下使用的特定对象 ########

@dataclass
class NixlUpKVStatus:
    group_request_id: int
    pd_master_node_id: int
    nixl_params: bytes  # nixl 建立连接所使用的元数据对象

    def __post_init__(self):

        if not isinstance(self.group_request_id, int):
            error_info = "group_request_id only can be int"
            logger.error(error_info)
            raise ValueError(error_info)

        if not isinstance(self.pd_master_node_id, int):
            error_info = "pd_master_node_id only can be int"
            logger.error(error_info)
            raise ValueError(error_info)
        return

@dataclass
class NIXLDecodeNodeInfo:
    decode_node_id: int
    pd_master_node_id: int

    agent_name: str
    agent_metadata: bytes
    num_pages: int
    page_reg_desc: bytes

    ready_kv_len: int  # decode 节点上已经准备好的kv长度

@dataclass
class NixlAgentMetadata:
    agent_name: str
    agent_metadata: bytes
    num_pages: int
    page_reg_desc: Optional[bytes] = None
    page_xfer_handles: Optional[int] = None


@dataclass
class NIXLChunckedTransTask:
    request_id: int
    start_kv_index: int
    end_kv_index: int

    prefill_dp_index: Optional[int]
    decode_dp_index: Optional[int]
    src_device_id: Optional[int]  # 传输设备 id
    dst_device_id: Optional[int]  # 接收设备 id
     
    mem_indexes: List[int]

    peer_agent_name: Optional[str]
    peer_agent_metadata: Optional[bytes]
    peer_num_pages: Optional[int]
    peer_page_req_desc: Optional[bytes]
    peer_page_xfer_handles: Optional[int]

    # transfer params
    nixl_src_page_index: Optional[int] = None
    nixl_dst_page_index: Optional[int] = None



    def __post_init__(self):
        if self.start_kv_index < 0 or self.end_kv_index <= self.start_kv_index:
            error_info = "start_kv_index must >=0 and end_kv_index > start_kv_index"
            logger.error(error_info)
            raise ValueError(error_info)
        if len(self.mem_indexes) == 0:
            error_info = "mem_indexes must len > 0"
            logger.error(error_info)
            raise ValueError(error_info)
        assert len(self.mem_indexes) == (self.end_kv_index - self.start_kv_index)
        return


@dataclass
class NIXLChunckedTaskSuccessRet:
    trans_id: int  # 每一个传送事件都有一个唯一id


@dataclass
class ChunckedTransTaskRet:
    request_id: int
    start_kv_index: int
    end_kv_index: int
    is_error: bool
    error_info: str = None



