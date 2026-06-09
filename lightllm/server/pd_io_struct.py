import enum
import time
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Set
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from fastapi import WebSocket

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# 节点的行为
class NodeRole(enum.Enum):
    NP = "nixl_prefill"
    ND = "nixl_decode"

    NORMAL = "normal"
    PD_MASTER = "pd_master"

    def is_D(self):
        return self == NodeRole.ND

    def is_P(self):
        return self == NodeRole.NP

    def is_NP(self):
        return self == NodeRole.NP

    def is_ND(self):
        return self == NodeRole.ND

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
    NIXL_UPLOAD_NP_PROMPT_IDS = 4  # nixl p 节点上报生成的 prompt ids 信息。
    NIXL_REQ_DECODE_NODE_INFO = 5  # nixl pd master 节点下发给 nixl p 节点的对应请求对应的 d 节点的信息。


@dataclass
class _PD_Client_RunStatus:
    total_token_usage_rate: float = 0.0  # pd 节点上的 token 使用率


@dataclass
class PD_Client_Obj:
    node_id: int
    client_ip_port: str
    mode: str  # 只能是 nixl_prefill 或者 nixl_decode 节点
    start_args: object  # 节点的启动参数信息，用于做匹配性的校验，防止运行过程中出现问题。
    websocket: WebSocket = None  # 用于通信的 websocket 连接对象
    run_status: _PD_Client_RunStatus = field(default_factory=_PD_Client_RunStatus)

    def __post_init__(self):
        if self.mode not in ["nixl_prefill", "nixl_decode"]:
            error_info = f"""mode must in ["nixl_prefill", "nixl_decode"], but get {self.mode}"""
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

    def __str__(self):
        req_id = self.group_request_id
        pd_m_id = self.pd_master_node_id
        return f"group_request_id: {req_id} pd_master_node_id: {pd_m_id} nixl_params_len: {len(self.nixl_params)}"


@dataclass
class NIXLDecodeNodeInfo:
    decode_node_id: int
    pd_master_node_id: int

    agent_name: str
    agent_metadata: bytes
    num_pages: int
    page_reg_desc: bytes

    request_id: int
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
    time_out_secs: int

    pd_master_node_id: int
    prefill_dp_index: Optional[int]
    decode_dp_index: Optional[int]
    src_device_id: Optional[int]  # 传输设备 id
    dst_device_id: Optional[int]  # 接收设备 id

    mem_indexes: List[int]

    prefill_agent_name: Optional[str]
    prefill_agent_metadata: Optional[bytes]
    prefill_num_pages: Optional[int]
    prefill_page_reg_desc: Optional[bytes]

    decode_agent_name: Optional[str]
    decode_agent_metadata: Optional[bytes]
    decode_num_pages: Optional[int]
    decode_page_reg_desc: Optional[bytes]

    first_gen_token_id: Optional[int]
    first_gen_token_logprob: Optional[float]

    nixl_write_stage: Optional[str] = None

    # transfer params
    nixl_src_page_index: Optional[int] = None
    nixl_dst_page_index: Optional[int] = None

    # xfer_handle
    xfer_handle: Optional[int] = None

    create_time: float = None
    start_trans_time: float = None  # 用于标记传输开始的时间。同时标记是否正在传输中

    error_info: Optional[str] = None
    transfer_time_out_secs: int = 66

    def __post_init__(self):
        if self.start_kv_index < 0 or self.end_kv_index < self.start_kv_index:
            error_info = "start_kv_index must >=0 and end_kv_index > start_kv_index"
            logger.error(error_info)
            raise ValueError(error_info)
        assert len(self.mem_indexes) == (self.end_kv_index - self.start_kv_index)
        self.create_time = time.time()
        return

    def time_out(self) -> bool:
        if self.start_trans_time is None:
            if time.time() - self.create_time > self.time_out_secs:
                return True
            return False
        else:
            if time.time() - self.start_trans_time > self.transfer_time_out_secs:
                return True
            else:
                return False

    def waiting_time(self):
        return time.time() - self.create_time

    def transfer_time(self):
        return time.time() - self.start_trans_time

    def get_key(self) -> str:
        return f"{self.request_id}_{self.start_kv_index}_{self.end_kv_index}"

    def to_str(self):
        obj: NIXLChunckedTransTask = copy.copy(self)
        obj.mem_indexes = None
        if obj.decode_agent_metadata is not None:
            obj.decode_agent_metadata = b"xxx"
        if obj.prefill_agent_metadata is not None:
            obj.prefill_agent_metadata = b"xxx"
        if obj.decode_page_reg_desc is not None:
            obj.decode_page_reg_desc = b"xxx"
        if obj.prefill_page_reg_desc is not None:
            obj.prefill_page_reg_desc = b"xxx"
        return obj.__str__()

    def transfer_kv_num(self):
        return self.end_kv_index - self.start_kv_index

    def createRetObj(self) -> "NIXLChunckedTransTaskRet":
        ret = NIXLChunckedTransTaskRet(
            request_id=self.request_id,
            start_kv_index=self.start_kv_index,
            end_kv_index=self.end_kv_index,
            has_error=self.error_info is not None,
            error_info=self.error_info,
            first_gen_token_id=self.first_gen_token_id,
            first_gen_token_logprob=self.first_gen_token_logprob,
        )
        return ret

    def create_prefill_agent_obj(self) -> NixlAgentMetadata:
        return NixlAgentMetadata(
            agent_name=self.prefill_agent_name,
            agent_metadata=self.prefill_agent_metadata,
            num_pages=self.prefill_num_pages,
            page_reg_desc=self.prefill_page_reg_desc,
        )

    def create_decode_agent_obj(self) -> NixlAgentMetadata:
        return NixlAgentMetadata(
            agent_name=self.decode_agent_name,
            agent_metadata=self.decode_agent_metadata,
            num_pages=self.decode_num_pages,
            page_reg_desc=self.decode_page_reg_desc,
        )


@dataclass
class NIXLChunckedTransTaskRet:
    request_id: int
    start_kv_index: int
    end_kv_index: int
    has_error: bool
    error_info: str = None
    first_gen_token_id: Optional[int] = None
    first_gen_token_logprob: Optional[float] = None

    def get_key(self) -> str:
        return f"{self.request_id}_{self.start_kv_index}_{self.end_kv_index}"


@dataclass
class NIXLChunckedTransTaskGroup:
    task_list: List[NIXLChunckedTransTask] = field(default_factory=list)


@dataclass
class NIXLAbortReq:
    request_id: int
    device_id: int
