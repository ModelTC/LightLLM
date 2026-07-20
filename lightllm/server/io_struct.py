"""HTTP / Router / Model 之间的 RL 与运维控制面数据结构。

本文件只放「非生成主路径」的类型，结构如下：

1. **Generation control** — ``AbortReq``（多在 HTTP/RL controller 本地处理）
2. **RL control plane** — 统一经 ``RlOpReq`` 下发到 Model ``RlBackendOps``
   - **Envelope**：``RlOpReq`` / ``RlOpRsp``（传输信封）
   - **Op payloads**：具体操作的入参，只是 ``op_name`` 不同
     - cache / memory：``FlushCacheReq`` / ``ReleaseMemoryReq`` / ``ResumeMemoryReq``
     - weight update：Init/Destroy group、三种 UpdateWeights*

信封与 payload 不是两类类东西：payload 填进 ``RlOpReq.op_args``，
由 ``op_name`` 决定后端调用哪个方法。

生成请求的采样/多模态参数不在此文件。
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

from lightllm.utils.torch_memory_saver_utils import MemoryTag


# ---------------------------------------------------------------------------
# 1. Generation control
# ---------------------------------------------------------------------------


@dataclass
class AbortReq:
    """中止指定请求，或中止当前所有请求。

    ``request_id`` 对应内部的 ``group_req_id``。
    """

    request_id: Optional[int] = None
    abort_all: bool = False


# ---------------------------------------------------------------------------
# 2. RL control plane
# ---------------------------------------------------------------------------


# 2.1 Envelope — HTTP → Router → Model 的统一传输层


@dataclass
class RlOpReq:
    """RL 控制面请求信封。

    ``op_name`` 对应 :class:`RlBackendOps` 中的方法名，
    ``op_args`` 为该方法的入参（下列 Op payload，或已拆好的 tags 等）。
    """

    op_name: str
    op_args: Optional[Any] = None


@dataclass
class RlOpRsp:
    """RL 控制面响应信封。"""

    success: bool
    msg: Optional[str]
    op_name: str
    op_result: Optional[Any] = None


# 2.2 Op payloads — cache / memory
#     与权重更新同属 RlBackendOps 可 dispatch 的操作，仅 op_name / 入参不同。


def _normalize_memory_tags(tags):
    if tags is None:
        return None
    return [tag if isinstance(tag, MemoryTag) else MemoryTag(tag) for tag in tags]


@dataclass
class FlushCacheReq:
    """清空 radix / prompt cache。对应 op: ``flush_cache``。"""

    pass


@dataclass
class ReleaseMemoryReq:
    """暂停指定 MemoryTag 显存占用。对应 op: ``release_memory_occupation``。"""

    tags: Optional[List[MemoryTag]] = None

    def __post_init__(self):
        self.tags = _normalize_memory_tags(self.tags)


@dataclass
class ResumeMemoryReq:
    """恢复先前 pause 的显存占用。对应 op: ``resume_memory_occupation``。"""

    tags: Optional[List[MemoryTag]] = None

    def __post_init__(self):
        self.tags = _normalize_memory_tags(self.tags)


# 2.3 Op payloads — weight update


@dataclass
class InitWeightsUpdateGroupReq:
    """初始化在线权重更新 process group。对应 op: ``init_weights_update_group``。"""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str = "weight_update_group"
    backend: str = "nccl"


@dataclass
class DestroyWeightsUpdateGroupReq:
    """销毁权重更新 process group。对应 op: ``destroy_weights_update_group``。"""

    group_name: str = "weight_update_group"


@dataclass
class UpdateWeightsFromDistributedReq:
    """经 NCCL group 拉取并更新权重。对应 op: ``update_weights_from_distributed``。"""

    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    group_name: str = "weight_update_group"
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromTensorReq:
    """经序列化 tensor 更新权重。对应 op: ``update_weights_from_tensor``。"""

    serialized_named_tensors: List[Union[str, bytes]]
    load_format: Optional[str] = None
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromIPCReq:
    """经 CUDA IPC / shm 更新权重。对应 op: ``update_weights_from_ipc``。"""

    ipc_handle: Optional[Union[str, dict]] = None
    use_shm: bool = False
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None
