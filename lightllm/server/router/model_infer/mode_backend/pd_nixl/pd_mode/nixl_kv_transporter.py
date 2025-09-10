import pickle
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from torch import Tensor
from lightllm.server.pd_io_struct import NIXLChunckedTransTask, NixlAgentMetadata, NIXLChunckedTaskSuccessRet
from lightllm.utils.log_utils import init_logger

from ..pd_remote_prefill_obj import (
    RemoteAgent,
    KVMoveRequest,
    PrefillRequest,
    RemotePrefillStatus,
    ThreadSafeDict,
    KVMoveRequestState,
    PageTransferAck,
    RemoteTransferStatusType,
    RemoteTransferType,
    NotificationType,
    Notification,
)


logger = init_logger(__name__)

try:
    from nixl._api import nixl_agent as NixlWrapper
    from nixl._api import nixlBind

    logger.info("Nixl is available")
except ImportError:
    logger.warning("nixl is not installed, which is required for pd disagreggation!!!")
    NixlWrapper = None


class NixlKVTransporter:
    def __init__(self, node_id: int, tp_idx: int, kv_move_buffer: Tensor):
        self.node_id = node_id
        self.tp_idx = tp_idx
        self.nixl_agent = NixlWrapper(self.agent_name, None)
        self.remote_agents: Dict[str, NixlAgentMetadata] = {}
        self._register_kv_move_buffer(kv_move_buffer=kv_move_buffer)

        self.inflight_page_transfers: Dict[int, Tuple[NIXLChunckedTransTask, int]] = {}
        return

    @property
    def agent_name(self) -> str:
        return f"{self.node_id}_{self.tp_idx}"

    @property
    def agent_metadata(self):
        return self.nixl_agent.get_agent_metadata()

    @property
    def local_page_mem_desc(self):
        return self.nixl_agent.get_serialized_descs(self.page_reg_desc)

    def get_new_notifs(self):
        return self.nixl_agent.get_new_notifs()

    def _register_kv_move_buffer(self, kv_move_buffer: Tensor):
        self.num_pages, self.page_size, self.num_layers, self.kv_head_num, self.head_dims = kv_move_buffer.shape
        self.page_len = self.page_size * self.num_layers * self.kv_head_num * self.head_dims
        self.page_reg_desc = self.nixl_agent.register_memory(kv_move_buffer)
        self.page_local_xfer_handles = self._create_paged_xfer_handles(self.page_reg_desc, self.num_pages)

    def _create_paged_xfer_handles(self, reg_desc: nixlBind.nixlRegDList, page_num: int, agent_name: str = ""):
        base_addr, _, device_id, _ = reg_desc[0]
        pages_data = []
        for page_id in range(page_num):
            pages_data.append((base_addr + page_id * self.page_len, self.page_len, device_id))
        descs = self.nixl_agent.get_xfer_descs(pages_data, "VRAM", True)
        return self.nixl_agent.prep_xfer_dlist(agent_name, descs, is_sorted=True)

    def connect_add_remote_agent(self, remote_agent: NixlAgentMetadata):
        peer_name = self.nixl_agent.add_remote_agent(remote_agent.agent_metadata)
        if isinstance(peer_name, bytes):
            peer_name = peer_name.decode()

        assert (
            peer_name == remote_agent.agent_name
        ), f"Peer name {peer_name} does not match remote name {remote_agent.agent_name}"

        self.nixl_agent.send_notif(
            peer_name, Notification(type=NotificationType.REMOTE_MD, data=self.agent_metadata).to_bytes()
        )

        page_mem_desc = self.nixl_agent.deserialize_descs(remote_agent.page_reg_desc)
        kv_page_xfer_handles = self._create_paged_xfer_handles(
            page_mem_desc, remote_agent.num_pages, agent_name=peer_name
        )
        remote_agent.page_remote_xfer_handles = kv_page_xfer_handles

        logger.info("Added remote agent %s with mem desc %s", peer_name, page_mem_desc)
        return

    def send_readtask_to_decode_node(self, remote_agent_name: str, trans_task: NIXLChunckedTransTask):
        """
        prefill node call this function to send read task to decode node
        """
        # 将页面读取任务发送给 decode 节点
        remote_agent: NixlAgentMetadata = self.remote_agents[remote_agent_name]
        assert trans_task.nixl_src_page_index is not None
        new_trans_task: NIXLChunckedTransTask = trans_task.copy()
        # 不需要传输细节的 mem_indexes 信息
        new_trans_task.mem_indexes = None
        self.nixl_agent.send_notif(
            remote_agent.agent_name,
            pickle.dumps(new_trans_task),
        )
        return

    def read_blocks_paged(
        self,
        remote_agent_name: str,
        trans_task: NIXLChunckedTransTask,
    ):
        """
        decode node call this function to read kv blocks from prefill node
        """
        assert trans_task.nixl_src_page_index is not None and trans_task.nixl_dst_page_index is not None
        remote_agent: NixlAgentMetadata = self.remote_agents[remote_agent_name]
        src_handle = remote_agent.page_remote_xfer_handles
        dst_handle = self.page_local_xfer_handles
        notify_obj = NIXLChunckedTaskSuccessRet(trans_id=trans_task.trans_id)
        handle = self.nixl_agent.make_prepped_xfer(
            "READ",
            dst_handle,
            [trans_task.nixl_dst_page_index],
            src_handle,
            [trans_task.nixl_src_page_index],
            pickle.dumps(notify_obj),
        )
        status = self.nixl_agent.transfer(handle)
        assert status != "ERR", f"Transfer failed with status {status} for handle {handle}"
        self.inflight_page_transfers[trans_task.trans_id] = (trans_task, handle)
        return

    async def get_done_page_transfers(self):
        done_taskes = []
        for trans_id, (trans_task, handle) in self.inflight_page_transfers.items():
            xfer_state = self.nixl_agent.check_xfer_state(handle)
            if xfer_state == "DONE":
                done_taskes.append(trans_task)
                self.nixl_agent.release_xfer_handle(handle)
                self.inflight_page_transfers.pop(trans_id, None)
            elif xfer_state == "PROC":
                continue
            else:
                logger.warning(f"Transfer failed with trans task {trans_task} for handle {handle}")
                self.nixl_agent.release_xfer_handle(handle)
                del self.inflight_page_transfers[handle]

        return done_taskes

    def shutdown(self):
        self.nixl_agent.deregister_memory(self.page_reg_desc)
        self.nixl_agent.release_dlist_handle(self.page_local_xfer_handles)
        for agent_name, remote_agent in self.remote_agents.items():
            self.nixl_agent.remove_remote_agent(remote_agent.agent_name)
            if remote_agent.page_remote_xfer_handles is not None:
                self.nixl_agent.release_dlist_handle(remote_agent.page_remote_xfer_handles)

        for trans_id, (trans_task, handle) in self.inflight_page_transfers:
            self.nixl_agent.release_xfer_handle(handle)
