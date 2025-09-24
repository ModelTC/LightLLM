import pickle
import copy
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from torch import Tensor
from lightllm.server.pd_io_struct import NIXLChunckedTransTask, NixlAgentMetadata, NIXLChunckedTransTaskRet
from lightllm.utils.log_utils import init_logger


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
        self._register_kv_move_buffer(kv_move_buffer=kv_move_buffer)
        self.remote_agents: Dict[str, NixlAgentMetadata] = {}
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

    def get_new_notifs(self) -> Dict[str, list[bytes]]:
        return self.nixl_agent.get_new_notifs()

    def _register_kv_move_buffer(self, kv_move_buffer: Tensor):
        self.num_pages, self.page_size, self.num_layers, self.kv_head_num, self.head_dims = kv_move_buffer.shape
        self.dtype_byte_size = kv_move_buffer.element_size()
        self.page_len = self.page_size * self.num_layers * self.kv_head_num * self.head_dims * self.dtype_byte_size
        self.page_reg_desc = self.nixl_agent.register_memory(kv_move_buffer)
        self.page_local_xfer_handles = self._create_paged_xfer_handles(self.page_reg_desc, self.num_pages)

    def _create_paged_xfer_handles(self, reg_desc: "nixlBind.nixlRegDList", page_num: int, agent_name: str = ""):
        base_addr, _, device_id, _ = reg_desc[0]
        pages_data = []
        for page_id in range(page_num):
            pages_data.append((base_addr + page_id * self.page_len, self.page_len, device_id))
        descs = self.nixl_agent.get_xfer_descs(pages_data, "VRAM")
        return self.nixl_agent.prep_xfer_dlist(agent_name, descs, "VRAM")

    def connect_add_remote_agent(self, remote_agent: NixlAgentMetadata):
        if remote_agent.agent_name in self.remote_agents:
            return

        peer_name = self.nixl_agent.add_remote_agent(remote_agent.agent_metadata)
        if isinstance(peer_name, bytes):
            peer_name = peer_name.decode()

        assert (
            peer_name == remote_agent.agent_name
        ), f"Peer name {peer_name} does not match remote name {remote_agent.agent_name}"

        page_mem_desc = self.nixl_agent.deserialize_descs(remote_agent.page_reg_desc)
        kv_page_xfer_handles = self._create_paged_xfer_handles(
            page_mem_desc, remote_agent.num_pages, agent_name=peer_name
        )
        remote_agent.page_xfer_handles = kv_page_xfer_handles

        logger.info(f"Added remote agent {peer_name} with mem desc {page_mem_desc}")

        self.remote_agents[remote_agent.agent_name] = remote_agent
        return

    def remove_remote_agent(self, peer_name: str):
        if peer_name in self.remote_agents:
            try:
                remote_agent: NixlAgentMetadata = self.remote_agents.pop(peer_name, None)
                assert remote_agent.agent_name == peer_name
                self.nixl_agent.remove_remote_agent(remote_agent.agent_name)
                if remote_agent.page_xfer_handles is not None:
                    self.nixl_agent.release_dlist_handle(remote_agent.page_xfer_handles)
            except BaseException as e:
                logger.error(f"remove remote agent {peer_name} failed")
                logger.exception(str(e))
        else:
            logger.warning(f"try to remove remote agent, but peer name {peer_name} agent did not exist")

    def send_readtask_to_decode_node(self, trans_task: NIXLChunckedTransTask):
        """
        prefill node call this function to send read task to decode node
        """
        decode_agent_name = trans_task.decode_agent_name
        if decode_agent_name not in self.remote_agents:
            logger.warning(f"decode_agent_name {decode_agent_name} not exist")
            _remote_agent = trans_task.create_decode_agent_obj()
            self.connect_add_remote_agent(_remote_agent)

        # 将页面读取任务发送给 decode 节点
        remote_agent: NixlAgentMetadata = self.remote_agents[decode_agent_name]
        assert trans_task.nixl_src_page_index is not None
        new_trans_task: NIXLChunckedTransTask = copy.copy(trans_task)

        new_trans_task.decode_agent_name = None
        new_trans_task.decode_agent_metadata = None
        new_trans_task.decode_num_pages = None
        new_trans_task.decode_page_reg_desc = None

        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc

        # 不需要传输细节的 mem_indexes 信息
        new_trans_task.mem_indexes = None
        self.nixl_agent.send_notif(
            remote_agent.agent_name,
            pickle.dumps(new_trans_task),
        )
        return

    def send_notify_to_prefill_node(self, prefill_agent_name: str, notify: bytes):
        self.nixl_agent.send_notif(remote_agent_name=prefill_agent_name, notif_msg=notify)
        return

    def read_blocks_paged(
        self,
        trans_task: NIXLChunckedTransTask,
    ) -> int:
        """
        decode node call this function to read kv blocks from prefill node
        """
        prefill_agent_name = trans_task.prefill_agent_name
        if prefill_agent_name not in self.remote_agents:
            logger.warning(f"prefill_agent_name {prefill_agent_name} not exist")
            _remote_agent = trans_task.create_prefill_agent_obj()
            self.connect_add_remote_agent(_remote_agent)

        assert trans_task.nixl_src_page_index is not None and trans_task.nixl_dst_page_index is not None
        remote_agent: NixlAgentMetadata = self.remote_agents[prefill_agent_name]
        src_handle = remote_agent.page_xfer_handles
        dst_handle = self.page_local_xfer_handles
        notify_obj = NIXLChunckedTransTaskRet(
            request_id=trans_task.request_id,
            start_kv_index=trans_task.start_kv_index,
            end_kv_index=trans_task.end_kv_index,
            has_error=False,
            error_info=None,
            first_gen_token_id=None,
            first_gen_token_logprob=None,
        )
        handle = self.nixl_agent.make_prepped_xfer(
            "READ",
            dst_handle,
            [trans_task.nixl_dst_page_index],
            src_handle,
            [trans_task.nixl_src_page_index],
            pickle.dumps(notify_obj),
        )
        if not handle:
            raise RuntimeError(f"make_prepped_xfer failed for task: {trans_task.to_str()}")

        self.nixl_agent.transfer(handle)

        return handle

    def check_task_status(self, trans_task: NIXLChunckedTransTask) -> str:
        assert trans_task.xfer_handle is not None
        handle = trans_task.xfer_handle
        xfer_state = self.nixl_agent.check_xfer_state(handle)
        if xfer_state == "ERR":
            logger.warning(f"Transfer failed with trans task {trans_task.to_str()} for handle {handle}")
        return xfer_state

    def release_xfer_handle(self, handle):
        self.nixl_agent.release_xfer_handle(handle=handle)
        return

    def shutdown(self):
        self.nixl_agent.deregister_memory(self.page_reg_desc)
        self.nixl_agent.release_dlist_handle(self.page_local_xfer_handles)
        agent_names = list(self.remote_agents.keys())
        for agent_name in agent_names:
            self.remove_remote_agent(agent_name)
        return
