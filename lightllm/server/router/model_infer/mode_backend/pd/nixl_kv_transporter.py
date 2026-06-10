import pickle
import copy
import os
import time
from dataclasses import dataclass
from typing import Dict
from torch import Tensor
from lightllm.server.pd_io_struct import PDChunckedTransTask, PDAgentMetadata
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

try:
    from nixl._api import nixl_agent as NixlWrapper
    from nixl._api import nixlBind
    from nixl._api import nixl_agent_config

    logger.info("Nixl is available")
except ImportError:
    logger.warning("nixl is not installed, which is required for pd disagreggation!!!")
    NixlWrapper = None


class NixlKVTransporter:
    def __init__(self, node_id: int, tp_idx: int, kv_move_buffer: Tensor):
        self.node_id = node_id
        self.tp_idx = tp_idx
        self.capture_telemetry = os.getenv("LIGHTLLM_NIXL_CAPTURE_TELEMETRY", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        conf = None
        if self.capture_telemetry:
            conf = nixl_agent_config()
            conf.capture_telemetry = True
            logger.info("NIXL telemetry enabled")
        self.nixl_agent = NixlWrapper(self.agent_name, conf)
        self._register_kv_move_buffer(kv_move_buffer=kv_move_buffer)
        self.remote_agents: Dict[str, PDAgentMetadata] = {}
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

    def connect_add_remote_agent(self, remote_agent: PDAgentMetadata):
        if remote_agent.agent_name in self.remote_agents:
            return

        start_time = time.time()

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

        logger.info(
            f"Added remote agent {peer_name} with mem desc {page_mem_desc} cost time: {time.time() - start_time} s"
        )

        self.remote_agents[remote_agent.agent_name] = remote_agent
        return

    def remove_remote_agent(self, peer_name: str):
        if peer_name in self.remote_agents:
            try:
                remote_agent: PDAgentMetadata = self.remote_agents.pop(peer_name, None)
                assert remote_agent.agent_name == peer_name
                self.nixl_agent.remove_remote_agent(remote_agent.agent_name)
                if remote_agent.page_xfer_handles is not None:
                    self.nixl_agent.release_dlist_handle(remote_agent.page_xfer_handles)
            except BaseException as e:
                logger.error(f"remove remote agent {peer_name} failed")
                logger.exception(str(e))
        else:
            logger.warning(f"try to remove remote agent, but peer name {peer_name} agent did not exist")

    def send_write_done_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        decode_agent_name = trans_task.decode_agent_name
        if decode_agent_name not in self.remote_agents:
            logger.warning(f"decode_agent_name {decode_agent_name} not exist")
            _remote_agent = trans_task.create_decode_agent_obj()
            self.connect_add_remote_agent(_remote_agent)

        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.write_stage = "done"
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        new_trans_task.decode_agent_metadata = None
        new_trans_task.decode_page_reg_desc = None
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
        self.nixl_agent.send_notif(
            remote_agent_name=decode_agent_name,
            notif_msg=pickle.dumps(new_trans_task),
        )
        return

    def send_write_request_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        decode_agent_name = trans_task.decode_agent_name
        if decode_agent_name not in self.remote_agents:
            logger.warning(f"decode_agent_name {decode_agent_name} not exist")
            _remote_agent = trans_task.create_decode_agent_obj()
            self.connect_add_remote_agent(_remote_agent)

        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.write_stage = "request"
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
        self.nixl_agent.send_notif(
            remote_agent_name=decode_agent_name,
            notif_msg=pickle.dumps(new_trans_task),
        )
        return

    def send_write_ready_task_to_prefill_node(self, trans_task: PDChunckedTransTask):
        prefill_agent_name = trans_task.prefill_agent_name
        if prefill_agent_name not in self.remote_agents:
            logger.warning(f"prefill_agent_name {prefill_agent_name} not exist")
            _remote_agent = trans_task.create_prefill_agent_obj()
            self.connect_add_remote_agent(_remote_agent)

        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.write_stage = "ready"
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        new_trans_task.decode_agent_name = self.agent_name
        new_trans_task.decode_agent_metadata = self.agent_metadata
        new_trans_task.decode_num_pages = self.num_pages
        new_trans_task.decode_page_reg_desc = self.local_page_mem_desc
        self.nixl_agent.send_notif(
            remote_agent_name=prefill_agent_name,
            notif_msg=pickle.dumps(new_trans_task),
        )
        return

    def send_error_info_to_prefill_node(self, trans_task: PDChunckedTransTask):
        # decode node 主动发送错误信息给 prefill node, 但是只有到达一定阶段的任务才有对端的信息
        # 才能发送
        if trans_task.prefill_agent_name is None:
            return

        try:
            prefill_agent_name = trans_task.prefill_agent_name
            if prefill_agent_name not in self.remote_agents:
                logger.warning(f"prefill_agent_name {prefill_agent_name} not exist")
                _remote_agent = trans_task.create_prefill_agent_obj()
                self.connect_add_remote_agent(_remote_agent)
            assert trans_task.error_info is not None
            new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
            new_trans_task.write_stage = "error"
            new_trans_task.mem_indexes = None
            new_trans_task.xfer_handle = None
            new_trans_task.decode_agent_name = self.agent_name
            new_trans_task.decode_agent_metadata = self.agent_metadata
            new_trans_task.decode_num_pages = self.num_pages
            new_trans_task.decode_page_reg_desc = self.local_page_mem_desc
            self.nixl_agent.send_notif(
                remote_agent_name=prefill_agent_name,
                notif_msg=pickle.dumps(new_trans_task),
            )
        except BaseException as e:
            logger.error(f"send error info to prefill node failed: {trans_task.to_str()}")
            logger.exception(str(e))
            self.remove_remote_agent(peer_name=prefill_agent_name)
        return

    def send_error_info_to_decode_node(self, trans_task: PDChunckedTransTask):
        try:
            decode_agent_name = trans_task.decode_agent_name
            if decode_agent_name not in self.remote_agents:
                logger.warning(f"decode_agent_name {decode_agent_name} not exist")
                _remote_agent = trans_task.create_decode_agent_obj()
                self.connect_add_remote_agent(_remote_agent)
            assert trans_task.error_info is not None
            new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
            new_trans_task.write_stage = "error"
            new_trans_task.mem_indexes = None
            new_trans_task.xfer_handle = None
            new_trans_task.prefill_agent_name = self.agent_name
            new_trans_task.prefill_agent_metadata = self.agent_metadata
            new_trans_task.prefill_num_pages = self.num_pages
            new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
            self.nixl_agent.send_notif(
                remote_agent_name=decode_agent_name,
                notif_msg=pickle.dumps(new_trans_task),
            )
        except BaseException as e:
            logger.error(f"send error info to decode node failed: {trans_task.to_str()}")
            logger.exception(str(e))
            self.remove_remote_agent(peer_name=decode_agent_name)
        return

    def write_blocks_paged(
        self,
        trans_task: PDChunckedTransTask,
    ) -> int:
        """
        prefill node call this function to write kv blocks into decode node pages
        """
        decode_agent_name = trans_task.decode_agent_name
        if decode_agent_name not in self.remote_agents:
            logger.warning(f"decode_agent_name {decode_agent_name} not exist")
            _remote_agent = trans_task.create_decode_agent_obj()
            self.connect_add_remote_agent(_remote_agent)

        assert trans_task.src_page_index is not None and trans_task.dst_page_index is not None
        remote_agent: PDAgentMetadata = self.remote_agents[decode_agent_name]
        src_handle = self.page_local_xfer_handles
        dst_handle = remote_agent.page_xfer_handles
        handle = self.nixl_agent.make_prepped_xfer(
            "WRITE",
            src_handle,
            [trans_task.src_page_index],
            dst_handle,
            [trans_task.dst_page_index],
            b"",
        )
        if not handle:
            raise RuntimeError(f"make_prepped_xfer failed for task: {trans_task.to_str()}")

        self.nixl_agent.transfer(handle)

        return handle

    def check_task_status(self, trans_task: PDChunckedTransTask) -> str:
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
