import pickle
import copy
import os
import threading
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
    from nixl._api import nixl_thread_sync_t

    # getXferStatus raises these binding exceptions for negative native status
    # codes; the Python wrapper's string ERR branch is not reached in that case.
    _NIXL_XFER_ERRORS = (
        nixlBind.nixlNotPostedError,
        nixlBind.nixlInvalidParamError,
        nixlBind.nixlBackendError,
        nixlBind.nixlNotFoundError,
        nixlBind.nixlMismatchError,
        nixlBind.nixlNotAllowedError,
        nixlBind.nixlRepostActiveError,
        nixlBind.nixlUnknownError,
        nixlBind.nixlNotSupportedError,
        nixlBind.nixlRemoteDisconnectError,
        nixlBind.nixlCancelledError,
    )

    logger.info("Nixl is available")
except ImportError:
    logger.warning("nixl is not installed, which is required for pd disagreggation!!!")
    NixlWrapper = None
    _NIXL_XFER_ERRORS = ()


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
        conf = nixl_agent_config(sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW)
        if self.capture_telemetry:
            conf.capture_telemetry = True
            logger.info("NIXL telemetry enabled")
        self.nixl_agent = NixlWrapper(self.agent_name, conf)
        self._register_kv_move_buffer(kv_move_buffer=kv_move_buffer)
        self._remote_agents_lock = threading.Lock()
        self.remote_agents: Dict[str, PDAgentMetadata] = {}
        self._write_contexts = {}
        # Serialize complete peer add/remove operations, including native NIXL
        # calls and descriptor cleanup, across worker threads (see GH-1470).
        self._remote_agents_lock = threading.Lock()
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
        with self._remote_agents_lock:
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
                f"Added remote agent {peer_name} with mem desc {page_mem_desc} "
                f"cost time: {time.time() - start_time} s"
            )

            self.remote_agents[remote_agent.agent_name] = remote_agent
        return

    def remove_remote_agent(self, peer_name: str):
        with self._remote_agents_lock:
            remote_agent: PDAgentMetadata = self.remote_agents.pop(peer_name, None)
            if remote_agent is not None:
                start_time = time.monotonic()
                try:
                    assert remote_agent.agent_name == peer_name
                    self.nixl_agent.remove_remote_agent(remote_agent.agent_name)
                    if remote_agent.page_xfer_handles is not None:
                        self.nixl_agent.release_dlist_handle(remote_agent.page_xfer_handles)
                    logger.info(f"Removed remote agent {peer_name}, cost time: {time.monotonic() - start_time:.6f} s")
                except BaseException as e:
                    logger.error(
                        f"remove remote agent {peer_name} failed, cost time: {time.monotonic() - start_time:.6f} s"
                    )
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
            # 控制消息失败不能撤销仍被在途 WRITE 使用的远端描述符；由隔离任务重试。
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
            new_trans_task.write_stage = "stopped" if trans_task.write_quiesced else "error"
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
            # 控制消息失败不能撤销仍被在途 WRITE 使用的远端描述符；由隔离任务重试。
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

        # transfer() 抛异常也可能已经提交部分操作，先保留 handle 和提交状态。
        trans_task.xfer_handle = handle
        trans_task.write_submitted = True
        self._write_contexts[handle] = (threading.get_ident(), src_handle, dst_handle)
        status = self.nixl_agent.transfer(handle)
        if status not in ("PROC", "DONE"):
            raise RuntimeError(f"NIXL WRITE submission returned {status}")

        return handle

    def start_write_fence(self, original: PDChunckedTransTask, fence: PDChunckedTransTask):
        """Drain earlier UCX WRITEs using their exact connection and issuing thread.

        The source and destination staging pages remain quarantined throughout.
        A fresh WRITE includes UCX endpoint flush, so its DONE can prove older
        operations on that endpoint quiescent even when their handles report ERR.
        Never reconnect here: flushing a replacement endpoint proves nothing
        about the old endpoint. Reuse the original prepared descriptors instead.
        """
        thread_id, src_handle, dst_handle = self._write_contexts[original.xfer_handle]
        assert thread_id == threading.get_ident(), "WRITE fence must run on the original issuing thread"
        if self.nixl_agent.query_xfer_backend(original.xfer_handle) != "UCX":
            raise RuntimeError("WRITE recovery requires UCX endpoint flush semantics")
        handle = self.nixl_agent.make_prepped_xfer(
            "WRITE",
            src_handle,
            [original.src_page_index],
            dst_handle,
            [original.dst_page_index],
            b"",
            backends=["UCX"],
        )
        if not handle:
            raise RuntimeError("make_prepped_xfer failed for WRITE recovery fence")
        fence.xfer_handle = handle
        fence.write_submitted = True
        return self.nixl_agent.transfer(handle)

    def check_task_status(self, trans_task: PDChunckedTransTask) -> str:
        assert trans_task.xfer_handle is not None
        handle = trans_task.xfer_handle
        try:
            xfer_state = self.nixl_agent.check_xfer_state(handle)
        except _NIXL_XFER_ERRORS as e:
            logger.error(f"Native NIXL transfer failed with trans task {trans_task.to_str()} for handle {handle}")
            logger.exception(str(e))
            return "ERR"
        except Exception as e:
            logger.error(f"Check transfer state failed with trans task {trans_task.to_str()} for handle {handle}")
            logger.exception(str(e))
            return "UNKNOWN"
        if xfer_state == "ERR":
            logger.warning(f"Transfer failed with trans task {trans_task.to_str()} for handle {handle}")
        return xfer_state

    def release_xfer_handle(self, handle):
        self.nixl_agent.release_xfer_handle(handle=handle)
        self._write_contexts.pop(handle, None)
        return

    def shutdown(self):
        self.nixl_agent.deregister_memory(self.page_reg_desc)
        self.nixl_agent.release_dlist_handle(self.page_local_xfer_handles)
        agent_names = list(self.remote_agents.keys())
        for agent_name in agent_names:
            self.remove_remote_agent(agent_name)
        return
