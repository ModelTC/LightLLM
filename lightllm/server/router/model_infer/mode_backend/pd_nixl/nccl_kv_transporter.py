import copy
import pickle
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.distributed import TCPStore

from lightllm.distributed.pynccl import PyNcclCommunicator, StatelessP2PProcessGroup
from lightllm.server.pd_io_struct import NIXLChunckedTransTask, NixlAgentMetadata
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_hostname_ip

logger = init_logger(__name__)


@dataclass
class NcclAgentMetadata:
    agent_name: str
    host_ip: str
    store_port: int
    device_id: int


@dataclass
class _NcclXferHandle:
    thread: Optional[threading.Thread]
    status: str = "PROC"
    error_info: Optional[str] = None


class _PeerSeqTurn:
    def __init__(self, transporter: "NcclKVTransporter", peer_name: str, seq: int):
        self.transporter = transporter
        self.peer_name = peer_name
        self.seq = seq

    def __enter__(self):
        with self.transporter._peer_seq_cond:
            while self.transporter._peer_seq_to_run.get(self.peer_name, 0) != self.seq:
                self.transporter._peer_seq_cond.wait()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self.transporter._peer_seq_cond:
            self.transporter._peer_seq_to_run[self.peer_name] = self.seq + 1
            self.transporter._peer_seq_cond.notify_all()
        return False


class NcclKVTransporter:
    """
    NIXL-compatible transporter backed by NCCL point-to-point operations.

    NIXL provides remote notifications and one-sided WRITE. NCCL does not, so this
    class uses a small TCPStore control plane for notifications and communicator
    bootstrap while preserving the same request/ready/done/error interface used by
    pd_nixl trans-process management.
    """

    def __init__(
        self,
        node_id: int,
        tp_idx: int,
        kv_move_buffer: Tensor,
        host_ip: Optional[str] = None,
        store_port: Optional[int] = None,
        store_port_min: int = 20000,
        store_port_max: int = 30000,
    ):
        self.node_id = node_id
        self.tp_idx = tp_idx
        self.kv_move_buffer = kv_move_buffer
        self.capture_telemetry = False
        self.num_pages, self.page_size, self.num_layers, self.kv_head_num, self.head_dims = kv_move_buffer.shape

        self.host_ip = host_ip or get_hostname_ip()
        assert self.host_ip is not None, "can not get host ip for NcclKVTransporter"

        self.store, self.store_port = self._create_local_store(
            store_port=store_port,
            store_port_min=store_port_min,
            store_port_max=store_port_max,
        )
        self.remote_agents: Dict[str, NixlAgentMetadata] = {}
        self.remote_stores: Dict[str, TCPStore] = {}
        self._comms: Dict[str, PyNcclCommunicator] = {}
        self._comm_create_lock = threading.Lock()
        self._peer_seq_cond = threading.Condition()
        self._peer_seq_to_assign: Dict[str, int] = {}
        self._peer_seq_to_run: Dict[str, int] = {}
        self._recv_notif_counter = 0
        self._deferred_notifs: List[bytes] = []
        self._recv_task_status: Dict[str, _NcclXferHandle] = {}
        self._xfer_handle_counter = 0
        self._xfer_handles: Dict[int, _NcclXferHandle] = {}
        return

    def _create_local_store(
        self, store_port: Optional[int], store_port_min: int, store_port_max: int
    ) -> tuple[TCPStore, int]:
        if store_port is not None:
            ports = [store_port]
        else:
            ports = list(range(store_port_min, store_port_max + 1))

        last_error = None
        for port in ports:
            try:
                store = TCPStore(
                    host_name=self.host_ip,
                    port=port,
                    is_master=True,
                    use_libuv=True,
                    timeout=timedelta(seconds=30),
                )
                return store, port
            except BaseException as e:
                last_error = e
                logger.warning(f"Create NCCL TCPStore on {self.host_ip}:{port} failed: {e}")

        raise RuntimeError(
            f"can not allocate NCCL TCPStore port in [{store_port_min}, {store_port_max}]"
        ) from last_error

    @property
    def agent_name(self) -> str:
        return f"{self.node_id}_{self.tp_idx}"

    @property
    def agent_metadata(self) -> bytes:
        return pickle.dumps(
            NcclAgentMetadata(
                agent_name=self.agent_name,
                host_ip=self.host_ip,
                store_port=self.store_port,
                device_id=self.tp_idx,
            )
        )

    @property
    def local_page_mem_desc(self) -> bytes:
        return pickle.dumps(
            {
                "num_pages": self.num_pages,
                "page_size": self.page_size,
                "num_layers": self.num_layers,
                "kv_head_num": self.kv_head_num,
                "head_dims": self.head_dims,
                "dtype": str(self.kv_move_buffer.dtype),
            }
        )

    def get_new_notifs(self) -> Dict[str, List[bytes]]:
        notifs: Dict[str, List[bytes]] = {}
        still_deferred = []
        for notify in self._deferred_notifs:
            ready_notify = self._get_ready_notify(notify)
            if ready_notify is None:
                still_deferred.append(notify)
            else:
                notifs.setdefault(self._get_notify_source_agent_name(ready_notify), []).append(ready_notify)
        self._deferred_notifs = still_deferred

        while True:
            key = self._notif_key(self.agent_name, self._recv_notif_counter)
            if not self.store.check([key]):
                break
            notify = bytes(self.store.get(key))
            ready_notify = self._get_ready_notify(notify)
            if ready_notify is None:
                self._deferred_notifs.append(notify)
            else:
                notifs.setdefault(self._get_notify_source_agent_name(ready_notify), []).append(ready_notify)
            self._recv_notif_counter += 1
        return notifs

    def connect_add_remote_agent(self, remote_agent: NixlAgentMetadata):
        if remote_agent.agent_name in self.remote_agents:
            return

        metadata: NcclAgentMetadata = pickle.loads(remote_agent.agent_metadata)
        assert (
            metadata.agent_name == remote_agent.agent_name
        ), f"Peer name {metadata.agent_name} does not match remote name {remote_agent.agent_name}"

        self.remote_agents[remote_agent.agent_name] = remote_agent
        self.remote_stores[remote_agent.agent_name] = TCPStore(
            host_name=metadata.host_ip,
            port=metadata.store_port,
            is_master=False,
            use_libuv=True,
            timeout=timedelta(seconds=30),
        )
        logger.info(f"Added NCCL remote agent {remote_agent.agent_name} at {metadata.host_ip}:{metadata.store_port}")
        return

    def remove_remote_agent(self, peer_name: str):
        if peer_name in self.remote_agents:
            self.remote_agents.pop(peer_name, None)
            self.remote_stores.pop(peer_name, None)
            comm = self._comms.pop(peer_name, None)
            if comm is not None:
                comm.destroy()
            with self._peer_seq_cond:
                self._peer_seq_to_assign.pop(peer_name, None)
                self._peer_seq_to_run.pop(peer_name, None)
        else:
            logger.warning(f"try to remove remote agent, but peer name {peer_name} agent did not exist")
        return

    def send_write_done_task_to_decode_node(self, trans_task: NIXLChunckedTransTask):
        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.nixl_write_stage = "done"
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)
        return

    def send_write_request_task_to_decode_node(self, trans_task: NIXLChunckedTransTask):
        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.nixl_write_stage = "request"
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)
        return

    def send_write_ready_task_to_prefill_node(self, trans_task: NIXLChunckedTransTask):
        self._start_recv_task(trans_task)

        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.nixl_write_stage = "ready"
        new_trans_task.decode_agent_name = self.agent_name
        new_trans_task.decode_agent_metadata = self.agent_metadata
        new_trans_task.decode_num_pages = self.num_pages
        new_trans_task.decode_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.prefill_agent_name, new_trans_task)
        return

    def send_error_info_to_prefill_node(self, trans_task: NIXLChunckedTransTask):
        if trans_task.prefill_agent_name is None:
            return
        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.nixl_write_stage = "error"
        new_trans_task.decode_agent_name = self.agent_name
        new_trans_task.decode_agent_metadata = self.agent_metadata
        new_trans_task.decode_num_pages = self.num_pages
        new_trans_task.decode_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.prefill_agent_name, new_trans_task)
        return

    def send_error_info_to_decode_node(self, trans_task: NIXLChunckedTransTask):
        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.nixl_write_stage = "error"
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)
        return

    def write_blocks_paged(self, trans_task: NIXLChunckedTransTask) -> int:
        assert trans_task.nixl_src_page_index is not None and trans_task.nixl_dst_page_index is not None
        decode_agent_name = trans_task.decode_agent_name
        if decode_agent_name not in self.remote_agents:
            self.connect_add_remote_agent(trans_task.create_decode_agent_obj())

        self._ensure_comm(
            remote_agent_name=decode_agent_name,
            is_server=True,
            store=self.store,
        )
        handle = self._next_xfer_handle()
        seq = self._assign_peer_seq(decode_agent_name)
        xfer_handle = _NcclXferHandle(
            thread=threading.Thread(target=self._send_page_task, args=(handle, trans_task, seq), daemon=True)
        )
        self._xfer_handles[handle] = xfer_handle
        xfer_handle.thread.start()
        return handle

    def check_task_status(self, trans_task: NIXLChunckedTransTask) -> str:
        assert trans_task.xfer_handle is not None
        handle = self._xfer_handles[trans_task.xfer_handle]
        if handle.status == "ERR":
            logger.warning(f"Transfer failed with trans task {trans_task.to_str()}: {handle.error_info}")
        return handle.status

    def release_xfer_handle(self, handle):
        xfer_handle = self._xfer_handles.pop(handle, None)
        if xfer_handle is not None:
            xfer_handle.thread.join(timeout=1)
        return

    def shutdown(self):
        for handle in list(self._xfer_handles.keys()):
            self.release_xfer_handle(handle)
        for comm in list(self._comms.values()):
            comm.destroy()
        self._comms.clear()
        self.remote_agents.clear()
        self.remote_stores.clear()
        return

    def _start_recv_task(self, trans_task: NIXLChunckedTransTask):
        if trans_task.prefill_agent_name not in self.remote_agents:
            self.connect_add_remote_agent(trans_task.create_prefill_agent_obj())
        self._recv_task_status[trans_task.get_key()] = _NcclXferHandle(thread=None)
        seq = self._assign_peer_seq(trans_task.prefill_agent_name)
        threading.Thread(target=self._recv_page_task, args=(copy.copy(trans_task), seq), daemon=True).start()
        return

    def _send_page_task(self, handle: int, trans_task: NIXLChunckedTransTask, seq: int):
        xfer_handle = self._xfer_handles[handle]
        try:
            remote_agent = self.remote_agents[trans_task.decode_agent_name]
            remote_metadata: NcclAgentMetadata = pickle.loads(remote_agent.agent_metadata)
            page_tensor = self.kv_move_buffer[trans_task.nixl_src_page_index]
            comm = self._get_cached_comm(trans_task.decode_agent_name)
            with self._peer_seq_turn(trans_task.decode_agent_name, seq):
                comm.send(page_tensor, dst=1)
                torch.cuda.current_stream().synchronize()
            xfer_handle.status = "DONE"
            logger.info(
                f"NCCL send page done request_id={trans_task.request_id} "
                f"src_page={trans_task.nixl_src_page_index} dst_agent={remote_metadata.agent_name}"
            )
        except BaseException as e:
            xfer_handle.status = "ERR"
            xfer_handle.error_info = str(e)
            logger.exception(str(e))
            self._drop_comm(trans_task.decode_agent_name)
        return

    def _recv_page_task(self, trans_task: NIXLChunckedTransTask, seq: int):
        try:
            page_tensor = self.kv_move_buffer[trans_task.nixl_dst_page_index]
            remote_agent = self.remote_agents[trans_task.prefill_agent_name]
            remote_store = self.remote_stores[remote_agent.agent_name]
            comm = self._ensure_comm(
                remote_agent_name=trans_task.prefill_agent_name,
                is_server=False,
                store=remote_store,
            )
            with self._peer_seq_turn(trans_task.prefill_agent_name, seq):
                comm.recv(page_tensor, src=0)
                torch.cuda.current_stream().synchronize()
            self._recv_task_status[trans_task.get_key()].status = "DONE"
            logger.info(
                f"NCCL recv page done request_id={trans_task.request_id} "
                f"dst_page={trans_task.nixl_dst_page_index}"
            )
        except BaseException as e:
            trans_task.error_info = str(e)
            recv_status = self._recv_task_status.get(trans_task.get_key(), None)
            if recv_status is not None:
                recv_status.status = "ERR"
                recv_status.error_info = str(e)
            logger.exception(str(e))
            self._drop_comm(trans_task.prefill_agent_name)
            self.send_error_info_to_prefill_node(trans_task)
        return

    def _get_ready_notify(self, notify: bytes) -> Optional[bytes]:
        try:
            notify_obj = pickle.loads(notify)
        except BaseException:
            return notify

        if not isinstance(notify_obj, NIXLChunckedTransTask):
            return notify

        if notify_obj.nixl_write_stage != "done":
            return notify

        recv_status = self._recv_task_status.get(notify_obj.get_key(), None)
        if recv_status is None or recv_status.status == "PROC":
            return None

        self._recv_task_status.pop(notify_obj.get_key(), None)
        if recv_status.status == "ERR":
            notify_obj.error_info = recv_status.error_info or "nccl recv failed"
            return pickle.dumps(notify_obj)

        return notify

    def _get_cached_comm(self, remote_agent_name: str) -> PyNcclCommunicator:
        comm = self._comms.get(remote_agent_name)
        if comm is None:
            raise RuntimeError(f"NCCL communicator with peer {remote_agent_name} is not initialized")
        return comm

    def _ensure_comm(
        self,
        remote_agent_name: str,
        is_server: bool,
        store: TCPStore,
    ) -> PyNcclCommunicator:
        comm = self._comms.get(remote_agent_name)
        if comm is not None:
            return comm

        with self._comm_create_lock:
            comm = self._comms.get(remote_agent_name)
            if comm is not None:
                return comm

            if is_server:
                src_id = self.agent_name
                dest_id = remote_agent_name
            else:
                src_id = remote_agent_name
                dest_id = self.agent_name

            group = StatelessP2PProcessGroup.create(
                src_id=src_id,
                dest_id=dest_id,
                is_server=is_server,
                store=store,
            )
            comm = PyNcclCommunicator(group, self.tp_idx)
            self._comms[remote_agent_name] = comm
            logger.info(f"Created NCCL communicator with peer {remote_agent_name}")
            return comm

    def _drop_comm(self, remote_agent_name: str):
        with self._comm_create_lock:
            comm = self._comms.pop(remote_agent_name, None)
            if comm is not None:
                comm.destroy()
                logger.warning(f"Dropped NCCL communicator with peer {remote_agent_name}")
        return

    def _assign_peer_seq(self, peer_name: str) -> int:
        with self._peer_seq_cond:
            seq = self._peer_seq_to_assign.get(peer_name, 0)
            self._peer_seq_to_assign[peer_name] = seq + 1
            self._peer_seq_to_run.setdefault(peer_name, 0)
            return seq

    def _peer_seq_turn(self, peer_name: str, seq: int):
        return _PeerSeqTurn(self, peer_name, seq)

    def _send_task_notif(self, remote_agent_name: str, trans_task: NIXLChunckedTransTask):
        if remote_agent_name not in self.remote_agents:
            if remote_agent_name == trans_task.decode_agent_name:
                self.connect_add_remote_agent(trans_task.create_decode_agent_obj())
            else:
                self.connect_add_remote_agent(trans_task.create_prefill_agent_obj())

        remote_store = self.remote_stores[remote_agent_name]
        counter = remote_store.add(f"notif/{remote_agent_name}/counter", 1) - 1
        remote_store.set(self._notif_key(remote_agent_name, counter), pickle.dumps(trans_task))
        return

    def _copy_notify_task(self, trans_task: NIXLChunckedTransTask) -> NIXLChunckedTransTask:
        new_trans_task: NIXLChunckedTransTask = copy.copy(trans_task)
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        return new_trans_task

    def _next_xfer_handle(self):
        self._xfer_handle_counter += 1
        return self._xfer_handle_counter

    @staticmethod
    def _notif_key(agent_name: str, counter: int) -> str:
        return f"notif/{agent_name}/{counter}"

    @staticmethod
    def _get_notify_source_agent_name(notify: bytes) -> str:
        try:
            notify_obj = pickle.loads(notify)
        except BaseException:
            return "unknown"

        if not isinstance(notify_obj, NIXLChunckedTransTask):
            return "unknown"

        if notify_obj.nixl_write_stage == "request":
            return notify_obj.prefill_agent_name or "unknown"
        if notify_obj.nixl_write_stage in ["ready", "done"]:
            return notify_obj.decode_agent_name or "unknown"
        return notify_obj.prefill_agent_name or notify_obj.decode_agent_name or "unknown"
