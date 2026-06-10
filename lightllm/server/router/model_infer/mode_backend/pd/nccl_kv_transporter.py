import copy
import errno
import queue
import pickle
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import rpyc
import torch
from torch import Tensor
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer

from lightllm.distributed.pynccl import PyNcclCommunicator, StatelessP2PProcessGroup
from lightllm.server.pd_io_struct import PDChunckedTransTask, PDAgentMetadata
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_hostname_ip

logger = init_logger(__name__)


@dataclass
class NcclAgentMetadata:
    agent_name: str
    host_ip: str
    control_port: int
    device_id: int


class NcclKVTransporter:
    """
    PD KV transporter backed by NCCL point-to-point operations.

    NCCL does not provide remote notifications or one-sided WRITE, so this class
    uses a small RPyC control channel for notifications and communicator bootstrap
    while preserving the same request/ready/done/error interface used by pd
    trans-process management.
    """

    def __init__(
        self,
        node_id: int,
        tp_idx: int,
        kv_move_buffer: Tensor,
        host_ip: Optional[str] = None,
        control_port_min: int = 20000,
        control_port_max: int = 30000,
    ):
        self.node_id = node_id
        self.tp_idx = tp_idx
        self.kv_move_buffer = kv_move_buffer
        args = get_env_start_args()
        assert args.run_mode in ["prefill", "decode"], args.run_mode
        self.is_prefill_node = args.run_mode == "prefill"
        self.capture_telemetry = False
        self.num_pages, self.page_size, self.num_layers, self.kv_head_num, self.head_dims = kv_move_buffer.shape

        self.host_ip = host_ip or get_hostname_ip()
        assert self.host_ip is not None, "can not get host ip for NcclKVTransporter"

        self.control_channel = _NcclControlChannel(
            host_ip=self.host_ip,
            port_min=control_port_min,
            port_max=control_port_max,
        )
        self.remote_agents: Dict[str, PDAgentMetadata] = {}
        self._peers: Dict[str, "_NcclPeer"] = {}
        self._peer_lock = threading.Lock()
        return

    @property
    def agent_name(self) -> str:
        return f"{self.node_id}_{self.tp_idx}"

    @property
    def agent_metadata(self) -> bytes:
        return pickle.dumps(
            NcclAgentMetadata(
                agent_name=self.agent_name,
                host_ip=self.host_ip,
                control_port=self.control_channel.port,
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
        for notify in self.control_channel.get_notifs():
            notifs.setdefault(self._get_notify_source_agent_name(notify), []).append(notify)
        return notifs

    def connect_add_remote_agent(self, remote_agent: PDAgentMetadata):
        if remote_agent.agent_name in self.remote_agents:
            return

        metadata: NcclAgentMetadata = pickle.loads(remote_agent.agent_metadata)
        assert (
            metadata.agent_name == remote_agent.agent_name
        ), f"Peer name {metadata.agent_name} does not match remote name {remote_agent.agent_name}"

        self.remote_agents[remote_agent.agent_name] = remote_agent
        logger.info(f"Added NCCL remote agent {remote_agent.agent_name} at {metadata.host_ip}:{metadata.control_port}")
        return

    def remove_remote_agent(self, peer_name: str):
        if peer_name in self.remote_agents:
            self.remote_agents.pop(peer_name, None)
            with self._peer_lock:
                peer = self._peers.pop(peer_name, None)
            if peer is not None:
                peer.close()
        else:
            logger.warning(f"try to remove remote agent, but peer name {peer_name} agent did not exist")
        return

    def send_write_done_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.write_stage = "done"
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)
        return

    def send_write_request_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.write_stage = "request"
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)
        return

    def send_write_ready_task_to_prefill_node(self, trans_task: PDChunckedTransTask):
        if trans_task.prefill_agent_name not in self.remote_agents:
            self.connect_add_remote_agent(trans_task.create_prefill_agent_obj())

        self._get_peer(trans_task.prefill_agent_name).start_recv(trans_task)

        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.write_stage = "ready"
        new_trans_task.decode_agent_name = self.agent_name
        new_trans_task.decode_agent_metadata = self.agent_metadata
        new_trans_task.decode_num_pages = self.num_pages
        new_trans_task.decode_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.prefill_agent_name, new_trans_task)
        return

    def send_error_info_to_prefill_node(self, trans_task: PDChunckedTransTask):
        if trans_task.prefill_agent_name is None:
            return
        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.write_stage = "error"
        new_trans_task.decode_agent_name = self.agent_name
        new_trans_task.decode_agent_metadata = self.agent_metadata
        new_trans_task.decode_num_pages = self.num_pages
        new_trans_task.decode_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.prefill_agent_name, new_trans_task)
        return

    def send_error_info_to_decode_node(self, trans_task: PDChunckedTransTask):
        new_trans_task = self._copy_notify_task(trans_task)
        new_trans_task.write_stage = "error"
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)
        return

    def write_blocks_paged(self, trans_task: PDChunckedTransTask) -> "_NcclXferHandle":
        assert trans_task.src_page_index is not None and trans_task.dst_page_index is not None
        decode_agent_name = trans_task.decode_agent_name
        if decode_agent_name not in self.remote_agents:
            self.connect_add_remote_agent(trans_task.create_decode_agent_obj())

        return self._get_peer(decode_agent_name).send_page(trans_task)

    def check_task_status(self, trans_task: PDChunckedTransTask) -> str:
        assert trans_task.xfer_handle is not None
        return trans_task.xfer_handle.check_status()

    def release_xfer_handle(self, handle):
        return

    def shutdown(self):
        with self._peer_lock:
            peers = list(self._peers.values())
            self._peers.clear()
        for peer in peers:
            peer.close()
        self.remote_agents.clear()
        self.control_channel.close()
        return

    def _get_peer(self, peer_name: str) -> "_NcclPeer":
        with self._peer_lock:
            peer = self._peers.get(peer_name)
            if peer is None:
                peer = _NcclPeer(self, peer_name)
                self._peers[peer_name] = peer
            return peer

    def _send_task_notif(self, remote_agent_name: str, trans_task: PDChunckedTransTask):
        if remote_agent_name not in self.remote_agents:
            if remote_agent_name == trans_task.decode_agent_name:
                self.connect_add_remote_agent(trans_task.create_decode_agent_obj())
            else:
                self.connect_add_remote_agent(trans_task.create_prefill_agent_obj())

        remote_metadata = self._get_remote_metadata(remote_agent_name)
        self.control_channel.send_notif(
            remote_agent_name,
            remote_metadata.host_ip,
            remote_metadata.control_port,
            pickle.dumps(trans_task),
        )
        return

    def _get_remote_metadata(self, remote_agent_name: str) -> NcclAgentMetadata:
        remote_agent = self.remote_agents[remote_agent_name]
        return pickle.loads(remote_agent.agent_metadata)

    def _copy_notify_task(self, trans_task: PDChunckedTransTask) -> PDChunckedTransTask:
        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        return new_trans_task

    def _get_notify_source_agent_name(self, notify: bytes) -> str:
        notify_obj = pickle.loads(notify)
        assert isinstance(notify_obj, PDChunckedTransTask), type(notify_obj)

        if notify_obj.error_info is not None:
            if self.is_prefill_node:
                assert notify_obj.decode_agent_name is not None
                return notify_obj.decode_agent_name
            else:
                assert notify_obj.prefill_agent_name is not None
                return notify_obj.prefill_agent_name

        if notify_obj.write_stage == "request":
            assert notify_obj.prefill_agent_name is not None
            return notify_obj.prefill_agent_name

        if notify_obj.write_stage in ["ready", "done"]:
            assert notify_obj.decode_agent_name is not None
            return notify_obj.decode_agent_name

        raise AssertionError(f"unexpected notify stage: {notify_obj.write_stage}")


@dataclass
class _NcclXferHandle:
    peer_name: str
    event: torch.cuda.Event
    status: str = "PROC"
    error_info: Optional[str] = None

    def check_status(self) -> str:
        if self.status != "PROC":
            return self.status

        try:
            if self.event.query():
                self.status = "DONE"
        except BaseException as e:
            self.status = "ERR"
            self.error_info = str(e)
        return self.status


class _NcclPeer:
    def __init__(self, transporter: NcclKVTransporter, peer_name: str):
        self.transporter = transporter
        self.peer_name = peer_name
        self.comm: Optional[PyNcclCommunicator] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.recv_queue: Optional["queue.Queue[Optional[PDChunckedTransTask]]"] = None
        self._lock = threading.Lock()

    def send_page(self, trans_task: PDChunckedTransTask) -> _NcclXferHandle:
        assert trans_task.src_page_index is not None and trans_task.dst_page_index is not None
        page_tensor = self.transporter.kv_move_buffer[trans_task.src_page_index]
        comm = self._ensure_comm(is_server=True)
        stream = self._get_stream()

        comm.send(page_tensor, dst=1, stream=stream)
        event = torch.cuda.Event()
        event.record(stream)

        logger.info(
            f"NCCL send page posted request_id={trans_task.request_id} "
            f"src_page={trans_task.src_page_index} dst_agent={self.peer_name}"
        )
        return _NcclXferHandle(peer_name=self.peer_name, event=event)

    def start_recv(self, trans_task: PDChunckedTransTask):
        self._get_recv_queue().put(copy.copy(trans_task))
        return

    def close(self):
        with self._lock:
            recv_queue = self.recv_queue
            self.recv_queue = None
            comm = self.comm
            self.comm = None
            self.stream = None

        if recv_queue is not None:
            recv_queue.put(None)
        if comm is not None:
            comm.destroy()
        return

    def _get_stream(self) -> torch.cuda.Stream:
        with self._lock:
            if self.stream is None:
                torch.cuda.set_device(self.transporter.tp_idx)
                self.stream = torch.cuda.Stream()
            return self.stream

    def _get_recv_queue(self) -> "queue.Queue[Optional[PDChunckedTransTask]]":
        with self._lock:
            if self.recv_queue is not None:
                return self.recv_queue

            self.recv_queue = queue.Queue()
            threading.Thread(target=self._recv_page_loop, args=(self.recv_queue,), daemon=True).start()
            return self.recv_queue

    def _recv_page_loop(self, recv_queue: "queue.Queue[Optional[PDChunckedTransTask]]"):
        torch.cuda.set_device(self.transporter.tp_idx)
        while True:
            trans_task = recv_queue.get()
            if trans_task is None:
                return
            self._recv_page(trans_task)

    def _recv_page(self, trans_task: PDChunckedTransTask):
        try:
            page_tensor = self.transporter.kv_move_buffer[trans_task.dst_page_index]
            comm = self._ensure_comm(is_server=False)
            stream = self._get_stream()
            comm.recv(page_tensor, src=0, stream=stream)
            logger.info(
                f"NCCL recv page done request_id={trans_task.request_id} " f"dst_page={trans_task.dst_page_index}"
            )
        except BaseException as e:
            trans_task.error_info = str(e)
            logger.exception(str(e))
            self._drop_comm()
            self.transporter.send_error_info_to_prefill_node(trans_task)
        return

    def _ensure_comm(self, is_server: bool) -> PyNcclCommunicator:
        with self._lock:
            if self.comm is not None:
                return self.comm

            if is_server:
                src_id = self.transporter.agent_name
                dest_id = self.peer_name
            else:
                src_id = self.peer_name
                dest_id = self.transporter.agent_name

            group = StatelessP2PProcessGroup.create(
                src_id=src_id,
                dest_id=dest_id,
                is_server=is_server,
                store=_NcclControlStore(self.transporter, self.peer_name),
            )
            self.comm = PyNcclCommunicator(group, self.transporter.tp_idx)
            logger.info(f"Created NCCL communicator with peer {self.peer_name}")
            return self.comm

    def _drop_comm(self):
        with self._lock:
            comm = self.comm
            self.comm = None

        if comm is not None:
            comm.destroy()
            logger.warning(f"Dropped NCCL communicator with peer {self.peer_name}")
        return


class _NcclControlService(rpyc.Service):
    def __init__(self, channel: "_NcclControlChannel"):
        super().__init__()
        self.channel = channel

    def exposed_push_notif(self, payload: bytes):
        payload = obtain(payload)
        self.channel.notif_queue.put(payload)
        return

    def exposed_set_value(self, key: str, value: bytes):
        key = obtain(key)
        value = obtain(value)
        self.channel.add_store_value(key, value)
        return


class _NcclControlChannel:
    def __init__(
        self,
        host_ip: str,
        port_min: int,
        port_max: int,
    ):
        self.notif_queue: "queue.Queue[bytes]" = queue.Queue()
        self._store_values: Dict[str, bytes] = {}
        self._store_cond = threading.Condition()
        self._conn_lock = threading.Lock()
        self._conns: Dict[tuple[str, str, int], rpyc.Connection] = {}
        self._server, self.port = self._start_server(host_ip, port_min, port_max)

    def _start_server(self, host_ip: str, port_min: int, port_max: int) -> tuple[ThreadedServer, int]:
        last_error = None
        for cur_port in range(port_min, port_max + 1):
            try:
                server = ThreadedServer(
                    _NcclControlService(self),
                    hostname=host_ip,
                    port=cur_port,
                    protocol_config={
                        "allow_pickle": True,
                        "allow_all_attrs": True,
                        "allow_getattr": True,
                        "allow_setattr": True,
                    },
                )
                threading.Thread(target=server.start, daemon=True).start()
                logger.info(f"NCCL RPyC control channel listen on {host_ip}:{cur_port}")
                return server, cur_port
            except OSError as e:
                last_error = e
                if e.errno == errno.EADDRINUSE:
                    logger.info(f"NCCL RPyC control port {host_ip}:{cur_port} is in use, try next port")
                else:
                    logger.warning(f"Create NCCL RPyC control channel on {host_ip}:{cur_port} failed: {e}")
        raise RuntimeError(f"can not allocate NCCL control port in [{port_min}, {port_max}]") from last_error

    def close(self):
        with self._conn_lock:
            for conn in self._conns.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()
        self._server.close()
        return

    def add_store_value(self, key: str, value: bytes):
        with self._store_cond:
            self._store_values[key] = value
            self._store_cond.notify_all()
        return

    def wait_store_value(self, key: str, timeout: float = 30.0) -> bytes:
        with self._store_cond:
            ok = self._store_cond.wait_for(lambda: key in self._store_values, timeout=timeout)
            if not ok:
                raise TimeoutError(f"wait timeout after {int(timeout * 1000)}ms, key: {key}")
            return self._store_values.pop(key)

    def get_notifs(self) -> List[bytes]:
        notifs = []
        while True:
            try:
                notifs.append(self.notif_queue.get_nowait())
            except queue.Empty:
                break
        return notifs

    def send_notif(self, peer_name: str, host_ip: str, port: int, payload: bytes):
        self._call(peer_name, host_ip, port, "push_notif", payload)
        return

    def send_store_value(self, peer_name: str, host_ip: str, port: int, key: str, value: bytes):
        self._call(peer_name, host_ip, port, "set_value", key, value)
        return

    def _call(self, peer_name: str, host_ip: str, port: int, method: str, *args):
        conn_key = (peer_name, host_ip, port)
        with self._conn_lock:
            conn = self._conns.get(conn_key)
            if conn is None:
                conn = rpyc.connect(
                    host_ip,
                    port,
                    config={
                        "allow_pickle": True,
                        "allow_all_attrs": True,
                        "allow_getattr": True,
                        "allow_setattr": True,
                    },
                )
                self._conns[conn_key] = conn
            try:
                getattr(conn.root, method)(*args)
            except Exception as e:
                self._conns.pop(conn_key, None)
                try:
                    conn.close()
                except Exception:
                    pass
                raise RuntimeError(f"NCCL control RPC {method} to {peer_name} failed") from e
        return


class _NcclControlStore:
    def __init__(self, transporter: "NcclKVTransporter", remote_agent_name: str):
        self.transporter = transporter
        self.remote_agent_name = remote_agent_name

    def set(self, key: str, value: bytes):
        remote_metadata = self.transporter._get_remote_metadata(self.remote_agent_name)
        self.transporter.control_channel.send_store_value(
            self.remote_agent_name,
            remote_metadata.host_ip,
            remote_metadata.control_port,
            self._send_key(key),
            bytes(value),
        )
        return

    def get(self, key: str) -> bytes:
        return self.transporter.control_channel.wait_store_value(self._recv_key(key))

    def _send_key(self, key: str) -> str:
        return f"{self.transporter.agent_name}->{self.remote_agent_name}:{key}"

    def _recv_key(self, key: str) -> str:
        return f"{self.remote_agent_name}->{self.transporter.agent_name}:{key}"
