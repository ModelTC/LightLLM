import time
import torch.multiprocessing as mp
from typing import Dict, List
import queue
import numpy as np


from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs.req import PDChunkedPrefillReq
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq

from .nixl_kv_transporter import NixlMetadata, NixlKVTransporter
from .pd_remote_prefill_obj import (
    PrefillRequest,
    RemoteRequest,
    RemoteRequstType,
    ConnectRequest,
    KVMoveRequest,
    RemotePrefillStatus,
    ThreadSafeDict,
)

logger = init_logger(__name__)


class PDNIXLBackendBase(ModeBackend):
    _THEAD_WAIT_INTERVAL = 0.001

    def __init__(self, to_remote_queue: mp.Queue, from_remote_queue: mp.Queue, nixl_meta_queue: mp.Queue):
        super().__init__()
        self.to_remote_queue = to_remote_queue
        self.from_remote_queue = from_remote_queue
        self.nixl_meta_queue = nixl_meta_queue

        # for decode
        self.remote_prefilled_reqs: ThreadSafeDict = ThreadSafeDict()

        # for prefill
        self.remote_prefill_requests: ThreadSafeDict = ThreadSafeDict()
        self.inflght_transfer_requests: ThreadSafeDict = ThreadSafeDict()

    def init_custom(self):
        self.nixl_agent = NixlKVTransporter(self.args.pd_node_id, self.tp_rank)
        self.nixl_agent.register_kv_buffer(self.model.mem_manager.kv_buffer)
        self.nixl_meta_queue.put(
            (self.nixl_agent.agent_metadata, self.nixl_agent.num_tokens, self.nixl_agent.local_mem_desc)
        )

    def _prefill_wait_loop(self):
        while True:

            def handle_remote_prefill(req_status: RemotePrefillStatus):
                group_req_id = req_status.group_req_id
                status = req_status.status
                if status != 1:
                    logger.warning(f"remote prefill reqeust: {group_req_id} done with state: {status}")
                if run_req := self.remote_prefilled_reqs.get(group_req_id, None):
                    shm_req: PDChunkedPrefillReq = run_req.shm_req
                    shm_req.set_pd_req_rank_state(self.rank_in_dp, status)
                    self.remote_prefilled_reqs.pop(group_req_id)
                    if self.is_master_in_dp:
                        logger.info(
                            f"remote prefill reqeust: {group_req_id} done with status: {status} "
                            f"took: {time.time() - run_req.remote_prefill_start} seconds"
                        )
                else:
                    if self.is_master_in_dp:
                        logger.warning(f"remote prefill reqeust: {group_req_id} not found")

            # from local
            try:
                req_status = self.from_remote_queue.get_nowait()
                handle_remote_prefill(req_status)
            except queue.Empty:
                pass

            # from remote
            notifies = self.nixl_agent.get_new_notifs()
            for agent_name, req_statuses in notifies.items():
                for req_status in req_statuses:
                    prefill_status = RemotePrefillStatus.deserialize(req_status)
                    handle_remote_prefill(prefill_status)

            time.sleep(PDNIXLBackendBase._THEAD_WAIT_INTERVAL)

    def _wait_transfer_loop(self):
        while True:
            done_req_ids = self.nixl_agent.get_done_tranfers()

            for req_id, state in done_req_ids:
                if state != 1:
                    logger.info(f"wait transfer done: {req_id} state: {state}")

                if req_id not in self.inflght_transfer_requests:
                    if self.is_master_in_dp:
                        logger.warning(f"{req_id} not found in inflght_transfer_requests")
                    continue

                req: InferReq = self.inflght_transfer_requests[req_id]
                shm_req: PDChunkedPrefillReq = req.shm_req
                shm_req.set_pd_req_rank_state(self.rank_in_dp, state)
                del self.inflght_transfer_requests[req_id]
                if self.is_master_in_dp:
                    logger.info(
                        f"req: {req_id} kv transfer with state: {state} "
                        f"took: {time.time() - req.kv_transfer_start} seconds"
                    )
            time.sleep(PDNIXLBackendBase._THEAD_WAIT_INTERVAL)

    def _handle_prefill_loop(self):
        while True:
            request: RemoteRequest = self.from_remote_queue.get()
            if request.type == RemoteRequstType.REMOTE_CONNECT:
                request: ConnectRequest
                logger.info(f"connect request received from: {request.decode_id}")
                self.nixl_agent.add_remote_agent(
                    NixlMetadata(
                        id=request.decode_id,
                        num_tokens=request.num_tokens,
                        agent_metadatas=request.agent_metadatas,
                        agent_mem_descs=request.agent_mem_descs,
                    )
                )
                self.to_remote_queue.put("OK")

            if request.type == RemoteRequstType.REMOTE_PREFILL:
                request: PrefillRequest
                group_request_id = request.data.sampling_params.group_request_id
                logger.info(
                    f"prefill request received from decode: {request.decode_id} "
                    f"and group request id: {group_request_id}"
                )
                self.remote_prefill_requests[group_request_id] = request

    def _transfer_kv_to_remote(self, req: InferReq):
        group_req_id = req.shm_req.group_req_id
        # set state
        if group_req_id not in self.remote_prefill_requests:
            logger.info(f"remote prefill request {group_req_id} not found")
            return

        # kick off kv transfer
        if req.finish_status.is_finished():
            req.kv_transfer_start = time.time()
            kv_transfer_req = KVMoveRequest(
                group_req_id=group_req_id,
                token_ids=self.model.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].tolist(),
            )
            remote_request = self.remote_prefill_requests[group_req_id]
            self.nixl_agent.write_blocks(kv_transfer_req, remote_request)
            shm_req: PDChunkedPrefillReq = req.shm_req
            shm_req.set_pd_req_rank_state(self.rank_in_dp, 0)
            req.kv_transfering = True
            self.inflght_transfer_requests[group_req_id] = req

    def _decode_filter_reqs(
        self, prefill_reqs: List[InferReq], aborted_reqs: List[InferReq], decode_reqs: List[InferReq]
    ):
        new_prefill_reqs: List[InferReq] = []
        new_aborted_reqs: List[InferReq] = []
        remote_prefill_reqs: List[InferReq] = []

        # filter out aborted requests
        for req in aborted_reqs:
            if req.in_prefill_or_transfer:
                shm_req: PDChunkedPrefillReq = req.shm_req
                state = shm_req.get_pd_req_state()
                if state != 0:
                    new_aborted_reqs.append(req)
                    req.in_prefill_or_transfer = False
                else:
                    # TODO trigger remote abort
                    remote_prefill_reqs.append(req)

        for req in prefill_reqs:
            if req.in_prefill_or_transfer:
                shm_req: PDChunkedPrefillReq = req.shm_req
                # state is updated by router
                state = shm_req.get_pd_req_state()
                if state == 1:  # success
                    req.cur_kv_len = req.get_cur_total_len() - 1
                    decode_reqs.append(req)
                    req.in_prefill_or_transfer = False
                elif state == -1:  # failure
                    aborted_reqs.append(req)
                    req.in_prefill_or_transfer = False
                elif state == 0:  # in progress
                    remote_prefill_reqs.append(req)
                else:
                    logger.warning(f"remote prefill request {shm_req.group_req_id} unexpected state {state}")
                continue

            new_prefill_reqs.append(req)

        return new_prefill_reqs, new_aborted_reqs, decode_reqs, remote_prefill_reqs

    def _prefill_filter_reqs(self, ok_finished_reqs: List[InferReq], aborted_reqs: List[InferReq]):
        new_ok_finished_reqs = []
        kv_transfer_reqs = []

        for req in ok_finished_reqs:
            if req.in_prefill_or_transfer:
                shm_req: PDChunkedPrefillReq = req.shm_req
                state = shm_req.get_pd_req_state()
                if state == 1:  # success
                    new_ok_finished_reqs.append(req)
                    req.in_prefill_or_transfer = False
                elif state == -1:  # failure
                    aborted_reqs.append(req)
                    req.in_prefill_or_transfer = False
                elif state == 0:
                    kv_transfer_reqs.append(req)
                else:
                    logger.warning(f"remote prefill request {shm_req.group_req_id} unexpected state {state}")
                continue

            new_ok_finished_reqs.append(req)

        return new_ok_finished_reqs, aborted_reqs, kv_transfer_reqs

    def _prepare_remote_prefill_inputs(self, req_objs: List[InferReq]):
        run_reqs = []
        start_loc = 0
        input_ids = []
        nopad_b_req_idx = []
        nopad_b_start_loc = []
        nopad_b_seq_len = []

        for req in req_objs:
            run_reqs.append(req)
            nopad_b_req_idx.append(req.req_idx)
            nopad_b_start_loc.append(start_loc)

            input_token_ids = req.get_input_token_ids()
            seq_len = len(input_token_ids)
            input_token_len = seq_len - req.cur_kv_len
            input_id = input_token_ids[req.cur_kv_len :]
            nopad_b_seq_len.append(seq_len)
            input_ids.append(input_id)
            start_loc += input_token_len

        nopad_b_start_loc.append(start_loc)  # last request

        input_ids = np.concatenate(input_ids, dtype=np.int64)
        # g_infer_state_lock.acquire() # I don't think it's needed
        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
        mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0])
        # g_infer_state_lock.release()
        kwargs = {
            "batch_size": len(run_reqs),
            "input_ids": input_ids,
            "mem_indexes": mem_indexes.tolist(),
            "b_req_idx": nopad_b_req_idx,
            "b_start_loc": nopad_b_start_loc,
            "b_seq_len": nopad_b_seq_len,
        }
        return kwargs, run_reqs

    def _prefill_abort_remote(self, req_objs: List[InferReq]):
        for req_obj in req_objs:
            group_req_id = req_obj.shm_req.group_req_id
            if group_req_id in self.remote_prefill_requests:
                self.nixl_agent.send_abort_notify(self.remote_prefill_requests[group_req_id].decode_id, group_req_id)
                del self.remote_prefill_requests[group_req_id]
