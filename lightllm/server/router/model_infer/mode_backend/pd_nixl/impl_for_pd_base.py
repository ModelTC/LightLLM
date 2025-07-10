import time
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
import torch
from typing import Dict, List
import queue
import numpy as np
import asyncio
import pickle
import threading


from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs.req import PDNIXLChunkedPrefillReq
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
    TransferState,
    SafePageIndexScheduler,
    RemoteTransferType,
    RemoteTransferStatusType,
    PageTransferAck,
    NotificationType,
    Notification,
)

logger = init_logger(__name__)


class PDNIXLBackendBase(ModeBackend):
    _THREAD_WAIT_INTERVAL = 0.001

    def __init__(self, to_remote_queue: mp.Queue, from_remote_queue: mp.Queue, nixl_meta_queue: mp.Queue):
        super().__init__()
        self.to_remote_queue = to_remote_queue
        self.from_remote_queue = from_remote_queue
        self.nixl_meta_queue = nixl_meta_queue
        self.prefill_post_handle_queue = queue.Queue()

        # for decode
        self.remote_prefilled_reqs: ThreadSafeDict = ThreadSafeDict()
        self.request_to_page_ids: ThreadSafeDict = ThreadSafeDict()
        self.request_to_first_token: ThreadSafeDict = ThreadSafeDict()

        # for prefill
        self.remote_prefill_requests: ThreadSafeDict = ThreadSafeDict()
        self.inflght_transfer_requests: ThreadSafeDict = ThreadSafeDict()

    def init_custom(self):
        self.nixl_agent = NixlKVTransporter(self.args.pd_node_id, self.rank_in_node)
        self.nixl_agent.register_kv_buffer(self.model.mem_manager.kv_buffer)
        self.nixl_agent.register_kv_move_buffer(self.model.mem_manager.kv_move_buffer)
        self.page_scheduer = SafePageIndexScheduler(self.nixl_agent.num_pages)

        self.nixl_meta_queue.put(
            (self.nixl_agent.agent_metadata, self.nixl_agent.num_tokens, self.nixl_agent.num_pages,
             self.nixl_agent.local_mem_desc, self.nixl_agent.local_page_mem_desc)
        )

    def _start_async_loop(self, async_loop_func):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_loop_func())


    async def _handle_remote_prefill(self, req_status: RemotePrefillStatus):
        group_req_id = req_status.group_req_id
        status = req_status.status
        if status != RemoteTransferStatusType.SUCCESS:
            logger.warning(f"remote prefill reqeust: {group_req_id} done with state: {status}")

        ret = None
        if run_req := self.remote_prefilled_reqs.get(group_req_id, None):
            if req_status.transfer_type == RemoteTransferType.PAGE_TRANSFER and status == RemoteTransferStatusType.SUCCESS:
                kv_start, kv_len = req_status.kv_start, req_status.kv_len
                token_ids = g_infer_context.req_manager.req_to_token_indexs[run_req.req_idx][kv_start: kv_start + kv_len] # gpu tensor
                self.model.mem_manager.kv_buffer[:, token_ids, :, :] = self.model.mem_manager.kv_move_buffer[req_status.page_id][:kv_len].transpose(0, 1)
                ret = PageTransferAck(group_req_id=group_req_id, page_id=req_status.page_id)

            if req_status.is_last or status != RemoteTransferStatusType.SUCCESS:
                    shm_req: PDNIXLChunkedPrefillReq = run_req.shm_req
                    shm_req.set_pd_req_rank_state(self.rank_in_dp, status.value)
                    self.remote_prefilled_reqs.pop(group_req_id)
                    self.request_to_first_token[group_req_id] = (req_status.next_token_id, req_status.next_token_logprob)

                    if self.is_master_in_dp:
                        # return page ids
                        if group_req_id in self.request_to_page_ids:
                            self.page_scheduer.return_(self.request_to_page_ids[group_req_id])
                            del self.request_to_page_ids[group_req_id]

                        logger.info(
                            f"remote prefill reqeust: {group_req_id} done with status: {status} "
                            f"took: {time.time() - run_req.remote_prefill_start} seconds"
                        )
                    ret = None

        else:
            if self.is_master_in_dp:
                logger.warning(f"remote prefill reqeust: {group_req_id} not found")

        return ret

    async def _prefill_wait_loop_async(self):
        while True:
             # from local
            try:
                req_status = self.from_remote_queue.get_nowait()
                await self._handle_remote_prefill(req_status)
            except queue.Empty:
                pass

            # from remote
            notifies = self.nixl_agent.get_new_notifs()
            for agent_name, req_statuses in notifies.items():
                acks = []
                for req_statuses_bytes in req_statuses:
                    noti: Notification = Notification.from_bytes(req_statuses_bytes)
                    if noti.type == NotificationType.REMOTE_MD:
                        self.nixl_agent.connect_to_remote(agent_name, noti.data)
                    elif noti.type == NotificationType.TRANSFER_NOTIFY:
                        for req_status in noti.data:
                            prefill_status = RemotePrefillStatus.deserialize(req_status)
                            ack = await self._handle_remote_prefill(prefill_status)
                            if ack:
                                acks.append(ack)
                if len(acks) > 0:
                    # wait for copy done
                    torch.cuda.current_stream().synchronize()
                    logger.info(f"send {len(acks)} acks to {agent_name}")
                    self.nixl_agent.send_transfer_notify(agent_name, acks)

            await asyncio.sleep(PDNIXLBackendBase._THREAD_WAIT_INTERVAL)

    def _handle_chunked_transfer(self, req: InferReq, next_token_id: int=None, next_token_logprob: float=None):
        if next_token_id:
            next_token_id = int(next_token_id)
            next_token_logprob = float(next_token_logprob)

        shm_req: PDNIXLChunkedPrefillReq = req.shm_req
        group_req_id = shm_req.group_req_id
        if group_req_id not in self.remote_prefill_requests:
            logger.info(f"remote prefill request {group_req_id} not found")
            return

        remote_request: PrefillRequest = self.remote_prefill_requests[group_req_id]
        if remote_request.transfer_state is None:
            remote_request.transfer_state = TransferState(
                start_time=time.time(),
                current_chunk_id=0,
                transfered_kv_len=remote_request.data.local_cached_len,
                current_kv_len=req.cur_kv_len,
                is_finished=req.finish_status.is_finished(),
                token_index=self.model.req_manager.req_to_token_indexs[req.req_idx].tolist(),
                free_page_ids=remote_request.data.page_ids.copy(),
                next_token_id=next_token_id,
                next_token_logprob=next_token_logprob,
                lock=threading.Lock()
            )
            shm_req.set_pd_req_rank_state(self.rank_in_dp, RemoteTransferStatusType.IN_PROGRESS.value)
            req.in_prefill_or_transfer = True
            self.inflght_transfer_requests[group_req_id] = req
        else:
            transfer_state: TransferState = remote_request.transfer_state
            with transfer_state.lock:
                transfer_state.current_chunk_id += 1
                transfer_state.current_kv_len = req.cur_kv_len
                transfer_state.is_finished = req.finish_status.is_finished()
                if next_token_id:
                    transfer_state.next_token_id = next_token_id
                    transfer_state.next_token_logprob = next_token_logprob


    async def _transfer_kv_to_remote_paged_batch(self, transfer_reqs: List[KVMoveRequest]):
        start = time.time()
        requests_by_agents = dict()
        transfer_pages = self.page_scheduer.borrow(len(transfer_reqs))
        # first copy the kv to transfer pages & build notification
        for trans_req, page_index in zip(transfer_reqs, transfer_pages):
            trans_req: KVMoveRequest
            group_req_id = trans_req.group_req_id
            remote_request: PrefillRequest = self.remote_prefill_requests.get(group_req_id)
            transfer_state: TransferState = remote_request.transfer_state
            decode_id: int = remote_request.decode_id
            if decode_id not in requests_by_agents:
                requests_by_agents[decode_id] = ([], [], [])

            with transfer_state.lock:

                start_kv_len = transfer_state.transfered_kv_len
                trans_kv_len = min(trans_req.cur_kv_len - trans_req.prev_kv_len, self.nixl_agent.page_size)
                trans_kv_index = transfer_state.token_index[start_kv_len: start_kv_len + trans_kv_len]
                self.model.mem_manager.kv_move_buffer[page_index][:trans_kv_len] = self.model.mem_manager.kv_buffer[:,trans_kv_index, :, : ].transpose(0, 1)

                receive_page = transfer_state.free_page_ids.pop(0)
                requests_by_agents[decode_id][0].append(page_index)
                requests_by_agents[decode_id][1].append(receive_page)
                is_last = (transfer_state.is_finished and start_kv_len + trans_kv_len == transfer_state.current_kv_len)

                requests_by_agents[decode_id][2].append(RemotePrefillStatus(
                    transfer_type=RemoteTransferType.PAGE_TRANSFER,
                    group_req_id=group_req_id,
                    status=RemoteTransferStatusType.SUCCESS,
                    chunk_id=transfer_state.current_chunk_id,
                    is_last=is_last,
                    page_id=receive_page,
                    kv_start=start_kv_len,
                    kv_len=trans_kv_len,
                    next_token_id=transfer_state.next_token_id,
                    next_token_logprob=transfer_state.next_token_logprob
                ))
                transfer_state.transfered_kv_len += trans_kv_len

        # wait copy done
        torch.cuda.current_stream().synchronize()
        for decode_id, (transfer_pages, receive_pages, notifications) in requests_by_agents.items():
            assert len(transfer_reqs) == len(receive_pages), "transfer_reqs and receive_pages should have same length"
            # transfer
            self.nixl_agent.write_blocks_paged(decode_id, transfer_pages, receive_pages, notifications)


        logger.info(
            f"transfer kv to remote paged batch: {len(transfer_reqs)} "
            f"took: {time.time() - start} seconds"
        )

    async def _handle_transfer_loop(self):
        while True:
            free_transfer_pages = self.page_scheduer.current_size()
            if free_transfer_pages < 1:
                await asyncio.sleep(PDNIXLBackendBase._THREAD_WAIT_INTERVAL)
                continue

            transfer_reqs = []
            for group_req_id, req in self.inflght_transfer_requests.items():
                remote_request: PrefillRequest = self.remote_prefill_requests.get(group_req_id)
                transfer_state: TransferState = remote_request.transfer_state
                with transfer_state.lock:
                    if transfer_state.completed() or len(transfer_state.free_page_ids) == 0:
                        continue

                    if transfer_state.transfered_kv_len >= transfer_state.current_kv_len:
                        continue

                    transfer_reqs.append(
                        KVMoveRequest(
                            group_req_id=group_req_id,
                            prev_kv_len=transfer_state.transfered_kv_len,
                            cur_kv_len=transfer_state.current_kv_len,
                        )
                    )
                if len(transfer_reqs) >= free_transfer_pages:
                    break

            if len(transfer_reqs) > 0:
                await self._transfer_kv_to_remote_paged_batch(transfer_reqs)

            await asyncio.sleep(PDNIXLBackendBase._THREAD_WAIT_INTERVAL)

    async def _wait_page_transfer_loop(self):
        while True:
            # local pages can be reused as soon as transfer is done
            done_pages, done_requests = await self.nixl_agent.get_done_page_transfers()
            if len(done_pages):
                self.page_scheduer.return_(done_pages)

            # release requests when prefill done
            for req_id, status in done_requests:
                if req_id not in self.inflght_transfer_requests:
                    if self.is_master_in_dp:
                        logger.warning(f"{req_id} not found in inflght_transfer_requests")
                    continue

                req: InferReq = self.inflght_transfer_requests[req_id]
                shm_req: PDNIXLChunkedPrefillReq = req.shm_req
                shm_req.set_pd_req_rank_state(self.rank_in_dp, status.value)
                transfer_state = self.remote_prefill_requests[req_id].transfer_state
                if self.is_master_in_dp:
                    logger.info(
                        f"req: {req_id} kv transfer with state: {status} "
                        f"took: {time.time() - transfer_state.start_time} seconds"
                    )
                # only delete success transfers, failed / aborted will delete after send abort notification
                if status == RemoteTransferStatusType.SUCCESS:
                    del self.inflght_transfer_requests[req_id]
                    del self.remote_prefill_requests[req_id]

            # remote pages should be released after nofication received
            notifies = self.nixl_agent.get_new_notifs()
            for _, trans_acks in notifies.items():
                for trans_ack_bytes in trans_acks:
                    trans_acks_noti: Notification = Notification.from_bytes(trans_ack_bytes)
                    assert trans_acks_noti.type == NotificationType.TRANSFER_NOTIFY_ACK
                    for trans_ack in trans_acks_noti.data:
                        ack = PageTransferAck.deserialize(trans_ack)
                        remote_request: PrefillRequest = self.remote_prefill_requests.get(ack.group_req_id)
                        if remote_request is None:
                            continue

                        transfer_state: TransferState = remote_request.transfer_state
                        with transfer_state.lock:
                            transfer_state.free_page_ids.append(ack.page_id)

            await asyncio.sleep(PDNIXLBackendBase._THREAD_WAIT_INTERVAL)


    async def _wait_transfer_loop(self):
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
                shm_req: PDNIXLChunkedPrefillReq = req.shm_req
                shm_req.set_pd_req_rank_state(self.rank_in_dp, state)
                transfer_state = self.remote_prefill_requests[req_id].transfer_state
                if self.is_master_in_dp:
                    logger.info(
                        f"req: {req_id} kv transfer with state: {state} "
                        f"took: {time.time() - transfer_state.start_time} seconds"
                    )
                del self.remote_prefill_requests[req_id]
                del self.inflght_transfer_requests[req_id]

            time.sleep(PDNIXLBackendBase._THREAD_WAIT_INTERVAL)

    async def _handle_prefill_loop(self):
        while True:
            request: RemoteRequest = self.from_remote_queue.get()
            if request.type == RemoteRequstType.REMOTE_CONNECT:
                request: ConnectRequest
                logger.info(f"connect request received from: {request.decode_id}")
                self.nixl_agent.add_remote_agent(
                    NixlMetadata(
                        id=request.decode_id,
                        num_tokens=request.num_tokens,
                        num_pages=request.num_pages,
                        agent_metadatas=request.agent_metadatas,
                        agent_mem_descs=request.agent_mem_descs,
                        agent_page_mem_descs=request.agent_page_mem_descs,
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

    def _transfer_kv_to_remote(self, req: InferReq, group_req_id: int, cur_kv_len: int, is_finished: bool):
        start = time.time()
        remote_request: PrefillRequest = self.remote_prefill_requests[group_req_id]

        transfer_state = remote_request.transfer_state
        token_index = self.model.req_manager.req_to_token_indexs[req.req_idx]

        kv_transfer_req = KVMoveRequest(
            group_req_id=group_req_id,
            token_ids=token_index[: cur_kv_len].tolist(),
            prev_kv_len=transfer_state.current_kv_len,
            cur_kv_len=cur_kv_len,
        )
        if transfer_state.current_chunk_id == 0:
            self.inflght_transfer_requests[group_req_id] = req
            logger.debug(
                f"put {group_req_id} into inflght_transfer_requests and size: {len(self.inflght_transfer_requests)}"
            )

        # kick off kv transfer
        self.nixl_agent.write_blocks(kv_transfer_req, remote_request, is_finished)

        transfer_state.current_kv_len = cur_kv_len
        transfer_state.current_chunk_id += 1
        logger.info(
            f"transfer kv to remote: {group_req_id} "
            f"current chunk id: {transfer_state.current_chunk_id} {cur_kv_len} "
            f"took: {time.time() - start} seconds"
        )

    def _post_remote_prefill(self, req: InferReq, success: bool = True):

        req.in_prefill_or_transfer = False
        req.cur_kv_len = req.get_cur_total_len()
        if self.is_master_in_dp:
            req.shm_req.shm_cur_kv_len = req.cur_kv_len

        if not success:
            self.request_to_first_token.pop(group_req_id, None)
            return

        group_req_id = req.shm_req.group_req_id
        assert group_req_id in self.request_to_first_token
        token_id, token_logprob = self.request_to_first_token.pop(group_req_id)

        req.set_next_gen_token_id(token_id, token_logprob)
        req.cur_output_len += 1

        req.out_token_id_count[token_id] += 1
        req.update_finish_status(self.eos_id)

        if self.is_master_in_dp:
            req.shm_req.shm_cur_output_len = req.cur_output_len

            if req.finish_status.is_finished():
                req.shm_req.finish_token_index = req.get_cur_total_len() - 1
                req.shm_req.finish_status = req.finish_status

            req.shm_req.candetoken_out_len = req.cur_output_len

    def _decode_filter_reqs(
        self, prefill_reqs: List[InferReq], aborted_reqs: List[InferReq], decode_reqs: List[InferReq]
    ):
        new_prefill_reqs: List[InferReq] = []
        new_aborted_reqs: List[InferReq] = []
        remote_prefill_reqs: List[InferReq] = []

        # filter out aborted requests
        for req in aborted_reqs:
            if req.in_prefill_or_transfer:
                shm_req: PDNIXLChunkedPrefillReq = req.shm_req
                state = shm_req.get_pd_req_state()
                if state != RemoteTransferStatusType.IN_PROGRESS.value:
                    new_aborted_reqs.append(req)
                    self._post_remote_prefill(req, False)
                else:
                    remote_prefill_reqs.append(req)
            else:
                new_aborted_reqs.append(req)

        for req in prefill_reqs:
            if req.in_prefill_or_transfer:
                shm_req: PDNIXLChunkedPrefillReq = req.shm_req
                # state is updated by router
                state = shm_req.get_pd_req_state()
                if state == RemoteTransferStatusType.SUCCESS.value:  # success
                    self._post_remote_prefill(req)
                    decode_reqs.append(req)
                elif state == RemoteTransferStatusType.FAILED.value:  # failure
                    self._post_remote_prefill(req, False)
                    new_aborted_reqs.append(req)
                elif state == RemoteTransferStatusType.IN_PROGRESS.value:  # in progress
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
                shm_req: PDNIXLChunkedPrefillReq = req.shm_req
                state = shm_req.get_pd_req_state()
                if state == RemoteTransferStatusType.SUCCESS.value:  # success
                    new_ok_finished_reqs.append(req)
                    req.in_prefill_or_transfer = False
                elif state == RemoteTransferStatusType.FAILED.value:  # failure
                    aborted_reqs.append(req)
                    req.in_prefill_or_transfer = False
                elif state == RemoteTransferStatusType.IN_PROGRESS.value:
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

        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
        mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0])


        req_to_token_indexs = g_infer_context.req_manager.req_to_token_indexs
        for idx, req_idx in enumerate(nopad_b_req_idx):
            cur_kv_len = req_objs[idx].cur_kv_len
            seq_len = nopad_b_seq_len[idx]
            mem_start = nopad_b_start_loc[idx]
            mem_end = nopad_b_start_loc[idx+1]
            req_to_token_indexs[req_idx, cur_kv_len:nopad_b_seq_len[idx]] = mem_indexes[mem_start:mem_end]

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
                if group_req_id in self.inflght_transfer_requests:
                    del self.inflght_transfer_requests[group_req_id]