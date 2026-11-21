"""HttpServerManagerForPDMaster 的扩展 Mixin。

通过多继承挂到主类上，把 PD 分离默认模式的会话接口从 manager 主体中拆出。
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Union

from fastapi import Request

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import ObjType, PDDecodeNodeInfo, PDUpKVStatus, PD_Client_Obj
from lightllm.utils.error_utils import ClientDisconnected, RequestStopped, ServerBusyError

if TYPE_CHECKING:
    from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, ReqStatus


class HttpPdNormalManagerHelper:
    """PD 分离默认模式的 master 会话。"""

    async def fetch_pd_stream(
        self: HttpServerManagerForPDMaster,
        p_node: PD_Client_Obj,
        d_node: PD_Client_Obj,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
        origin_request_id: int,
    ):
        from lightllm.server.httpserver_for_pd_master.manager import ReqStatus, logger

        group_request_id = sampling_params.group_request_id
        sampling_params.pd_master_node_id.initialize(self.args.pd_node_id)
        p_websocket = p_node.pick_websocket()
        d_websocket = d_node.pick_websocket()
        if p_websocket is None or d_websocket is None:
            raise ServerBusyError("pd node has no live httpserver connection")
        req_status = ReqStatus(group_request_id, p_node, d_node, p_websocket, d_websocket)
        self.req_id_to_out_inf[group_request_id] = req_status

        up_status_event = req_status.up_status_event
        prefill_prompt_ids_event = req_status.prefill_prompt_ids_event

        old_max_new_tokens = sampling_params.max_new_tokens
        sampling_params.max_new_tokens = 1
        prefill_payload = pickle.dumps((ObjType.REQ, (prompt, sampling_params, multimodal_params)))
        self._reject_oversized_control_message(payload=prefill_payload, group_request_id=group_request_id)
        await p_websocket.send_bytes(prefill_payload)

        try:
            try:
                await self._wait_for_event_or_disconnect(
                    prefill_prompt_ids_event,
                    request,
                    timeout=60,
                    group_request_id=group_request_id,
                    origin_request_id=origin_request_id,
                    stage="prefill",
                )
            except ServerBusyError:
                logger.warning(f"group_request_id: {group_request_id} wait prefill prompt ids time out")
                raise
            req_status.raise_if_error()

            prompt_ids = prefill_prompt_ids_event.prompt_ids
            logger.debug(f"group_request_id: {group_request_id} get prefill prompt ids len {len(prompt_ids)}")

            sampling_params.max_new_tokens = old_max_new_tokens
            decode_payload = pickle.dumps((ObjType.REQ, (prompt_ids, sampling_params, MultimodalParams())))
            self._reject_oversized_control_message(payload=decode_payload, group_request_id=group_request_id)
            await d_websocket.send_bytes(decode_payload)

            # Master 在等 Decode 上报。截止时间要盖住 Decode 拿到请求后的两段资源等待。
            # 两段各用一份预算，并且前后分开计时：_alloc_shm_req_indexes 先等共享请求槽，
            # 上限是 resource_wait_seconds；槽到位、请求进入 Router 后，router_arrival_time 重新起算，
            # has_timed_out_waiting_for_inference 再给一整份 resource_wait_seconds。
            # Router 这段不是到点立刻失败。_wait_to_token_package 每 5 秒才查一次，
            # 最坏会在期限之后将近 5 秒才抛出 ServerBusyError，这个错误还要回到这次握手。
            # 所以在两份预算之外再留 15 秒，这是检查迟到和错误传回的余量，不是第三段资源等待。
            # 预算为负表示没有下发资源等待上限（自限流关闭时字段保持 -1，或环境变量显式为负）。
            # 这时没有两段预算可包住，握手保持原来的 180 秒，避免 Decode 一直不上报时这次等待没有截止时间。
            # 预算较小时 max 仍保持原来的 180 秒下限。
            # Prefill 在此基础上再加 5 秒，因为它更早开始等，需要让 Master 先超时并发送 ABORT。
            resource_wait_seconds = sampling_params.pd_node_resource_wait_timeout_seconds
            if resource_wait_seconds < 0:
                decode_handshake_timeout = 180
            else:
                decode_handshake_timeout = max(180, 2 * resource_wait_seconds + 15)
            try:
                await self._wait_for_event_or_disconnect(
                    up_status_event,
                    request,
                    timeout=decode_handshake_timeout,
                    group_request_id=group_request_id,
                    origin_request_id=origin_request_id,
                    stage="decode",
                )
            except ServerBusyError:
                logger.warning(
                    f"group_request_id: {group_request_id} wait decode stage time out err, server is busy now."
                )
                raise
            req_status.raise_if_error()

            # 将 decode 节点上报的当前请求使用的decode节点的信息下发给 p 节点，这样 p 节点才知道将 kv 传输给那个 d 节点。
            upkv_status: PDUpKVStatus = up_status_event.upkv_status
            pd_kv_trans_params: bytes = upkv_status.pd_kv_trans_params
            decode_node_info: PDDecodeNodeInfo = pickle.loads(pd_kv_trans_params)
            await p_websocket.send_bytes(
                pickle.dumps((ObjType.PD_REQ_DECODE_NODE_INFO, group_request_id, decode_node_info))
            )

            first_token_gen = False
            needs_prefill_first_token = decode_node_info.ready_kv_len != len(prompt_ids) - 1
            prompt_cache_len_from_prefill = await self._wait_for_prefill_token_if_needed(
                req_status=req_status,
                request=request,
                group_request_id=group_request_id,
                needs_prefill_first_token=needs_prefill_first_token,
                ready_kv_len=decode_node_info.ready_kv_len,
            )

            while True:
                await req_status.wait_to_ready()
                # stop_request 标记的是原始 choice ID，group_request_id 只是当前 P/D 分段 ID。
                # Decode 不再输出 token 时，外层无法及时检查 stop；每次等待结束后主动检查，
                # 通过专用异常交由外层统一 abort 当前分段并结束该 choice。
                self.raise_if_request_stopped(origin_request_id)
                req_status.raise_if_error()
                if await request.is_disconnected():
                    raise ClientDisconnected(
                        group_request_id=group_request_id,
                        reason="fetch_pd_stream decode period check network disconnected",
                    )
                if await req_status.can_read(self.req_id_to_out_inf):
                    token_list = await req_status.pop_all_tokens()
                    for sub_req_id, request_output, metadata, finish_status in token_list:
                        output_index = metadata.get("count_output_tokens")
                        # 因为 pd 的 prefill 和 decode 节点都有可能上报首token，所以需要做一下过滤。
                        # Prefill 已返回首 token 后，Decode 若在出首 token 前遇到 KV 传输错误或取消，
                        # 仍可能以 count_output_tokens=1 上报 error/abort 的模拟结束 token。
                        # 该终态必须透传，否则会被当作重复首 token 丢弃，导致 HTTP 请求一直等待结束。
                        if output_index == 1 and not finish_status.is_error_finished():
                            if first_token_gen is False:
                                first_token_gen = True
                                node_run_mode = metadata.get("node_mode")
                                if node_run_mode == "prefill":
                                    if old_max_new_tokens != 1 and finish_status.is_finished_length():
                                        finish_status = FinishStatus(FinishStatus.NO_FINISH)
                                metadata["prompt_cache_len"] = prompt_cache_len_from_prefill
                                metadata["needs_prefill_first_token"] = needs_prefill_first_token
                                yield sub_req_id, request_output, metadata, finish_status
                            else:
                                continue
                        else:
                            metadata["prompt_cache_len"] = prompt_cache_len_from_prefill
                            yield sub_req_id, request_output, metadata, finish_status

        except RequestStopped:
            await self.abort(group_request_id=group_request_id, p_node=p_node, d_node=d_node)
            return

        return

    async def _wait_for_prefill_token_if_needed(
        self: HttpServerManagerForPDMaster,
        req_status: "ReqStatus",
        request: Request,
        group_request_id: int,
        needs_prefill_first_token: bool,
        ready_kv_len: int,
    ) -> int:
        if not needs_prefill_first_token:
            return ready_kv_len

        new_tokens = []
        while True:
            await req_status.wait_to_ready()
            req_status.raise_if_error()
            if await request.is_disconnected():
                raise ClientDisconnected(
                    group_request_id=group_request_id,
                    reason="fetch_pd_stream decode period check network disconnected",
                )
            if not await req_status.can_read(self.req_id_to_out_inf):
                continue

            new_tokens.extend(await req_status.pop_all_tokens())

            for index, token in enumerate(new_tokens):
                metadata = token[2]
                if metadata.get("node_mode") == "prefill":
                    prompt_cache_len = metadata.get("prompt_cache_len", 0)
                    # prefill 采样了这个首 token，放到已缓冲的 decode 输出前面再发布。
                    new_tokens.insert(0, new_tokens.pop(index))
                    await req_status.put_tokens_to_front(new_tokens)
                    return prompt_cache_len
