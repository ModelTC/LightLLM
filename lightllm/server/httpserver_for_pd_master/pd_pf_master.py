"""HttpServerManagerForPDMaster 的扩展 Mixin。

通过多继承挂到主类上，把 PD 分离的 Prefill 先推理模式的会话接口从 manager 主体中拆出。
"""

from __future__ import annotations

import asyncio
import pickle
from typing import TYPE_CHECKING, Any, Optional

from fastapi import Request

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import ObjType, PDDecodeNodeInfo, PDUpKVStatus, PD_Client_Obj, PdPfPrefillKvReady
from lightllm.utils.error_utils import ClientDisconnected, RequestStopped, ServerBusyError
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster

logger = init_logger(__name__)


class HttpPdPrefillFirstManagerHelper:
    """PD 分离的 Prefill 先推理模式的 master 会话。"""

    async def fetch_prefill_first_stream(
        self: HttpServerManagerForPDMaster,
        p_node: PD_Client_Obj,
        prompt: str,
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
        origin_request_id: int,
    ):
        """最后一段 prefill 被调度后让 decode 占页。upkv 到达就把传输端点交回 prefill，与首 token 并行。

        只生成 1 个 token 的请求在 prefill 上结束，不占 decode。
        """
        from lightllm.server.httpserver_for_pd_master.manager import ReqStatus

        group_request_id = sampling_params.group_request_id
        sampling_params.pd_prefill_first = True
        # 只生成 1 个 token 时，prefill 的结束状态就是请求结束，不提前占 decode。
        only_gen_one_token = sampling_params.max_new_tokens == 1
        # 多 worker 时每个 master 的节点 ID 不同。请求带上它，P/D 才把状态送回这次接住请求的 master。
        sampling_params.pd_master_node_id.initialize(self.args.pd_node_id)
        p_websocket = p_node.pick_websocket()
        if p_websocket is None:
            raise ServerBusyError("pd prefill node has no live httpserver connection")

        req_status = ReqStatus(
            req_id=group_request_id,
            p_node=p_node,
            d_node=None,
            p_websocket=p_websocket,
            d_websocket=None,
        )
        self.req_id_to_out_inf[group_request_id] = req_status
        saved_max_new_tokens = sampling_params.max_new_tokens
        sampling_params.max_new_tokens = 1
        wait_decode_node_kv_ready_task: Optional[asyncio.Task] = None

        async def _noop_close_wait_decode_node_kv_ready_task() -> None:
            return

        close_wait_decode_node_kv_ready_task = _noop_close_wait_decode_node_kv_ready_task

        try:
            # 1. 发给 prefill。max_new_tokens 临时改成 1，只采样首 token，不等 decode 端点。
            # 等 PD_PF_PREFILL_KV_READY：最后一段被调度时上报，forward 还没结束。
            # 资源等待和后面的 upkv 用同一档；计算另给 180 秒。
            # Prefill 自己的资源到点会上报繁忙并叫醒这次等待，这里是它不再上报时的截止。
            resource_wait_seconds = sampling_params.pd_node_resource_wait_timeout_seconds
            if resource_wait_seconds < 0:
                resource_budget_seconds = 180
            else:
                resource_budget_seconds = max(180, 2 * resource_wait_seconds + 15)
            prefill_compute_seconds = 180
            prefill_ready_timeout_seconds = resource_budget_seconds + prefill_compute_seconds
            logger.info(
                f"prefill-first waiting pf-prefill-kv-ready, group_request_id={group_request_id}, "
                f"resource_wait_seconds={resource_wait_seconds}, resource_budget_seconds={resource_budget_seconds}, "
                f"prefill_compute_seconds={prefill_compute_seconds}, timeout_seconds={prefill_ready_timeout_seconds}. "
                f"negative resource_wait_seconds uses a 180s resource budget; otherwise two resource-wait budgets "
                f"plus 15s margin, floored at 180s. another 180s is added for the prefill compute itself"
            )
            prefill_payload = pickle.dumps((ObjType.PD_PF_PREFILL_REQ, (prompt, sampling_params, multimodal_params)))
            self._reject_oversized_control_message(payload=prefill_payload, group_request_id=group_request_id)
            await p_websocket.send_bytes(prefill_payload)
            sampling_params.max_new_tokens = saved_max_new_tokens
            try:
                await self._wait_for_event_or_disconnect(
                    event=req_status.pf_prefill_kv_ready_event,
                    request=request,
                    timeout=prefill_ready_timeout_seconds,
                    group_request_id=group_request_id,
                    origin_request_id=origin_request_id,
                    stage="pf-prefill-kv-ready",
                )
            except ServerBusyError:
                logger.warning(
                    f"group_request_id: {group_request_id} wait pf-prefill-kv-ready time out, server is busy now."
                )
                raise
            req_status.raise_if_error()

            # 2. 还要继续生成时让 decode 占页，并开始等 upkv。upkv 到达就下发端点，与首 token 并行。
            if not only_gen_one_token:
                # TODO: 这里再次调用 select_p_d_node。round_robin 每次都会把 prefill 和 decode 的下标都加 1。
                # 请求开始时已经选过一次，这次只用返回的 decode，prefill 下标仍被空走一步。
                # 两台节点时，真正用到的 prefill 会固定在同一台，decode 固定在另一台。
                async with self.pd_manager.selector.selection_lock:
                    _, d_node, _ = await self.select_p_d_node(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        multimodal_params=multimodal_params,
                    )
                if d_node is None:
                    raise ServerBusyError("pd prefill-first found no decode node")
                d_websocket = d_node.pick_websocket()
                if d_websocket is None:
                    raise ServerBusyError("pd decode node has no live httpserver connection")
                req_status.d_node = d_node
                req_status.d_websocket = d_websocket

                ready: PdPfPrefillKvReady = req_status.pf_prefill_kv_ready_event.ready
                decode_payload = pickle.dumps((ObjType.REQ, (ready.prompt_ids, sampling_params, MultimodalParams())))
                self._reject_oversized_control_message(payload=decode_payload, group_request_id=group_request_id)
                await d_websocket.send_bytes(decode_payload)

                async def wait_decode_node_kv_ready() -> None:
                    # upkv 的时限从占页开始算，和最后一段 forward 重叠。
                    if resource_wait_seconds < 0:
                        decode_handshake_timeout = 180
                    else:
                        decode_handshake_timeout = max(180, 2 * resource_wait_seconds + 15)
                    logger.info(
                        f"prefill-first waiting upkv, group_request_id={group_request_id}, "
                        f"resource_wait_seconds={resource_wait_seconds}, timeout_seconds={decode_handshake_timeout}. "
                        f"negative resource_wait_seconds uses 180s; otherwise two resource-wait budgets "
                        f"plus 15s margin, floored at 180s"
                    )
                    try:
                        await self._wait_for_event_or_disconnect(
                            event=req_status.up_status_event,
                            request=request,
                            timeout=decode_handshake_timeout,
                            group_request_id=group_request_id,
                            origin_request_id=origin_request_id,
                            stage="prefill-first-decode",
                        )
                    except ServerBusyError:
                        logger.warning(
                            f"group_request_id: {group_request_id} wait prefill-first decode upkv time out, "
                            f"server is busy now."
                        )
                        raise
                    req_status.raise_if_error()
                    upkv_status: PDUpKVStatus = req_status.up_status_event.upkv_status
                    decode_node_info: PDDecodeNodeInfo = pickle.loads(upkv_status.pd_kv_trans_params)
                    ready_payload = pickle.dumps((ObjType.PD_PF_DECODE_KV_READY, group_request_id, decode_node_info))
                    await p_websocket.send_bytes(ready_payload)

                wait_decode_node_kv_ready_task = asyncio.create_task(wait_decode_node_kv_ready())

                async def _close_wait_decode_node_kv_ready_task() -> None:
                    if not wait_decode_node_kv_ready_task.done():
                        wait_decode_node_kv_ready_task.cancel()
                    await asyncio.gather(wait_decode_node_kv_ready_task, return_exceptions=True)

                close_wait_decode_node_kv_ready_task = _close_wait_decode_node_kv_ready_task

            # 3. 等这段 forward 的首 token。端点下发在后台进行。请求已经结束就停掉下发并 abort。
            try:
                first_token = await asyncio.wait_for(
                    self._wait_prefill_first_token(
                        req_status=req_status,
                        request=request,
                        saved_max_new_tokens=saved_max_new_tokens,
                        origin_request_id=origin_request_id,
                    ),
                    timeout=prefill_compute_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise ServerBusyError("prefill-first timed out waiting for the first token") from exc
            if first_token[3].is_finished():
                await close_wait_decode_node_kv_ready_task()
                await self.abort(group_request_id=group_request_id, p_node=p_node, d_node=req_status.d_node)
                yield first_token
                return

            # 4. 交出首 token。端点在 upkv 到达时已经开始下发，这里等到它结束。
            yield first_token
            if wait_decode_node_kv_ready_task is not None:
                await wait_decode_node_kv_ready_task

            # 5. 继续交出后到的 token。Decode 会再上报同一个首 token，在下面丢掉。
            # 端点下发和首 token 并行，这里恢复时 upkv 等待可能已经结束，先看 stop 和断连。
            self.raise_if_request_stopped(origin_request_id)
            req_status.raise_if_error()
            if await request.is_disconnected():
                raise ClientDisconnected(
                    group_request_id=req_status.req_id,
                    reason="prefill-first decode period check network disconnected",
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
                        group_request_id=req_status.req_id,
                        reason="prefill-first decode period check network disconnected",
                    )
                if not await req_status.can_read(self.req_id_to_out_inf):
                    continue
                token_list = await req_status.pop_all_tokens()
                for item in token_list:
                    _, _, metadata, finish_status = item
                    # Prefill 的首 token 已经交出。Decode 从传输结果里再报一次同一个 token。
                    # 容量不足是分段续跑标记，要交给外层，不能当成重复首 token 丢掉。
                    if (
                        metadata.get("count_output_tokens") == 1
                        and not finish_status.is_error_finished()
                        and not finish_status.is_finished_pd_decode_capacity()
                    ):
                        continue
                    yield item
        except RequestStopped:
            await close_wait_decode_node_kv_ready_task()
            await self.abort(group_request_id=group_request_id, p_node=p_node, d_node=req_status.d_node)
            return
        finally:
            # 先停掉端点下发，再 abort，避免两条发送叠在同一条 prefill 连接上。
            await close_wait_decode_node_kv_ready_task()
            # 生成器退出时请求还在就 abort。正常结束后多通知一次没有实质影响。
            if group_request_id in self.req_id_to_out_inf:
                await self.abort(group_request_id=group_request_id, p_node=p_node, d_node=req_status.d_node)
        return

    # 记下 prefill KV 就绪上报并唤醒等待。请求还要等 decode KV 就绪，这里不结束。
    def record_pf_prefill_kv_ready(
        self: HttpServerManagerForPDMaster,
        ready: PdPfPrefillKvReady,
    ) -> None:
        req_status = self.req_id_to_out_inf.get(ready.group_request_id)
        if req_status is None:
            logger.error(f"PD_PF_PREFILL_KV_READY finds no request, group_request_id={ready.group_request_id}")
            return
        req_status.pf_prefill_kv_ready_event.ready = ready
        req_status.pf_prefill_kv_ready_event.set()
        return

    # 等到 prefill 的那一个首 token，并改写结束状态。连接断开、请求失败或不能续跑时抛异常。
    async def _wait_prefill_first_token(
        self: HttpServerManagerForPDMaster,
        req_status: Any,
        request: Request,
        saved_max_new_tokens: int,
        origin_request_id: int,
    ):
        # 首 token 和 decode 输出走不同通道，谁先到不固定。队列里可能先有 decode token。
        pending_tokens = []
        while True:
            self.raise_if_request_stopped(origin_request_id)
            req_status.raise_if_error()
            if await request.is_disconnected():
                raise ClientDisconnected(
                    group_request_id=req_status.req_id,
                    reason="prefill-first prefill token check network disconnected",
                )
            if not await req_status.can_read(self.req_id_to_out_inf):
                await req_status.wait_to_ready()
                continue
            pending_tokens.extend(await req_status.pop_all_tokens())
            prefill_token = None
            other_tokens = []
            for item in pending_tokens:
                if prefill_token is None and item[2].get("node_mode") == "prefill":
                    prefill_token = item
                else:
                    other_tokens.append(item)
            if prefill_token is None:
                continue
            if other_tokens:
                await req_status.put_tokens_to_front(token_list=other_tokens)
            sub_req_id, request_output, metadata, finish_status = prefill_token
            # Prefill 为了只采样首 token，把 max_new_tokens 临时改成了 1。真实长度不是 1 时，这次长度结束改回未结束。
            if saved_max_new_tokens != 1 and finish_status.is_finished_length():
                finish_status = FinishStatus(FinishStatus.NO_FINISH)
            return sub_req_id, request_output, metadata, finish_status
