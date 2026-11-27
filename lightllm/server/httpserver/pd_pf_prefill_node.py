"""Prefill 节点 httpserver 上，PD 分离的 Prefill 先推理模式的会话。

由 pd_loop 在这条 master 连接上调用。
"""

import asyncio
import pickle

from lightllm.server.core.objs import SamplingParams
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import ObjType, PDDecodeNodeInfo
from lightllm.utils.error_utils import ServerBusyError
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


async def run_pd_pf_prefill_request(
    manager: HttpServerManager,
    prompt,
    sampling_params: SamplingParams,
    multimodal_params: MultimodalParams,
    forwarding_queue,
    pd_upload_websocket,
    decode_kv_ready_event: asyncio.Event,
) -> None:
    """在这条 master 连接上跑完一次 PD 分离的 Prefill 先推理模式，直到 KV 不再由本请求占着。"""
    group_request_id = sampling_params.group_request_id
    sampling_params.pd_prefill_first = True
    logger.info(
        f"prefill-first stage=start, group_request_id={group_request_id}, "
        f"max_new_tokens={sampling_params.max_new_tokens}. infer first, do not wait for decode"
    )

    try:
        # 1. 推理。pd_upload_websocket 置空，跳过老握手；token 转给 master，首 token 可先返回。
        # 最后一段被调度时，推理进程经 pd_report 上报 prompt，不在这里轮询。
        async for sub_req_id, request_output, metadata, finish_status in manager.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            request=None,
            pd_upload_websocket=None,
            pd_event=None,
        ):
            metadata["node_mode"] = manager.args.run_mode
            await forwarding_queue.put((sub_req_id, request_output, metadata, finish_status))
        logger.info(
            f"prefill-first stage=infer-finished, group_request_id={group_request_id}. kv still held by this request"
        )

        # 2. 等 decode KV 就绪。秒数比 master 等 upkv 多 30 秒，让 master 的 ABORT 通常先到。
        resource_wait_seconds = sampling_params.pd_node_resource_wait_timeout_seconds
        if resource_wait_seconds < 0:
            decode_wait_seconds = 180
        else:
            decode_wait_seconds = max(180, 2 * resource_wait_seconds + 15)
        timeout_seconds = decode_wait_seconds + 30
        logger.info(
            f"prefill-first stage=wait-decode-kv-ready, group_request_id={group_request_id}, "
            f"resource_wait_seconds={resource_wait_seconds}, decode_wait_seconds={decode_wait_seconds}, "
            f"timeout_seconds={timeout_seconds}. "
            f"negative resource_wait_seconds uses a 180s decode wait; otherwise two resource-wait budgets "
            f"plus 15s margin, floored at 180s. this wait adds 30s so the master ABORT usually arrives first"
        )
        await asyncio.wait_for(decode_kv_ready_event.wait(), timeout=timeout_seconds)
        decode_node_info = decode_kv_ready_event.decode_node_info
        logger.info(
            f"prefill-first stage=decode-kv-ready, group_request_id={group_request_id}, "
            f"ready_kv_len={decode_node_info.ready_kv_len}. transfer starts at the unmatched span"
        )

        # 3. 端点已在收到通知时写入共享内存。等到传输完成、请求离开。KV 由推理进程在结束路径释放。
        while group_request_id in manager.req_id_to_out_inf:
            await asyncio.sleep(0.1)
        logger.info(f"prefill-first stage=request-left, group_request_id={group_request_id}")
    except ServerBusyError as e:
        logger.warning(f"prefill-first request {group_request_id} rejected: {e.message}")
        try:
            await pd_upload_websocket.send(pickle.dumps((ObjType.PD_UPLOAD_SERVER_BUSY, group_request_id, e.message)))
        except BaseException as report_error:
            logger.exception(
                f"report prefill-first request rejection failed, group_request_id: {group_request_id}, "
                f"error: {type(report_error).__name__}: {report_error}"
            )
    except asyncio.TimeoutError:
        # generate 已经返回，这里的超时发生在占住 KV 之后，需要本节点先释放。
        logger.warning(
            f"prefill-first stage=wait-decode-kv-ready timed out, group_request_id={group_request_id}. "
            f"abort this request and release kv"
        )
        await manager.abort(group_req_id=group_request_id)
        try:
            await pd_upload_websocket.send(
                pickle.dumps(
                    (
                        ObjType.PD_UPLOAD_GENERATE_ERROR,
                        group_request_id,
                        "TimeoutError: wait decode kv ready timed out",
                    )
                )
            )
        except BaseException as report_error:
            logger.exception(
                f"report prefill-first generate error failed, group_request_id: {group_request_id}, "
                f"error: {type(report_error).__name__}: {report_error}"
            )
    except asyncio.CancelledError:
        # PD master 主动 abort 或连接断开清理任务时会走取消路径，不需要反向重复上报。
        pass
    except BaseException as e:
        logger.exception(f"prefill-first request {group_request_id} failed: {str(e)}")
        await manager.abort(group_req_id=group_request_id)
        try:
            await pd_upload_websocket.send(
                pickle.dumps((ObjType.PD_UPLOAD_GENERATE_ERROR, group_request_id, f"{type(e).__name__}: {str(e)}"))
            )
        except BaseException as report_error:
            logger.exception(
                f"report prefill-first generate error failed, group_request_id: {group_request_id}, "
                f"error: {type(report_error).__name__}: {report_error}"
            )
    return


def write_prefill_first_decode_node_info(
    manager: HttpServerManager, group_request_id: int, decode_node_info: PDDecodeNodeInfo
) -> bool:
    """收到 decode KV 就绪通知后，立刻把端点写入共享内存。"""
    req_status = manager.req_id_to_out_inf.get(group_request_id)
    if req_status is None:
        logger.error(f"prefill-first request {group_request_id} disappeared before writing decode kv ready")
        return False
    shm_req = req_status.group_req_objs.shm_req_objs[0]
    shm_req.sample_params.pd_kv_trans_params.set(pickle.dumps(decode_node_info))
    logger.info(f"prefill-first stage=wrote-decode-kv-ready, group_request_id={group_request_id}")
    return True
