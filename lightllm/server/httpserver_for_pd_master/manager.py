import sys
import os
import asyncio
import uvloop
import time
import datetime
import ujson as json
import pickle
import httpx
from contextlib import aclosing

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from typing import Union, List, Tuple, Dict, Optional
from lightllm.server.core.objs import FinishStatus
from ..pd_io_struct import PD_Client_Obj, PDUpKVStatus, ObjType, PDDecodeNodeInfo
from lightllm.server.core.objs import SamplingParams, StartArgs
from ..multimodal_params import MultimodalParams
from ..tokenizer import get_tokenizer
from ..req_id_generator import ReqIDGenerator, convert_sub_id_to_group_id
from fastapi import Request
from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.statics_utils import MovingAverage
from lightllm.server.httpserver.manager import AsyncQueue
from lightllm.utils.error_utils import ClientDisconnected, ServerBusyError
from lightllm.utils.envs_utils import (
    get_pd_cache_high_priority_max_age_seconds,
    get_pd_high_priority_request_timeout_seconds,
)
from lightllm.utils.shm_port_args import get_shm_port_args
from .pd_selector import PDSelectionExtraInfo, create_selector

logger = init_logger(__name__)


class HttpServerManagerForPDMaster:
    def __init__(
        self,
        args: StartArgs,
    ):
        self.args = args
        self.max_req_total_len = args.max_req_total_len
        assert self.max_req_total_len is not None
        self.metric_client = MetricClient(get_shm_port_args().metric_port)
        self.id_gen = ReqIDGenerator()

        self.pd_manager = PDManager(args)

        self.req_id_to_out_inf: Dict[int, ReqStatus] = {}
        self.infos_queues = None  # 这个需要延迟初始化，否则使用的loop不对
        self.health_timeout = int(os.getenv("HEALTH_TIMEOUT", "200"))
        self.latest_success_infer_time = time.time()
        self.running_request_count = 0
        # 高优先级请求仍可比普通请求等待更久，但通过请求参数向开启本地限流的
        # P/D 节点传递有限的等待时间，避免资源异常时永久占用请求链路。
        self.pd_high_priority_request_time_out_seconds = get_pd_high_priority_request_timeout_seconds()
        self.pd_cache_high_priority_max_age_seconds = get_pd_cache_high_priority_max_age_seconds()
        self.disable_pd_cache_high_priority = args.disable_pd_cache_high_priority

        self.tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)

        self.first_time_costs = MovingAverage()
        self.per_token_costs = MovingAverage()
        return

    def get_real_supported_max_req_total_len(self):
        # HttpServerManager.generate 会借用 _check_and_repair_length(self, ...)，其中会调用本方法。
        # PD master 无本地 token 池 shm 计数；上限与启动参数及子节点对齐的 max_req_total_len 一致。
        return self.max_req_total_len

    def is_healthy(self):
        time_since_last_success = time.time() - self.latest_success_infer_time
        if time_since_last_success <= self.health_timeout:
            return True
        if self.running_request_count == 0 and len(self.req_id_to_out_inf) == 0:
            return True

        logger.warning(
            f"PD Master health check failed: no successful inference for {int(time_since_last_success)}s "
            f"and {self.running_request_count} requests are still running"
        )
        return False

    async def register_pd(self, pd_info_json, websocket):
        self.pd_manager.register_pd(pd_info_json, websocket)
        return

    async def remove_pd(self, pd_info_json):
        self.pd_manager.remove_pd(pd_info_json)
        return

    async def update_req_status(self, upkv_status: PDUpKVStatus):
        try:
            group_request_id = convert_sub_id_to_group_id(upkv_status.group_request_id)
            up_status_event = self.req_id_to_out_inf[group_request_id].up_status_event
            up_status_event.upkv_status = upkv_status
            up_status_event.set()
        except:
            pass
        return

    def tokens(self, prompt, multimodal_params, samping_params: SamplingParams, kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        prompt_ids = self.tokenizer.encode(prompt, None, **kwargs)
        image_tokens = 0
        img_count = 0
        audio_tokens = 0
        audio_count = 0
        for img in multimodal_params.images:
            img_count += 1
            self.tokenizer.init_imageitem_extral_params(img, multimodal_params, samping_params)
            token_num = self.tokenizer.get_image_token_length(img)
            if token_num > self.args.max_image_token_count:
                err_msg = (
                    f"the image token count {token_num} > max_image_token_count {self.args.max_image_token_count}. "
                    f"You can increase this limit by setting --max_image_token_count to a larger value when starting "
                    f"LightLLM. Warning: increasing this limit raises runtime OOM risk."
                )
                logger.warning(err_msg)
                raise ValueError(err_msg)
            image_tokens += token_num
        for audio in multimodal_params.audios:
            audio_count += 1
            self.tokenizer.init_audioitem_extral_params(audio, multimodal_params, samping_params)
            audio_tokens += self.tokenizer.get_audio_token_length(audio)
        return len(prompt_ids) + image_tokens + img_count + audio_tokens + audio_count

    async def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj, PDSelectionExtraInfo]:
        return self.pd_manager.select_p_d_node(prompt, sampling_params, multimodal_params)

    async def _wait_for_pd_master_request_slot(self) -> None:
        """PD Master 请求准入的预留接口，当前不执行限流。"""
        return

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ):
        await self._wait_for_pd_master_request_slot()

        was_idle = self.running_request_count == 0
        self.running_request_count += 1
        if was_idle:
            self.latest_success_infer_time = time.time()
        try:
            async with aclosing(self._generate(prompt, sampling_params, multimodal_params, request)) as generator:
                async for result in generator:
                    yield result
        finally:
            self.running_request_count -= 1

    async def _generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ):
        assert isinstance(prompt, str), "prompt must be str"
        start_time = time.time()
        await multimodal_params.verify_and_preload(request)
        # 计算输入的 input_token_num, 进行校验，如果输入+输出参数设置太长，则将
        # sampling_params 的参数进行修正。
        input_token_num = await asyncio.to_thread(self.tokens, prompt, multimodal_params, sampling_params)
        fake_prompt_ids = [0 for _ in range(input_token_num)]
        from lightllm.server.httpserver.manager import HttpServerManager

        await HttpServerManager._check_and_repair_length(
            self, prompt_ids=fake_prompt_ids, sampling_params=sampling_params
        )

        origin_sampling_params = SamplingParams.from_buffer_copy(sampling_params)
        origin_group_request_id = self.id_gen.generate_id()

        # Record one user request even when it is expanded into multiple independent
        # n=1 requests below. The externally visible ids remain in the same request
        # group so OpenAI streaming can derive choice_index from the sub request id.
        await self._log_req_header(request, origin_group_request_id)
        self.metric_client.counter_inc("lightllm_request_count")
        self.metric_client.histogram_observe("lightllm_request_max_new_tokens", origin_sampling_params.max_new_tokens)

        choice_count = origin_sampling_params.n
        generators = []
        for choice_index in range(choice_count):
            choice_sampling_params = SamplingParams.from_buffer_copy(origin_sampling_params)
            choice_sampling_params.n = 1
            choice_sampling_params.best_of = 1
            generators.append(
                self._generate_one(
                    prompt,
                    choice_sampling_params,
                    multimodal_params,
                    request,
                    start_time,
                    origin_group_request_id + choice_index,
                )
            )

        request_finished_successfully = True
        async for result in self._merge_choice_generators(generators):
            finish_status = result[3]
            if finish_status.is_error_finished():
                request_finished_successfully = False
            yield result
        if request_finished_successfully:
            self.metric_client.counter_inc("lightllm_request_success")
        return

    async def _generate_one(
        self,
        prompt: str,
        origin_sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
        start_time: float,
        origin_request_id: int,
    ):
        block_group_request_id = origin_request_id
        p_node = None
        d_node = None
        pending_prefill_load_chars = None

        try:
            p_node, d_node, selection_extra_info = await self.select_p_d_node(
                prompt, origin_sampling_params, multimodal_params
            )
            if not p_node or not d_node:
                logger.error(f"{origin_request_id}: No p_node or d_node found")
                raise Exception(f"{origin_request_id}: No p_node or d_node found")

            cache_age_seconds = None
            if selection_extra_info.cache_last_insert_time is not None:
                cache_age_seconds = max(0.0, time.monotonic() - selection_extra_info.cache_last_insert_time)
            # TODO: 后续应收集系统实际请求的 cache 命中信息及其随时间变化的规律，自动估算并动态调整
            # pd_cache_high_priority_max_age_seconds，而不是继续使用环境变量配置的固定时间窗。
            has_fresh_high_cache_hit = (
                not self.disable_pd_cache_high_priority
                and selection_extra_info.estimated_cache_hit_rate > 0.8
                and cache_age_seconds is not None
                and cache_age_seconds <= self.pd_cache_high_priority_max_age_seconds
            )

            history_gen_token_strs = []
            origin_prompt_cache_len = None
            remaining_max_new_tokens = origin_sampling_params.max_new_tokens
            segment_index = 0
            # 后续分段的 prompt 会追加已生成内容；始终保留所有分段中最小的 prompt token 数，
            # 对外 usage 才能反映用户的原始输入长度，而不是最后一次续跑的 block_prompt 长度。
            prompt_tokens = sys.maxsize

            # Decode 节点容量不足时会用专用状态结束当前分段。
            # PD Master 吞掉该内部分段 marker，并用剩余 token 限额在同一组 P/D 节点上继续。
            while remaining_max_new_tokens > 0:
                sampling_params = SamplingParams.from_buffer_copy(origin_sampling_params)
                block_group_request_id = self.id_gen.generate_id()
                sampling_params.group_request_id = block_group_request_id
                logger.info(f"pd log gen sub req id {block_group_request_id} for main req id {origin_request_id}")
                sampling_params.max_new_tokens = remaining_max_new_tokens
                # 首段仅在预计输入 cache 命中率高于 0.8 且命中记录仍在有效时间窗内时
                # 提升优先级，避免为可能已被 P 节点淘汰的陈旧 KV cache 插队。第二段及
                # 后续分段仍统一使用高优先级，避免因临时资源紧张导致分段续跑失败。
                sampling_params.pd_high_priority_request = segment_index > 0 or has_fresh_high_cache_hit
                # 为高优先级请求下发较长的有限等待时间；P/D 节点仅在自身开启
                # 本地限流时使用该值，未开启限流时仍保持无限等待。
                if sampling_params.pd_high_priority_request:
                    sampling_params.pd_high_priority_request_time_out_seconds = (
                        self.pd_high_priority_request_time_out_seconds
                    )

                # 分段请求始终复用循环外选定的 P 节点；这里只按每段实际发送的
                # prompt 更新该节点的在途 prefill 负载，不会重新选点。
                block_prompt = prompt + "".join(history_gen_token_strs)
                pending_prefill_load_chars = len(block_prompt)
                p_node.dispatched_prompt_chars += pending_prefill_load_chars
                p_node.dispatched_req_num += 1
                results_generator = self._wait_to_token_package(
                    p_node,
                    d_node,
                    start_time,
                    block_prompt,
                    sampling_params,
                    multimodal_params,
                    request,
                )
                raw_finish_status = FinishStatus()
                async for sub_req_id, request_output, metadata, raw_finish_status in results_generator:
                    # PD 分离模式下 metadata 中的 token 序号可能不准确，按实际产出计数。
                    assert sub_req_id == block_group_request_id

                    # 收到当前分段的任意输出，说明该请求已经完成 P 节点的 prefill 派发阶段。
                    # 立即归还 selector 中记录的在途 prompt 字符数和请求数，并通过置空确保每段只更新一次。
                    if pending_prefill_load_chars is not None:
                        p_node.dispatched_prompt_chars = max(
                            0, p_node.dispatched_prompt_chars - pending_prefill_load_chars
                        )
                        p_node.dispatched_req_num = max(0, p_node.dispatched_req_num - 1)
                        pending_prefill_load_chars = None

                    if raw_finish_status.is_finished_pd_decode_capacity():
                        # 容量不足状态是 PD 内部分段边界：吞掉模拟结束 token，继续生成剩余 token。
                        break

                    # 容量 marker 已在上方过滤，能走到这里的每个 token 都立即扣减全局剩余输出额度。
                    remaining_max_new_tokens -= 1
                    history_gen_token_strs.append(request_output)
                    prompt_tokens = min(prompt_tokens, metadata["prompt_tokens"])
                    metadata["prompt_tokens"] = prompt_tokens
                    if origin_prompt_cache_len is None:
                        origin_prompt_cache_len = metadata.get("prompt_cache_len", 0)
                        prompt_cache_hit_rate = origin_prompt_cache_len / max(prompt_tokens, 1)
                        self.pd_manager.selector.record_prompt_cache_hit_rate(prompt_cache_hit_rate)
                        if not raw_finish_status.is_error_finished():
                            # 只有收到成功的推理结果后才将 prompt 写入前缀树，避免尚未进入
                            # 推理或已失败的请求被后续请求误判为可复用 cache。
                            self.pd_manager.selector.insert_prompt_cache(prompt, p_node)
                    metadata["prompt_cache_len"] = origin_prompt_cache_len or 0
                    yield origin_request_id, request_output, metadata, raw_finish_status

                await self.remove_req(group_request_id=block_group_request_id)
                segment_index += 1
                # 只有 PD Decode 容量不足产生的内部分段需要续跑；其他状态都结束整个请求。
                if not raw_finish_status.is_finished_pd_decode_capacity():
                    break

        except (ClientDisconnected, BaseException) as e:
            logger.error(f"has exception {str(e)}")

            if isinstance(e, ClientDisconnected):
                logger.warning(f"group_request_id: {origin_request_id} {e.reason}")

            try:
                await self.abort(block_group_request_id, p_node=p_node, d_node=d_node)
            except:
                await self.abort(block_group_request_id)
            raise e

        finally:
            if p_node is not None and pending_prefill_load_chars is not None:
                p_node.dispatched_prompt_chars = max(0, p_node.dispatched_prompt_chars - pending_prefill_load_chars)
                p_node.dispatched_req_num = max(0, p_node.dispatched_req_num - 1)
            await self.remove_req(block_group_request_id)
        return

    async def _merge_choice_generators(self, generators):
        """Merge independent n=1 PD generators into one multi-choice stream."""
        result_queue = asyncio.Queue()
        generator_done = object()

        async def forward_results(generator):
            try:
                async with aclosing(generator):
                    async for result in generator:
                        await result_queue.put(result)

                await result_queue.put(generator_done)
            except BaseException as error:
                await result_queue.put(error)

        tasks = [asyncio.create_task(forward_results(generator)) for generator in generators]
        remaining_generators = len(tasks)

        try:
            while remaining_generators:
                result = await result_queue.get()
                if result is generator_done:
                    remaining_generators -= 1
                elif isinstance(result, BaseException):
                    raise result
                else:
                    yield result
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _wait_for_event_or_disconnect(
        self,
        event: asyncio.Event,
        request: Request,
        timeout: float,
        group_request_id: int,
        stage: str,
    ) -> None:
        """Wait for an asyncio.Event but abort early if the HTTP client disconnects."""
        deadline = time.time() + timeout
        disconnect_reason = f"fetch_pd_stream {stage} period check network disconnected"

        async def raise_if_disconnected() -> None:
            if await request.is_disconnected():
                logger.warning(f"group_request_id: {group_request_id} {disconnect_reason}")
                raise ClientDisconnected(
                    group_request_id=group_request_id,
                    reason=disconnect_reason,
                )

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise ServerBusyError()
            await raise_if_disconnected()
            try:
                await asyncio.wait_for(event.wait(), timeout=min(1.0, remaining))
                await raise_if_disconnected()
                return
            except asyncio.TimeoutError:
                continue

    async def _log_req_header(self, request: Request, group_request_id: int):
        x_request_id = request.headers.get("X-Request-Id", "")
        x_session_id = request.headers.get("X-Session-Id", "")
        format_in_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"received req X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_in_time} "
            f"lightllm_req_id:{group_request_id} "
        )
        return

    async def fetch_pd_stream(
        self,
        p_node: PD_Client_Obj,
        d_node: PD_Client_Obj,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ):
        group_request_id = sampling_params.group_request_id
        sampling_params.pd_master_node_id.initialize(self.args.pd_node_id)

        req_status = ReqStatus(group_request_id, p_node, d_node)
        self.req_id_to_out_inf[group_request_id] = req_status

        up_status_event = req_status.up_status_event
        prefill_prompt_ids_event = req_status.prefill_prompt_ids_event

        old_max_new_tokens = sampling_params.max_new_tokens
        sampling_params.max_new_tokens = 1
        await p_node.websocket.send_bytes(pickle.dumps((ObjType.REQ, (prompt, sampling_params, multimodal_params))))

        try:
            await self._wait_for_event_or_disconnect(
                prefill_prompt_ids_event,
                request,
                timeout=60,
                group_request_id=group_request_id,
                stage="prefill",
            )
        except ServerBusyError:
            logger.warning(f"group_request_id: {group_request_id} wait prefill prompt ids time out")
            raise
        req_status.raise_if_error()

        prompt_ids = prefill_prompt_ids_event.prompt_ids
        logger.info(f"group_request_id: {group_request_id} get prefill prompt ids len {len(prompt_ids)}")

        sampling_params.max_new_tokens = old_max_new_tokens
        await d_node.websocket.send_bytes(
            pickle.dumps((ObjType.REQ, (prompt_ids, sampling_params, MultimodalParams())))
        )

        try:
            await self._wait_for_event_or_disconnect(
                up_status_event,
                request,
                timeout=180,
                group_request_id=group_request_id,
                stage="decode",
            )
        except ServerBusyError:
            logger.warning(f"group_request_id: {group_request_id} wait decode stage time out err, server is busy now.")
            raise
        req_status.raise_if_error()

        # 将 decode 节点上报的当前请求使用的decode节点的信息下发给 p 节点，这样 p 节点才知道将 kv 传输给那个 d 节点。
        upkv_status: PDUpKVStatus = up_status_event.upkv_status
        pd_kv_trans_params: bytes = upkv_status.pd_kv_trans_params
        decode_node_info: PDDecodeNodeInfo = pickle.loads(pd_kv_trans_params)
        await p_node.websocket.send_bytes(
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
                    if output_index == 1:
                        if first_token_gen is False:
                            first_token_gen = True
                            node_run_mode = metadata.pop("node_mode", None)
                            if node_run_mode == "prefill":
                                if old_max_new_tokens != 1 and finish_status.is_finished_length():
                                    finish_status = FinishStatus(FinishStatus.NO_FINISH)
                            metadata["prompt_cache_len"] = prompt_cache_len_from_prefill
                            yield sub_req_id, request_output, metadata, finish_status
                        else:
                            continue
                    else:
                        metadata["prompt_cache_len"] = prompt_cache_len_from_prefill
                        yield sub_req_id, request_output, metadata, finish_status

        return

    async def _wait_for_prefill_token_if_needed(
        self,
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

            for token in new_tokens:
                metadata = token[2]
                if metadata.get("node_mode") == "prefill":
                    prompt_cache_len = metadata.get("prompt_cache_len", 0)
                    await req_status.put_tokens_to_front(new_tokens)
                    return prompt_cache_len

    async def _wait_to_token_package(
        self,
        p_node: PD_Client_Obj,
        d_node: PD_Client_Obj,
        start_time: float,
        prompt: str,
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ):
        if sampling_params.disable_prompt_cache:
            assert False, "pd mode dont support set disable_prompt_cache to True"

        out_token_counter = 0
        first_token_cost_ms = float("inf")
        prompt_cache_len = 0
        group_request_id = sampling_params.group_request_id
        unfinished_count = sampling_params.best_of
        is_first_token = True
        sub_req_id_to_mtp_accepted_token_num: Dict[int, int] = {}
        sub_req_id_to_mtp_verify_step_num: Dict[int, int] = {}

        async for sub_req_id, out_str, metadata, finish_status in self.fetch_pd_stream(
            p_node, d_node, prompt, sampling_params, multimodal_params, request
        ):
            if await request.is_disconnected():
                raise ClientDisconnected(
                    group_request_id=group_request_id, reason="_wait_to_token_package check network disconnected"
                )

            prompt_tokens = metadata["prompt_tokens"]
            out_token_counter += 1
            prompt_cache_len = max(prompt_cache_len, metadata.get("prompt_cache_len", 0))
            sub_req_id_to_mtp_accepted_token_num[sub_req_id] = metadata.get("mtp_accepted_token_num", 0)
            sub_req_id_to_mtp_verify_step_num[sub_req_id] = metadata.get("mtp_verify_step_num", 0)
            if is_first_token:
                first_token_cost_ms = (time.time() - start_time) * 1000
                is_first_token = False
                self.first_time_costs.add(first_token_cost_ms)

            self.latest_success_infer_time = time.time()
            yield sub_req_id, out_str, metadata, finish_status
            if finish_status.is_finished():
                unfinished_count -= 1
            if unfinished_count == 0:
                break

        total_cost_time_ms = (time.time() - start_time) * 1000
        mean_per_token_cost_time_ms = (total_cost_time_ms - first_token_cost_ms) / out_token_counter
        self.per_token_costs.add(mean_per_token_cost_time_ms)
        x_request_id = request.headers.get("X-Request-Id", "")
        x_session_id = request.headers.get("X-Session-Id", "")
        prompt_cache_ratio = prompt_cache_len / prompt_tokens
        mtp_total_verify_steps = sum(sub_req_id_to_mtp_verify_step_num.values())
        if mtp_total_verify_steps <= 0:
            mtp_total_verify_steps = out_token_counter - sum(sub_req_id_to_mtp_accepted_token_num.values())
        mtp_avg_token_per_step = out_token_counter / max(mtp_total_verify_steps, 1)
        format_start_time = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_start_time} "
            f"lightllm_req_id:{group_request_id} first_token_cost:{first_token_cost_ms}ms "
            f"total_cost_time:{total_cost_time_ms}ms,out_token_counter:{out_token_counter} "
            f"mean_per_token_cost_time: {mean_per_token_cost_time_ms}ms "
            f"prompt_token_num:{prompt_tokens} "
            f"prompt_cache_len:{prompt_cache_len} "
            f"prompt_cache_ratio:{prompt_cache_ratio} "
            f"mtp_avg_token_per_step:{mtp_avg_token_per_step} "
        )
        self.metric_client.histogram_observe("lightllm_request_inference_duration", total_cost_time_ms / 1000.0)
        self.metric_client.histogram_observe(
            "lightllm_request_mean_time_per_token_duration", mean_per_token_cost_time_ms / 1000.0
        )
        self.metric_client.histogram_observe("lightllm_request_first_token_duration", first_token_cost_ms / 1000.0)
        self.metric_client.histogram_observe("lightllm_request_generated_tokens", out_token_counter)
        self.metric_client.histogram_observe("lightllm_request_mtp_avg_token_per_step", mtp_avg_token_per_step)
        return

    async def abort(
        self, group_request_id, p_node: Optional[PD_Client_Obj] = None, d_node: Optional[PD_Client_Obj] = None
    ):
        logger.warning(f"aborted group_request_id {group_request_id}")

        try:
            req_status = self.req_id_to_out_inf[group_request_id]
            del self.req_id_to_out_inf[group_request_id]
            p_node = req_status.p_node
            d_node = req_status.d_node
        except:
            pass

        try:
            await p_node.websocket.send_bytes(pickle.dumps((ObjType.ABORT, group_request_id)))
        except:
            pass

        try:
            await d_node.websocket.send_bytes(pickle.dumps((ObjType.ABORT, group_request_id)))
        except:
            pass

        return

    async def remove_req(self, group_request_id):
        try:
            del self.req_id_to_out_inf[group_request_id]
        except:
            pass

    async def timer_log(self):
        while True:
            await asyncio.sleep(30)
            self.first_time_costs.print_log("mean first cost")
            self.per_token_costs.print_log("mean per token cost")

    async def put_to_handle_queue(self, obj):
        await self.infos_queues.put(obj)

    async def handle_loop(self):
        self.infos_queues = AsyncQueue()
        asyncio.create_task(self.timer_log())

        use_config_server = self.args.config_server_host and self.args.config_server_port

        if use_config_server:
            from lightllm.server.httpserver_for_pd_master.register_loop import register_loop

            asyncio.create_task(register_loop(self))

        while True:
            objs = await self.infos_queues.wait_to_get_all_data()

            try:
                for obj in objs:
                    if obj[0] == ObjType.TOKEN_PACKS:
                        token_list, node_load_info = obj[1], obj[2]
                        self.pd_manager.update_node_load_info(node_load_info)

                        for sub_req_id, text, metadata, finish_status in token_list:
                            finish_status: FinishStatus = finish_status
                            group_req_id = convert_sub_id_to_group_id(sub_req_id)
                            try:
                                req_status: ReqStatus = self.req_id_to_out_inf[group_req_id]
                                async with req_status.lock:
                                    req_status.out_token_info_list.append((sub_req_id, text, metadata, finish_status))
                                    req_status.event.set()
                            except:
                                pass
                    elif obj[0] == ObjType.PD_UPLOAD_PREFILL_PROMPT_IDS:
                        _, group_req_id, prompt_ids = obj
                        try:
                            req_status: ReqStatus = self.req_id_to_out_inf[group_req_id]
                            async with req_status.lock:
                                req_status.prefill_prompt_ids_event.prompt_ids = prompt_ids
                                req_status.prefill_prompt_ids_event.set()
                        except:
                            logger.error(
                                f"PD_UPLOAD_PREFILL_PROMPT_IDS fail find req status for group_req_id: {group_req_id}"
                            )
                    elif obj[0] == ObjType.PD_UPLOAD_GENERATE_ERROR:
                        _, group_req_id, error_info = obj
                        logger.error(
                            f"received PD node generate error, group_req_id: {group_req_id}, error: {error_info}"
                        )
                        req_status = self.req_id_to_out_inf.get(group_req_id)
                        if req_status is None:
                            logger.error(
                                f"PD_UPLOAD_GENERATE_ERROR fail find req status for group_req_id: {group_req_id}"
                            )
                        else:
                            await req_status.set_error(error_info)
                    elif obj[0] == ObjType.PD_UPLOAD_SERVER_BUSY:
                        _, group_req_id, error_info = obj
                        logger.warning(
                            f"received PD node server busy, group_req_id: {group_req_id}, reason: {error_info}"
                        )
                        req_status = self.req_id_to_out_inf.get(group_req_id)
                        if req_status is None:
                            logger.error(f"PD_UPLOAD_SERVER_BUSY fail find req status for group_req_id: {group_req_id}")
                        else:
                            await req_status.set_error(error_info, is_server_busy=True)
                    else:
                        logger.error(f"recevie error obj {obj}")
            except BaseException as e:
                logger.exception(str(e))
        return


class ReqStatus:
    def __init__(self, req_id, p_node, d_node) -> None:
        self.req_id = req_id
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.up_status_event = asyncio.Event()
        self.prefill_prompt_ids_event = asyncio.Event()
        self.out_token_info_list: List[Tuple[int, str, dict, FinishStatus]] = []
        self.p_node: PD_Client_Obj = p_node
        self.d_node: PD_Client_Obj = d_node
        self.error_info: Optional[str] = None
        self.is_server_busy = False

    async def wait_to_ready(self):
        try:
            await asyncio.wait_for(self.event.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass

    async def set_error(self, error_info: str, is_server_busy: bool = False):
        async with self.lock:
            self.error_info = error_info
            self.is_server_busy = is_server_busy
            # 请求可能正在等待 Prefill prompt ids、Decode KV 资源或输出 token，
            # 设置全部事件，让请求自己的执行循环立即醒来并抛出异常。
            self.event.set()
            self.up_status_event.set()
            self.prefill_prompt_ids_event.set()

    def raise_if_error(self):
        if self.error_info is not None:
            if self.is_server_busy:
                raise ServerBusyError(self.error_info)
            logger.error(
                f"group_request_id: {self.req_id} detected PD node generate error, "
                f"raise exception to end the request flow early: {self.error_info}"
            )
            raise RuntimeError(f"PD node generate failed: {self.error_info}")

    async def can_read(self, req_id_to_out_inf):
        async with self.lock:
            self.event.clear()
            assert self.req_id in req_id_to_out_inf, f"error state req_id {self.req_id}"
            if len(self.out_token_info_list) == 0:
                return False
            else:
                return True

    async def pop_all_tokens(self):
        async with self.lock:
            ans = self.out_token_info_list.copy()
            self.out_token_info_list.clear()
        return ans

    async def put_tokens_to_front(self, token_list: List[Tuple[int, str, dict, FinishStatus]]):
        if not token_list:
            return

        async with self.lock:
            merged_tokens = token_list + self.out_token_info_list
            self.out_token_info_list.clear()
            self.out_token_info_list.extend(merged_tokens)
            self.event.set()


class PDManager:
    def __init__(self, args: StartArgs):
        self.args: StartArgs = args
        self.prefill_nodes: List[PD_Client_Obj] = []
        self.decode_nodes: List[PD_Client_Obj] = []
        self.url_to_pd_nodes: Dict[str, PD_Client_Obj] = {}
        self.selector = create_selector(args.select_p_d_node_strategy, self)
        return

    def is_pd_nodes_ready(self):
        prefill_node_count = len(self.prefill_nodes)
        decode_node_count = len(self.decode_nodes)
        if self.args.pd_master_mode == "elastic":
            return prefill_node_count >= 1 and decode_node_count >= 1

        try:
            expected_prefill_node_count, expected_decode_node_count = (
                int(node_count) for node_count in self.args.pd_master_mode[:-1].split("p")
            )
            is_ready = (
                prefill_node_count == expected_prefill_node_count and decode_node_count == expected_decode_node_count
            )
            if not is_ready:
                logger.warning(
                    f"PD nodes are not ready: current_prefill={prefill_node_count}, "
                    f"expected_prefill={expected_prefill_node_count}, current_decode={decode_node_count}, "
                    f"expected_decode={expected_decode_node_count}"
                )
            return is_ready
        except ValueError:
            logger.warning(
                f"invalid pd_master_mode={self.args.pd_master_mode!r}; expected 'elastic' or a fixed topology "
                "such as '2p4d'"
            )
            return False

    async def check_pd_nodes_health(self):
        pd_nodes = [*self.prefill_nodes, *self.decode_nodes]
        if not pd_nodes:
            return True

        async with httpx.AsyncClient(timeout=8, trust_env=False) as client:
            results = await asyncio.gather(
                *(client.get(f"http://{node.client_ip_port}/health") for node in pd_nodes),
                return_exceptions=True,
            )

        for node, result in zip(pd_nodes, results):
            if isinstance(result, BaseException):
                logger.warning(f"PD {node.mode} node {node.client_ip_port} health check failed: {str(result)}")
                return False
            if result.status_code != 200:
                logger.warning(
                    f"PD {node.mode} node {node.client_ip_port} health check returned HTTP {result.status_code}"
                )
                return False

        return True

    def register_pd(self, pd_info_json, websocket):
        pd_client = PD_Client_Obj(**pd_info_json)
        client_max_req_total_len = pd_client.start_args["max_req_total_len"]
        if client_max_req_total_len != self.args.max_req_total_len:
            logger.error(
                f"client dont has same max_req_total_len params, but pd master is {self.args.max_req_total_len}"
                f"client is {client_max_req_total_len}"
                f"client info {pd_info_json}"
            )
            assert False

        if pd_client.mode == "prefill":
            for arg_name in ("max_image_pixels", "disable_image_resize"):
                master_value = getattr(self.args, arg_name)
                client_value = pd_client.start_args.get(arg_name)
                if client_value != master_value:
                    error_info = (
                        f"prefill client must use the same {arg_name} as pd master: "
                        f"master={master_value}, client={client_value}, client info={pd_info_json}"
                    )
                    logger.error(error_info)
                    raise ValueError(error_info)

        pd_client.websocket = websocket
        self.url_to_pd_nodes[pd_client.client_ip_port] = pd_client

        if pd_client.mode == "prefill":
            self.prefill_nodes = [e for e in self.prefill_nodes if e.client_ip_port != pd_client.client_ip_port]
            self.prefill_nodes.append(pd_client)
        elif pd_client.mode == "decode":
            self.decode_nodes = [e for e in self.decode_nodes if e.client_ip_port != pd_client.client_ip_port]
            self.decode_nodes.append(pd_client)
        else:
            assert False, f"mode must in ['prefill', 'decode'], but get {pd_client.mode}"

        self.selector.update_nodes(self.prefill_nodes, self.decode_nodes)

        logger.info(f"mode: {pd_client.mode} url: {pd_client.client_ip_port} registed")
        return

    def remove_pd(self, pd_info_json):
        pd_client = PD_Client_Obj(**pd_info_json)

        self.url_to_pd_nodes.pop(pd_client.client_ip_port, None)
        self.prefill_nodes = [e for e in self.prefill_nodes if e.client_ip_port != pd_client.client_ip_port]
        self.decode_nodes = [e for e in self.decode_nodes if e.client_ip_port != pd_client.client_ip_port]

        self.selector.update_nodes(self.prefill_nodes, self.decode_nodes)

        logger.info(f"mode: {pd_client.mode} url: {pd_client.client_ip_port} removed")
        return

    def update_node_load_info(self, load_info: Optional[dict]):
        """更新节点负载信息
        load_info: 节点负载信息字典，内容格式如下，可以为 None
        {
        "total_token_usage_rate": xxxx,
        "client_ip_port": xxxx,
        }
        """
        try:
            if load_info is None:
                return
            client_ip_port = load_info["client_ip_port"]
            total_token_usage_rate = load_info["total_token_usage_rate"]
            pd_client = self.url_to_pd_nodes.get(client_ip_port)
            pd_client.run_status.total_token_usage_rate = total_token_usage_rate
        except BaseException as e:
            logger.warning(f"udpate node load info failed, load_info: {load_info} error: {str(e)}")
        return

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj, PDSelectionExtraInfo]:
        return self.selector.select_p_d_node(prompt, sampling_params, multimodal_params)
