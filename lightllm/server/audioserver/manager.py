import pickle
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import socket
import inspect
import setproctitle
import time
from typing import Dict, List

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs.shm_req_manager import ShmReqManager, StartArgs
from lightllm.server.multimodal_params import AudioItem
from .model_infer.model_rpc import start_model_process, AudioModelRpcClient
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain

logger = init_logger(__name__)


class AudioManager:
    def __init__(
        self,
        args: StartArgs,
        audio_model_rpc_ports,
    ):
        context = zmq.asyncio.Context(2)

        if args.enable_cpu_cache:
            self.send_to_next_module = context.socket(zmq.PUSH)
            self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.multi_level_kv_cache_port}")
        else:
            self.send_to_next_module = context.socket(zmq.PUSH)
            self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.router_port}")

        self.zmq_recv_socket = context.socket(zmq.PULL)
        self.zmq_recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.audio_port}")
        self.cache_client = rpyc.connect("localhost", args.cache_port, config={"allow_pickle": True})
        self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.cache_port = args.cache_port
        self.waiting_reqs: List[GroupReqIndexes] = []
        self.model_weightdir = args.model_dir
        self.tp_world_size = args.tp
        self.audio_dp = args.audio_dp
        self.infer_batch_size = args.audio_infer_batch_size
        self.trust_remote_code = args.trust_remote_code
        self.args = args
        self.audio_model_rpc_ports = audio_model_rpc_ports or [None] * self.audio_dp
        self.shm_req_manager = ShmReqManager()
        self.model_rpcs: List[AudioModelRpcClient] = []
        self.req_stage_times: Dict[int, Dict[str, float]] = {}
        self.next_module_port = args.multi_level_kv_cache_port if args.enable_cpu_cache else args.router_port
        self.waiting_reqs_event = asyncio.Event()

    def _mark_req_stage(self, req_id: int, stage: str):
        now = time.time()
        req_stage_dict = self.req_stage_times.setdefault(req_id, {})
        if "audio_recv" not in req_stage_dict:
            req_stage_dict["audio_recv"] = now
        req_stage_dict[stage] = now
        return now - req_stage_dict["audio_recv"]

    def _log_req_stage(self, req_id: int, stage: str, **kwargs):
        elapsed_s = self._mark_req_stage(req_id, stage)
        extras = " ".join(f"{k}:{v}" for k, v in kwargs.items())
        suffix = f" {extras}" if extras else ""
        logger.info(f"lightllm_req_id:{req_id} stage:{stage} elapsed_ms:{elapsed_s * 1000.0:.3f}{suffix}")
        return

    def _cleanup_req_stage(self, req_id: int):
        self.req_stage_times.pop(req_id, None)
        return

    async def wait_to_model_ready(self):
        self.model_rpcs = []
        for dp_rank_id in range(self.audio_dp):
            rpc_model = await start_model_process(
                world_size=self.audio_dp, port=self.audio_model_rpc_ports[dp_rank_id], device_id=dp_rank_id
            )
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for dp_rank_id in range(self.audio_dp):
            kvargs = {
                "weight_dir": self.model_weightdir,
                "trust_remote_code": self.trust_remote_code,
                "dp_rank_id": dp_rank_id,
                "cache_port": self.cache_port,
                "data_type": self.args.data_type,
            }
            init_model_ret.append(self.model_rpcs[dp_rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)

        warmup_start = time.time()
        logger.info(f"audio_warmup_start audio_dp:{self.audio_dp}")

        async def warmup_one_rank(dp_rank_id: int):
            rank_start = time.time()
            logger.info(f"audio_warmup_rank_start dp_rank_id:{dp_rank_id}")
            await self.model_rpcs[dp_rank_id].warmup_model()
            logger.info(
                f"audio_warmup_rank_done dp_rank_id:{dp_rank_id} elapsed_ms:{(time.time() - rank_start) * 1000.0:.3f}"
            )

        await asyncio.gather(*[warmup_one_rank(dp_rank_id) for dp_rank_id in range(self.audio_dp)])
        logger.info(f"audio_warmup_done elapsed_ms:{(time.time() - warmup_start) * 1000.0:.3f}")
        return

    async def infer_audios(self, audios: List[AudioItem]):
        if len(audios) == 0:
            return

        infer_start = time.time()
        rets = []
        for dp_rank_id in range(self.audio_dp):
            assigned_audios = [audios[i] for i in range(dp_rank_id, len(audios), self.audio_dp)]
            if assigned_audios:
                rets.append(self.model_rpcs[dp_rank_id].encode(assigned_audios))
        await asyncio.gather(*rets)
        logger.info(
            f"audio_infer_batch_done audio_count:{len(audios)} audio_dp:{self.audio_dp} "
            f"elapsed_ms:{(time.time() - infer_start) * 1000.0:.3f}"
        )

        return

    async def loop_for_fwd(self):
        while True:
            if len(self.waiting_reqs) == 0:
                self.waiting_reqs_event.clear()
                if len(self.waiting_reqs) == 0:
                    await self.waiting_reqs_event.wait()
                continue
            else:
                processing_group_reqs = []
                audios_need_infer = []
                while len(self.waiting_reqs) > 0:
                    group_req_indexes = self.waiting_reqs.pop(0)
                    self._log_req_stage(
                        group_req_indexes.group_req_id,
                        "audio_queue_picked",
                        waiting_queue_size=len(self.waiting_reqs),
                    )
                    shm_req = self.shm_req_manager.get_req_obj_by_index(group_req_indexes.shm_req_indexes[0])
                    disable_prompt_cache = shm_req.sample_params.disable_prompt_cache
                    is_aborted = shm_req.is_aborted
                    self.shm_req_manager.put_back_req_obj(shm_req)
                    if is_aborted:
                        # 因为连接断开 aborted 掉的请求也需要传输到后续的模块进行处理
                        # 因为采用 shm 来映射所有的 req 对象以后，引用管理情况复杂了
                        # 需要一些一致的流程来保证不出现异步问题。
                        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                        self._cleanup_req_stage(group_req_indexes.group_req_id)
                        continue

                    multimodal_params = group_req_indexes.multimodal_params

                    audio_uuids = [audio.uuid for audio in multimodal_params.audios]
                    # disable prompt cache通常用来测试，需要也去掉audio cache的影响
                    if disable_prompt_cache:
                        ready_audio = [False] * len(audio_uuids)
                    else:
                        ready_audio = obtain(self.cache_client.root.get_items_embed(audio_uuids))

                    current_req_has_pending_audio = False
                    for audio, ready in zip(multimodal_params.audios, ready_audio):
                        if not ready:
                            audios_need_infer.append(audio)
                            current_req_has_pending_audio = True

                        if len(audios_need_infer) == self.infer_batch_size:
                            batch_reqs = processing_group_reqs + (
                                [group_req_indexes] if current_req_has_pending_audio else []
                            )
                            batch_req_ids = [req.group_req_id for req in batch_reqs]
                            logger.info(
                                f"audio_batch_ready req_ids:{batch_req_ids} "
                                f"audio_count:{len(audios_need_infer)} infer_batch_size:{self.infer_batch_size}"
                            )
                            for batch_req_id in batch_req_ids:
                                self._log_req_stage(
                                    batch_req_id, "audio_infer_start", batch_audio_count=len(audios_need_infer)
                                )
                            await self.infer_audios(audios_need_infer)
                            for batch_req_id in batch_req_ids:
                                self._log_req_stage(
                                    batch_req_id, "audio_infer_done", batch_audio_count=len(audios_need_infer)
                                )
                            audios_need_infer = []
                            for _group_req_indexes in processing_group_reqs:
                                self._log_req_stage(
                                    _group_req_indexes.group_req_id,
                                    "audio_send_to_next_module",
                                    target_port=self.next_module_port,
                                )
                                self.send_to_next_module.send_pyobj(
                                    _group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL
                                )
                                self._cleanup_req_stage(_group_req_indexes.group_req_id)
                            processing_group_reqs = []

                    if len(audios_need_infer) == 0:
                        self._log_req_stage(
                            group_req_indexes.group_req_id,
                            "audio_send_to_next_module",
                            target_port=self.next_module_port,
                            pending_audio_count=0,
                        )
                        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                        self._cleanup_req_stage(group_req_indexes.group_req_id)
                    else:
                        processing_group_reqs.append(group_req_indexes)

                if len(audios_need_infer) > 0:
                    batch_req_ids = [req.group_req_id for req in processing_group_reqs]
                    logger.info(
                        f"audio_batch_ready req_ids:{batch_req_ids} "
                        f"audio_count:{len(audios_need_infer)} infer_batch_size:{self.infer_batch_size}"
                    )
                    for batch_req_id in batch_req_ids:
                        self._log_req_stage(batch_req_id, "audio_infer_start", batch_audio_count=len(audios_need_infer))
                    await self.infer_audios(audios_need_infer)
                    for batch_req_id in batch_req_ids:
                        self._log_req_stage(batch_req_id, "audio_infer_done", batch_audio_count=len(audios_need_infer))
                    for _group_req_indexes in processing_group_reqs:
                        self._log_req_stage(
                            _group_req_indexes.group_req_id,
                            "audio_send_to_next_module",
                            target_port=self.next_module_port,
                        )
                        self.send_to_next_module.send_pyobj(_group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                        self._cleanup_req_stage(_group_req_indexes.group_req_id)
                    processing_group_reqs = []
                    audios_need_infer = []

    async def loop_for_netio_req(self):
        while True:
            recv_req: GroupReqIndexes = await self.zmq_recv_socket.recv_pyobj()
            if isinstance(recv_req, GroupReqIndexes):
                logger.info(
                    f"audio recv req id {recv_req.group_req_id} "
                    f"audio count {len(recv_req.multimodal_params.audios)}"
                )
                self._log_req_stage(
                    recv_req.group_req_id,
                    "audio_recv",
                    audio_count=len(recv_req.multimodal_params.audios),
                    waiting_queue_size=len(self.waiting_reqs),
                )
                self.waiting_reqs.append(recv_req)
                self.waiting_reqs_event.set()
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            if model_rpc.rpc_server_process is not None:
                model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            if model_rpc.rpc_server_process is not None:
                model_rpc.rpc_server_process.join()
        return


def start_audio_process(args, model_rpc_ports, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::audio_server")

    audioserver = AudioManager(args=args, audio_model_rpc_ports=model_rpc_ports)
    try:
        asyncio.run(audioserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        audioserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"AudioServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    loop.create_task(audioserver.loop_for_fwd())
    loop.run_until_complete(audioserver.loop_for_netio_req())
    return
