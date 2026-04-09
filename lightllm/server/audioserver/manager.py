import zmq
import asyncio
import uvloop
import pickle
import inspect
import setproctitle
import threading
import collections
from typing import List

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager, StartArgs
from lightllm.server.multimodal_params import AudioItem
from .model_infer import start_model_process, AudioModelRpcClient
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.rpyc_fix_utils import connect_embed_cache_rpyc
from rpyc.utils.classic import obtain


logger = init_logger(__name__)


class AudioManager:
    def __init__(
        self,
        args: StartArgs,
    ):
        self.args = args
        context = zmq.Context(2)

        if args.enable_cpu_cache:
            self.send_to_next_module = context.socket(zmq.PUSH)
            self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.multi_level_kv_cache_port}")
        else:
            self.send_to_next_module = context.socket(zmq.PUSH)
            self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.router_port}")

        self.zmq_recv_socket = context.socket(zmq.PULL)
        self.zmq_recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.audio_port}")
        assert args.cache_socket_path is not None
        self.cache_client = connect_embed_cache_rpyc(args.cache_socket_path)
        self.model_weightdir = args.model_dir
        self.audio_dp = args.audio_dp
        self.audio_tp = args.audio_tp
        self.infer_batch_size = args.audio_infer_batch_size
        self.shm_req_manager = ShmReqManager()
        self.lock = asyncio.Lock()

    async def wait_to_model_ready(self):
        self.model_rpcs: List[List[AudioModelRpcClient]] = [[] for _ in range(self.audio_dp)]
        for dp_rank_id in range(self.audio_dp):
            for tp_rank_id in range(self.audio_tp):
                rpc_model = await start_model_process()
                self.model_rpcs[dp_rank_id].append(rpc_model)

        init_model_ret = []
        for dp_rank_id in range(self.audio_dp):
            for tp_rank_id in range(self.audio_tp):
                device_id = self.args.audio_gpu_ids[dp_rank_id * self.audio_tp + tp_rank_id]
                kvargs = {
                    "weight_dir": self.model_weightdir,
                    "device_id": device_id,
                    "audio_tp": self.audio_tp,
                    "cache_socket_path": self.args.cache_socket_path,
                    "tp_rank_id": tp_rank_id,
                    "dp_rank_id": dp_rank_id,
                    "data_type": self.args.data_type,
                    "audio_nccl_port": self.args.audio_nccl_ports[dp_rank_id],
                    "max_batch_size": max(self.infer_batch_size // self.audio_dp, 1),
                }
                init_model_ret.append(self.model_rpcs[dp_rank_id][tp_rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    def get_need_infer_audios(self, group_req_indexes: GroupReqIndexes) -> List[AudioItem]:
        shm_req = self.shm_req_manager.get_req_obj_by_index(group_req_indexes.shm_req_indexes[0])
        is_aborted = shm_req.is_aborted
        disable_prompt_cache = shm_req.sample_params.disable_prompt_cache
        self.shm_req_manager.put_back_req_obj(shm_req)
        if is_aborted:
            return []

        multimodal_params = group_req_indexes.multimodal_params
        audio_uuids = [audio.uuid for audio in multimodal_params.audios]
        if disable_prompt_cache:
            ready_audio = [False] * len(audio_uuids)
        else:
            if len(audio_uuids) > 0:
                ready_audio = obtain(self.cache_client.root.get_items_embed(audio_uuids))
            else:
                ready_audio = []

        audios_need_infer = []
        for audio, ready in zip(multimodal_params.audios, ready_audio):
            if not ready:
                audios_need_infer.append(audio)

        return audios_need_infer

    async def handle_group_indexes(self, group_req_indexes: GroupReqIndexes):
        audios_need_infer = self.get_need_infer_audios(group_req_indexes)

        if len(audios_need_infer) == 0:
            self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
            return
        await self.handle_audios(audios_need_infer)
        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
        return

    async def handle_audios(self, audios_need_infer: List[AudioItem]):
        if not hasattr(self, "cur_dp_index"):
            self.cur_dp_index = 0

        dp_to_handle_audios = collections.defaultdict(list)
        for audio in audios_need_infer:
            self.cur_dp_index += 1
            select_dp = self.cur_dp_index % self.audio_dp
            dp_to_handle_audios[select_dp].append((audio, threading.Event()))

        taskes = []
        for dp_index in range(self.audio_dp):
            _audios = dp_to_handle_audios[dp_index]
            if _audios:
                taskes.append(
                    self.infer_audios(dp_index, audios=[e[0] for e in _audios], events=[e[1] for e in _audios])
                )

        async with self.lock:
            try:
                await asyncio.gather(*taskes)
            except BaseException as e:
                logger.exception(str(e))
                raise e

        for dp_index in range(self.audio_dp):
            _audios = dp_to_handle_audios[dp_index]
            if _audios:
                await asyncio.to_thread(_audios[-1][1].wait)
        return

    async def infer_audios(self, dp_index: int, audios, events):
        taskes = []
        for audio_tp_rank in range(self.audio_tp):
            task = self.model_rpcs[dp_index][audio_tp_rank].run_task(audios, events)
            taskes.append(task)
        await asyncio.gather(*taskes)

    async def loop_for_netio_req(self):
        try:
            while True:
                recv_req: GroupReqIndexes = await asyncio.to_thread(self.zmq_recv_socket.recv_pyobj)
                if isinstance(recv_req, GroupReqIndexes):
                    logger.info(
                        f"audio recv req id {recv_req.group_req_id} "
                        f"audio count {len(recv_req.multimodal_params.audios)}"
                    )
                    asyncio.create_task(self.handle_group_indexes(group_req_indexes=recv_req))
                else:
                    assert False, f"Error Req Inf {recv_req}"
        except Exception as e:
            logger.exception(str(e))

    def clean_up(self):
        return


def start_audio_process(args, pipe_writer):
    import lightllm.utils.rpyc_fix_utils as _

    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::audio_server")
    start_parent_check_thread()
    try:
        audioserver = AudioManager(args=args)
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
    loop.run_until_complete(audioserver.loop_for_netio_req())
    return
