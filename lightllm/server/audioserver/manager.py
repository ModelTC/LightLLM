import pickle
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import inspect
import setproctitle
from typing import List

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs.shm_req_manager import ShmReqManager
from lightllm.server.multimodal_params import AudioItem
from .model_infer.model_rpc import start_model_process, AudioModelRpcClient
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain

logger = init_logger(__name__)


class AudioManager:
    def __init__(
        self,
        args,
        router_port,
        audio_port,
        cache_port,
        infer_batch_size=4,
    ):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"{args.zmq_mode}127.0.0.1:{router_port}")

        self.recv_from_visualserver = context.socket(zmq.PULL)
        self.recv_from_visualserver.bind(f"{args.zmq_mode}127.0.0.1:{audio_port}")
        self.cache_client = rpyc.connect("localhost", cache_port, config={"allow_pickle": True})
        self.cache_port = cache_port
        self.waiting_reqs: List[GroupReqIndexes] = []
        self.model_weightdir = args.model_dir
        self.tp_world_size = args.tp
        self.world_size = 1
        self.infer_batch_size = infer_batch_size
        self.trust_remote_code = args.trust_remote_code
        self.args = args
        self.shm_req_manager = ShmReqManager()

    async def wait_to_model_ready(self):

        self.model_rpcs: List[AudioModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):
            kvargs = {
                "weight_dir": self.model_weightdir,
                "trust_remote_code": self.trust_remote_code,
                "rank_id": rank_id,
                "cache_port": self.cache_port,
                "data_type": self.args.data_type,
            }
            init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    async def infer_audios(self, audios: List[AudioItem]):
        if len(audios) == 0:
            return

        rets = [self.model_rpcs[tp_rank].encode(audios) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)

        return

    async def loop_for_fwd(self):
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
            else:
                processing_group_reqs = []
                audios_need_infer = []
                while len(self.waiting_reqs) > 0:
                    group_req_indexes = self.waiting_reqs.pop(0)
                    shm_req = self.shm_req_manager.get_req_obj_by_index(group_req_indexes.shm_req_indexes[0])
                    disable_prompt_cache = shm_req.sample_params.disable_prompt_cache
                    is_aborted = shm_req.is_aborted
                    self.shm_req_manager.put_back_req_obj(shm_req)
                    if is_aborted:
                        # 因为连接断开 aborted 掉的请求也需要传输到后续的模块进行处理
                        # 因为采用 shm 来映射所有的 req 对象以后，引用管理情况复杂了
                        # 需要一些一致的流程来保证不出现异步问题。
                        self.send_to_router.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                        continue

                    multimodal_params = group_req_indexes.multimodal_params

                    audio_uuids = [audio.uuid for audio in multimodal_params.audios]
                    # disable prompt cache通常用来测试，需要也去掉audio cache的影响
                    if disable_prompt_cache:
                        ready_audio = [False] * len(audio_uuids)
                    else:
                        ready_audio = obtain(self.cache_client.root.get_items_embed(audio_uuids))

                    for audio, ready in zip(multimodal_params.audios, ready_audio):
                        if not ready:
                            audios_need_infer.append(audio)

                        if len(audios_need_infer) == self.infer_batch_size:
                            await self.infer_audios(audios_need_infer)
                            audios_need_infer = []
                            for _group_req_indexes in processing_group_reqs:
                                self.send_to_router.send_pyobj(_group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                            processing_group_reqs = []

                    if len(audios_need_infer) == 0:
                        self.send_to_router.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        processing_group_reqs.append(group_req_indexes)

                if len(audios_need_infer) > 0:
                    await self.infer_audios(audios_need_infer)
                    for _group_req_indexes in processing_group_reqs:
                        self.send_to_router.send_pyobj(_group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                    processing_group_reqs = []
                    audios_need_infer = []

    async def loop_for_netio_req(self):
        while True:
            recv_req: GroupReqIndexes = await self.recv_from_visualserver.recv_pyobj()
            if isinstance(recv_req, GroupReqIndexes):
                self.waiting_reqs.append(recv_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_audio_process(args, router_port, audio_port, cache_port, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::audio_server")

    try:
        audioserver = AudioManager(args, router_port, audio_port, cache_port)
        asyncio.run(audioserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        audioserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"VisualServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    loop.create_task(audioserver.loop_for_fwd())
    loop.run_until_complete(audioserver.loop_for_netio_req())
    return
