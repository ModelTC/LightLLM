import zmq
import zmq.asyncio
import asyncio
import uvloop
import inspect
import setproctitle
import pickle
import torch
from typing import List
from lightllm.server.core.objs import StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs.x2i_params import X2IParams, X2IResponse, X2ICacheRelease
from .past_kv_cache_client import PastKVCacheClient

logger = init_logger(__name__)

'''
manage a generation service,
1. start x2v pipelines
2. receive generation request from http_server.
3. call llm gen to obtain past key values
4. call x2v to generate images and pass the key values to it
5. return the generated images.
'''

class X2IManager:
    def __init__(
        self,
        args: StartArgs,
    ):
        context = zmq.Context(2)
        self.args = args

        self.zmq_recv_socket = context.socket(zmq.PULL)
        self.zmq_recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.x2i_port}")

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{args.http_server_port_for_x2i}")

        self.waiting_reqs: List[X2IParams] = []

        from lightllm.utils.dist_utils import set_current_device_id

        set_current_device_id(torch.cuda.current_device())

        self.past_kv_cache_client = PastKVCacheClient(only_create_meta_data=False, init_shm_data=True)

    async def wait_to_model_ready(self):
        # from lightx2v import LightX2VPipeline
        # self.gen_pipe = LightX2VPipeline(
        #     model_path = self.args.model_dir,
        #     model_cls = self.args.model_name,
        #     task="t2i"
        # )
        # self.gen_pipe.create_generator(
        #     config_json = self.args.x2v_gen_model_config,
        # )

        pass

    async def loop_for_fwd(self):
        while True:
            try:
                if len(self.waiting_reqs) == 0:
                    await asyncio.sleep(0.01)
                    continue

                x2i_param = self.waiting_reqs.pop(0)

                past_kv_cache = self.past_kv_cache_client.get_kv_cache_for_x2i(
                    x2i_param.past_kvcache.get_all(), x2i_param.past_kvcache.token_len
                )

                past_kv_cache_text = self.past_kv_cache_client.get_kv_cache_for_x2i(
                    x2i_param.past_kvcache_text.get_all(), x2i_param.past_kvcache_text.token_len
                )
                is_t2i = x2i_param.past_kvcache_img.is_empty()

                logger.info(f"past kv cache shape: {past_kv_cache.shape}, past_kv_cache_text shape: {past_kv_cache_text.shape}")

                past_kv_cache_img = None
                if not is_t2i: # t2i
                    past_kv_cache_img = self.past_kv_cache_client.get_kv_cache_for_x2i(
                        x2i_param.past_kvcache_img.get_all(), x2i_param.past_kvcache_img.token_len
                    )

                # release
                self.send_to_httpserver.send_pyobj(
                    X2ICacheRelease(request_id=x2i_param.request_id),
                    protocol=pickle.HIGHEST_PROTOCOL)

                # call generate images
                self.send_to_httpserver.send_pyobj(X2IResponse(
                    request_id=x2i_param.request_id,
                    images=[]),
                    protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                logger.error(e)


    async def loop_for_netio_req(self):
        while True:
            try:
                recv_req: X2IParams = self.zmq_recv_socket.recv_pyobj(zmq.NOBLOCK)
                self.waiting_reqs.append(recv_req)

            except zmq.ZMQError:
                await asyncio.sleep(0.1)

            await asyncio.sleep(0.01)

def start_x2i_process(args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::x2i_server")
    start_parent_check_thread()
    try:
        x2iserver = X2IManager(args=args,)
        asyncio.run(x2iserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        x2iserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"X2IServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    loop.create_task(x2iserver.loop_for_fwd())
    loop.run_until_complete(x2iserver.loop_for_netio_req())
    return
