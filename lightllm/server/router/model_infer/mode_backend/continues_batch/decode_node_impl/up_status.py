import time
import json
import asyncio
import threading
import websockets
from typing import List
from dataclasses import asdict
from lightllm.server.pd_io_struct import UpKVStatus
from lightllm.utils.log_utils import init_logger
import torch.multiprocessing as mp

logger = init_logger(__name__)


class UpStatusManager:
    def __init__(self, args, task_in_queue: mp.Queue, task_out_queue: mp.Queue):
        self.args = args
        self.task_queue: mp.Queue[UpKVStatus] = task_in_queue
        self.task_out_queue = task_out_queue
        self.daemon_thread = threading.Thread(target=self.thread_loop, daemon=True)
        self.daemon_thread.start()

    def thread_loop(self):
        asyncio.run(self.loop())

    async def loop(self):
        while True:
            try:
                uri = f"ws://{self.args.pd_master_ip}:{self.args.pd_master_port}/kv_move_status"
                async with websockets.connect(uri) as websocket:
                    while True:
                        try:
                            loop = asyncio.get_event_loop()
                            upkv_status: UpKVStatus = await loop.run_in_executor(None, self.task_queue.get)
                            await websocket.send(json.dumps(asdict(upkv_status)))
                            logger.info(f"up status: {upkv_status}")
                            # self.task_out_queue.put("ok")
                        except BaseException as e:
                            logger.error(str(e))
                            # self.task_out_queue.put("fail")
                            raise e

            except Exception as e:
                logger.error(f"connetion to pd_master has error: {str(e)}")
                logger.exception(str(e))
                await asyncio.sleep(10)
                logger.info("reconnection to pd_master")


def _init_env(args, task_in_queue: mp.Queue, task_out_queue: mp.Queue):
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)
    up_kv_manager = UpStatusManager(args, task_in_queue, task_out_queue)
    logger.info(f"up kv manager {str(up_kv_manager)} start ok")
    while True:
        time.sleep(10)
    return


def start_up_kv_status_process(args, task_in_queue: mp.Queue, task_out_queue: mp.Queue):
    proc = mp.Process(target=_init_env, args=(args, task_in_queue, task_out_queue))
    proc.start()
    assert proc.is_alive()
    logger.info("up_kv_status_process start")
    return proc
