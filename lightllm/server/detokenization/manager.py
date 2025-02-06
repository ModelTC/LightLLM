import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
import inspect
from lightllm.server.core.objs import ShmReqManager
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.utils.graceful_utils import graceful_registry
from typing import Union, Dict, List
from .decode import decode_token
from .decode_mode_fix import decode_mode_fix
from .decode_req import DecodeReq
from ..tokenizer import get_tokenizer
import pickle
import time
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class DeTokenizationManager:
    def __init__(
        self,
        args,
        eos_id,
        model_weightdir,
        tokenizor_mode,
        detokenization_port,
        detokenization_pub_port,
        trust_remote_code,
    ):
        self.args = args
        context = zmq.asyncio.Context(2)
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"{args.zmq_mode}127.0.0.1:{detokenization_port}")

        self.pub_to_httpserver = context.socket(zmq.PUB)
        self.pub_to_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{detokenization_pub_port}")
        logger.info(f"pub_to_httpserver sendhwm {self.pub_to_httpserver.getsockopt(zmq.SNDHWM)}")
        self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        self.all_special_ids = set(self.tokenizer.all_special_ids)
        self.req_id_to_out: Dict[int, DecodeReq] = {}
        self.eos_id = eos_id
        self._init_get_token_id_to_token_str()
        self.is_decode_mode = self.args.run_mode == "decode"
        self.shm_req_manager = ShmReqManager()

    def _init_get_token_id_to_token_str(self):
        self.token_id_to_token = {token_id: token for token, token_id in self.tokenizer.get_vocab().items()}
        return

    async def handle_loop(self):

        asyncio.create_task(self.timer_to_detoken())

        while True:
            try:
                recv_obj: Union[None, GroupReqIndexes] = await self.recv_from_router.recv_pyobj()

                if isinstance(recv_obj, GroupReqIndexes):
                    for req_index in recv_obj.shm_req_indexes:
                        req = self.shm_req_manager.get_req_obj_by_index(req_index)
                        req.link_prompt_ids_shm_array()
                        req.link_logprobs_shm_array()

                        logger.info(
                            f"detokenization recv req id {req.request_id} "
                            f"cost time {time.time() - recv_obj.time_mark} s"
                        )

                        # p d 分离模式，decode节点的解码需要做一些特殊的修复。
                        decode_req = DecodeReq(req)
                        if self.is_decode_mode:
                            decode_req = decode_mode_fix(decode_req, self.tokenizer, self.eos_id)
                        # token_healing mode 的特殊初始化
                        if self.args.token_healing_mode:
                            decode_req.init_token_healing_prefix_str(self.token_id_to_token, self.tokenizer)

                        self.req_id_to_out[req.request_id] = decode_req

                if recv_obj is None:
                    start_time = time.time()
                    self.gen_token_out()
                    cost_time = (time.time() - start_time) * 1000
                    if cost_time > 50:
                        logger.info(f"detokenize batch cost time {cost_time} ms")

            except Exception as e:
                logger.exception(f"detoken process has exception {str(e)}")

    async def timer_to_detoken(self):
        """
        这个函数是定时去执行 get_token_out() 的定时执行机制，主要是为了在CHUNCK PREFILL的时候，
        有特别长和特别短的请求合并到了一起进行PREFILL，部分请求已经推理出了首个token后，不用等到
        所有的请求都推理出首个token。通过这种定制执行的机制，将写入到shm req中的输出首token将其
        detoken出来，不然需要等待触发detoken的 None 数据包发送过来。这样对首包延迟是及其不友好的。
        """
        while True:
            try:
                start_time = time.time()
                self.gen_token_out()
                cost_time = (time.time() - start_time) * 1000
                if cost_time > 50:
                    logger.info(f"timer detokenize batch cost time {cost_time} ms")
                await asyncio.sleep(0.05)
            except BaseException as e:
                logger.exception(str(e))

    def gen_token_out(self):
        exist_need_detoken = False
        exist_decode = False
        for decode_req in self.req_id_to_out.values():
            if decode_req.need_detoken() and not decode_req.out_queue_is_full():
                new_token_id, src_index = decode_req.get_next_token_id_and_index()
                decode_req.output_ids.append(new_token_id)
                special = new_token_id in self.all_special_ids
                count_output_tokens = len(decode_req.output_ids)

                exist_decode = True
                out_text = decode_token(
                    self.tokenizer,
                    decode_req,
                    int(new_token_id),
                    self.eos_id,
                )
                if out_text.endswith("\ufffd"):
                    new_text = ""
                else:
                    new_text = out_text[len(decode_req.output_str) :]
                    decode_req.output_str = out_text

                    # 对应 token_healing 的特殊处理
                    if self.args.token_healing_mode:
                        if new_text.startswith(decode_req.prefix_str):
                            new_text = new_text[len(decode_req.prefix_str) :]
                            decode_req.prefix_str = ""
                        elif decode_req.prefix_str.startswith(new_text):
                            decode_req.prefix_str = decode_req.prefix_str[len(new_text) :]
                            new_text = ""
                        else:
                            logger.error(
                                f"error token healing state, prefix_str {decode_req.prefix_str} new_text {new_text}"
                            )

                decode_req.req.out_tokens_queue.push(new_text, src_index, special, count_output_tokens)

            if decode_req.need_detoken():
                exist_need_detoken = True

        # 通知 httpserver 进程
        if exist_decode:
            self.pub_to_httpserver.send_pyobj(None, protocol=pickle.HIGHEST_PROTOCOL)

        self.remove_finished_reqs()

        return exist_need_detoken

    def remove_finished_reqs(self):
        finished_reqs: List[DecodeReq] = []
        for decode_req in self.req_id_to_out.values():
            if decode_req.can_set_release_mark():
                finished_reqs.append(decode_req)

        for decode_req in finished_reqs:
            decode_req.req.can_released_mark = True
            logger.info(f"detoken release req id {decode_req.req.request_id}")
            self.shm_req_manager.put_back_req_obj(decode_req.req)
            self.req_id_to_out.pop(decode_req.request_id, None)
        return


def start_detokenization_process(args, detokenization_port, detokenization_pub_port, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        router = DeTokenizationManager(
            args,
            args.eos_id,
            args.model_dir,
            args.tokenizer_mode,
            detokenization_port=detokenization_port,
            detokenization_pub_port=detokenization_pub_port,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        pipe_writer.send(str(e))
        raise
    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(router.handle_loop())
    return
