import copy
import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from ..io_struct import BatchTokenIdOut, ReqDetokenizationState, BatchStrOut, AbortReq, FinishStatus
from ..req_id_generator import convert_sub_id_to_group_id
from typing import Union
from .decode import decode_token
from .decode_mode_fix import decode_mode_fix
from ..tokenizer import get_tokenizer
import pickle
import time
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
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
        httpserver_port,
        trust_remote_code,
    ):
        self.args = args
        context = zmq.asyncio.Context(2)
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{detokenization_port}")

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://127.0.0.1:{httpserver_port}")

        self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        self.all_special_ids = set(self.tokenizer.all_special_ids)
        self.req_id_to_out = {}
        self.eos_id = eos_id
        self._init_get_token_id_to_token_str()
        self.is_decode_mode = self.args.run_mode == "decode"

    def _init_get_token_id_to_token_str(self):
        self.token_id_to_token = {token_id: token for token, token_id in self.tokenizer.get_vocab().items()}
        return

    async def handle_loop(self):
        while True:
            try:
                recv_obj: Union[
                    BatchTokenIdOut, ReqDetokenizationState, AbortReq
                ] = await self.recv_from_router.recv_pyobj()
                start_time = time.time()
                assert isinstance(
                    recv_obj, (BatchTokenIdOut, ReqDetokenizationState, AbortReq)
                ), f"type is not right {type(recv_obj)}"
                if isinstance(recv_obj, ReqDetokenizationState):
                    if self.is_decode_mode:
                        recv_obj = decode_mode_fix(recv_obj, self.tokenizer, self.eos_id)

                    # 将解序列对象复制 best_of 份， 并为其生成请求id
                    for delta_id in range(recv_obj.best_of):
                        recv_obj.request_id = recv_obj.group_req_id + delta_id
                        self.req_id_to_out[recv_obj.request_id] = copy.deepcopy(recv_obj)

                if isinstance(recv_obj, AbortReq):
                    delete_group_req_id = recv_obj.group_req_id
                    del_sub_req_ids = []
                    for sub_req_id in self.req_id_to_out.keys():
                        if convert_sub_id_to_group_id(sub_req_id) == delete_group_req_id:
                            del_sub_req_ids.append(sub_req_id)
                    try:
                        for del_id in del_sub_req_ids:
                            del self.req_id_to_out[del_id]
                    except:
                        pass

                if isinstance(recv_obj, BatchTokenIdOut):
                    new_batch_str_out = BatchStrOut()
                    for req_id, new_token_id, new_gen_metadata, finish_status in recv_obj.reqs_infs:
                        if req_id not in self.req_id_to_out:
                            continue

                        req_out: ReqDetokenizationState = self.req_id_to_out[req_id]
                        req_out.output_ids.append(new_token_id)
                        new_gen_metadata["special"] = new_token_id in self.all_special_ids
                        new_gen_metadata["count_output_tokens"] = len(req_out.output_ids)
                        req_out.gen_metadata.update(new_gen_metadata)

                        out_text = decode_token(
                            self.tokenizer,
                            req_out,
                            new_token_id,
                            self.eos_id,
                        )

                        if out_text.endswith("\ufffd"):
                            new_text = ""
                        elif "prefix_str" in new_gen_metadata:
                            # 对应 token_healing 的特殊处理
                            token = new_gen_metadata["prefix_str"]
                            token_str = self.tokenizer.convert_tokens_to_string([token])
                            new_text = out_text[len(req_out.output_str) :]
                            logger.info(
                                f"token headling prefix_token_and_str: '{token}':'{token_str}' new_text: '{new_text}'"
                            )
                            assert new_text.startswith(token_str)
                            new_text = new_text[len(token_str) :]
                            req_out.output_str = out_text
                        else:
                            new_text = out_text[len(req_out.output_str) :]
                            req_out.output_str = out_text
                        new_batch_str_out.reqs_infs.append((req_id, new_text, new_gen_metadata, finish_status))
                        if FinishStatus(finish_status).is_finished():
                            try:
                                del self.req_id_to_out[req_id]
                            except:
                                pass
                    self.send_to_httpserver.send_pyobj(new_batch_str_out, protocol=pickle.HIGHEST_PROTOCOL)
                    cost_time = (time.time() - start_time) * 1000
                    if cost_time > 50:
                        logger.info(f"detokenize batch cost time {cost_time} ms")
            except Exception as e:
                logger.exception(f"detoken process has exception {str(e)}")


def start_detokenization_process(args, detokenization_port, httpserver_port, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        router = DeTokenizationManager(
            args,
            args.eos_id,
            args.model_dir,
            args.tokenizer_mode,
            detokenization_port=detokenization_port,
            httpserver_port=httpserver_port,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        pipe_writer.send(str(e))
        raise
    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(router.handle_loop())
    return
