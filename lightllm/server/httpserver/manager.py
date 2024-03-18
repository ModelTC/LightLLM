import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import hashlib

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from ..tokenizer import get_tokenizer
from ..io_struct import BatchStrOut, AbortReq, FinishStatus
from ..embed_cache.utils import get_shm_name_data, create_shm


class HttpServerManager:
    def __init__(
        self,
        args,
        router_port,
        cache_port,
        httpserver_port,
        visual_port,
        enable_multimodal,
    ):
        self.args = args
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.enable_multimodal = enable_multimodal
        if self.enable_multimodal:
            self.cache_client = rpyc.connect("localhost", cache_port)
            self.send_to_visual = context.socket(zmq.PUSH)
            self.send_to_visual.connect(f"tcp://127.0.0.1:{visual_port}")

        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")

        self.tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)

        self.total_token_num = args.max_total_token_num
        self.max_req_input_len = args.max_req_input_len
        self.max_req_total_len = args.max_req_total_len

        return

    # connect cache server, calculate md5, alloc resource, return uuid
    async def _alloc_resource(self, data, num):
        md5sum = hashlib.md5(data).hexdigest()
        wait_time = 1
        while True:
            record = self.cache_client.root.alloc(md5sum, num)
            # hit or new
            if record:
                uid = record["id"]
                if not self.cache_client.root.get_item_data(uid):
                    create_shm(get_shm_name_data(uid), data)
                    self.cache_client.root.set_item_data(uid)
                return record
            # cache full
            else:
                await asyncio.sleep(wait_time)
                wait_time = min(wait_time + 2, 9)

    async def _alloc_multimodal_resources(self, multimodal_params):
        for img in multimodal_params.images:
            record = await self._alloc_resource(img.read(), self.tokenizer.image_length)
            img.uuid = record["id"]
            img.token_id = record["token_id"]
            img.token_num = record["token_num"]

    async def _release_multimodal_resources(self, multimodal_params):
        if multimodal_params is not None:
            for img in multimodal_params.images:
                if img.uuid is not None:
                    self.cache_client.root.release(img.uuid)

    async def generate(self, prompt, sampling_params, request_id, multimodal_params):
        if self.enable_multimodal:
            assert len(multimodal_params.images) <= self.args.cache_capacity, "too many images!"
            await self._alloc_multimodal_resources(multimodal_params)
            prompt_ids = self.tokenizer.encode(prompt, multimodal_params)
        else:
            prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)

        if prompt_tokens > self.max_req_input_len:
            # use long_truncation_mode to truncate long input len req.
            if self.args.long_truncation_mode is None:
                raise ValueError(f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}")
            elif self.args.long_truncation_mode == "head":
                prompt_ids = prompt_ids[-self.max_req_input_len :]
                prompt_tokens = len(prompt_ids)
            elif self.args.long_truncation_mode == "center":
                prompt_ids = (
                    prompt_ids[0 : self.max_req_input_len // 2]
                    + prompt_ids[-(self.max_req_input_len - self.max_req_input_len // 2) :]
                )
                prompt_tokens = len(prompt_ids)
                assert prompt_tokens == self.max_req_input_len
            else:
                assert False, "error args"

        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(f"the req token total len + 1 is too long > max_total_token_num:{self.total_token_num}")

        sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        req_status = ReqStatus(request_id, multimodal_params)
        event = req_status.event
        self.req_id_to_out_inf[request_id] = req_status

        if self.enable_multimodal:
            self.send_to_visual.send_pyobj((prompt_ids, sampling_params, multimodal_params, request_id))
        else:
            self.send_to_router.send_pyobj((prompt_ids, sampling_params, multimodal_params, request_id))

        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

            async with req_status.lock:
                event.clear()
                if len(req_status.out_token_info_list) == 0:
                    continue

                for out_str, metadata, finish_status in req_status.out_token_info_list:
                    metadata["prompt_tokens"] = prompt_tokens
                    yield out_str, metadata, finish_status

                    if finish_status.is_finished():
                        try:
                            del self.req_id_to_out_inf[request_id]
                            await self._release_multimodal_resources(multimodal_params)
                        except:
                            pass
                        return
                req_status.out_token_info_list.clear()
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        if self.enable_multimodal:
            self.send_to_visual.send_pyobj(abort_req)
        try:
            req = self.req_id_to_out_inf[request_id]
            await self._release_multimodal_resources(req.multimodal_params)
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return

    async def handle_loop(self):
        while True:
            recv_ans: BatchStrOut = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(recv_ans, BatchStrOut), f"error recv type {type(recv_ans)}"
            for req_id, text, metadata, finish_status in recv_ans.reqs_infs:
                finish_status = FinishStatus(finish_status)
                try:
                    if not finish_status.is_aborted():
                        req_status: ReqStatus = self.req_id_to_out_inf[req_id]
                        async with req_status.lock:
                            req_status.out_token_info_list.append((text, metadata, finish_status))
                            req_status.event.set()
                    else:
                        del self.req_id_to_out_inf[req_id]
                except:
                    pass
        return


class ReqStatus:
    def __init__(self, req_id, multimodal_params) -> None:
        self.req_id = req_id
        self.multimodal_params = multimodal_params
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.out_token_info_list = []
