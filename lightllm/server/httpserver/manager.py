import zmq
import zmq.asyncio
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from ..tokenizer import get_tokenizer
from ..io_struct import BatchStrOut, AbortReq

class HttpServerManager:
    def __init__(self, model_weightdir, tokenizor_mode, router_port, httpserver_port, total_token_num, max_req_input_len, max_req_total_len, trust_remote_code):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")
        
        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")
        
        self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        
        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)
        
        self.total_token_num = total_token_num
        self.max_req_input_len = max_req_input_len
        self.max_req_total_len = max_req_total_len
        
    
    async def generate(self, prompt, sampling_params, request_id):
        prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) > self.max_req_input_len:
            raise ValueError(f"the input prompt token len is too long > {self.max_req_input_len}")
        req_total_len = len(prompt_ids) + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}")
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}")
        
        self.send_to_router.send_pyobj((prompt_ids, sampling_params, request_id))
        event = asyncio.Event()
        self.req_id_to_out_inf[request_id] =("", {}, False, event)
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            event.clear()
            out_str, metadata, finished, _ = self.req_id_to_out_inf[request_id]
            if len(metadata) != 0:
                self.req_id_to_out_inf[request_id] = ("", {}, finished, event)
                yield out_str, metadata
            if finished:
                try:
                    del self.req_id_to_out_inf[request_id]
                except:
                    pass
                break
        return
    
    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        try:
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return
    
    async def handle_loop(self):
        while True:
            recv_ans:BatchStrOut = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(recv_ans, BatchStrOut), f"error recv type {type(recv_ans)}"
            for req_id, text, metadata, finished, abort in recv_ans.reqs_infs:
                try:
                    if not abort:
                        _, _, _, event = self.req_id_to_out_inf[req_id]
                        self.req_id_to_out_inf[req_id] = (text, metadata, finished, event)
                        event.set()
                    else:
                        del self.req_id_to_out_inf[req_id]
                except:
                    pass
        return
