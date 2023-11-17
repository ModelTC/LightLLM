import asyncio
import numpy as np
import rpyc
import torch
import traceback
from datetime import timedelta
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.server.router.model_infer.infer_batch import InferBatch
from rpyc.utils.classic import obtain

from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama_wquant.model import LlamaTpPartModelWQuant
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder_wquant.model import StarcoderTpPartModelWQuant
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.qwen_wquant.model import QWenTpPartModelWQuant
from lightllm.models.baichuan7b.model import Baichuan7bTpPartModel
from lightllm.models.baichuan13b.model import Baichuan13bTpPartModel
from lightllm.models.baichuan2_7b.model import Baichuan2_7bTpPartModel
from lightllm.models.baichuan2_13b.model import Baichuan2_13bTpPartModel
from lightllm.models.chatglm2.model import ChatGlm2TpPartModel
from lightllm.models.internlm.model import InternlmTpPartModel
from lightllm.models.yi.model import YiTpPartModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from .pre_process import prepare_decode_inputs, prepare_prefill_inputs
from .post_process import sample
from .infer_batch import requests_mapping
from .infer_batch import InferReq
from lightllm.server.io_struct import ReqRunStatus

class ModelRpcServer(rpyc.Service):

    def exposed_init_model(self, kvargs):
        import torch
        import torch.distributed as dist
        world_size = kvargs["world_size"]
        if world_size != 1:
            kvargs = obtain(kvargs)
            world_size = kvargs["world_size"]

        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.cache = {}

        weight_dir = kvargs["weight_dir"]
        max_total_token_num = kvargs["max_total_token_num"]

        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{kvargs["nccl_port"]}', rank=self.tp_rank, world_size=world_size)
        torch.cuda.set_device(self.tp_rank)

        model_cfg, _ = PretrainedConfig.get_config_dict(
            weight_dir
        )

        model_kvargs = {
            "tp_rank": self.tp_rank,
            "world_size": self.world_size,
            "weight_dir": weight_dir,
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5)
        }

        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "bloom":
                self.model = BloomTpPartModel(model_kvargs)
            elif self.model_type == "llama":
                if any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = LlamaTpPartModelWQuant(model_kvargs)
                else:
                    self.model = LlamaTpPartModel(model_kvargs)
            elif self.model_type == "qwen":
                if any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = QWenTpPartModelWQuant(model_kvargs)
                else:
                    self.model = QWenTpPartModel(model_kvargs)
            elif self.model_type == "baichuan":
                if model_cfg['hidden_size'] == 4096:
                    if model_cfg['architectures'][0] == 'BaichuanForCausalLM':
                        self.model = Baichuan2_7bTpPartModel(model_kvargs)
                    else:
                        self.model = Baichuan7bTpPartModel(model_kvargs)
                elif model_cfg["hidden_size"] == 5120:
                    if model_cfg['architectures'][0] == 'BaichuanForCausalLM':
                        self.model = Baichuan2_13bTpPartModel(model_kvargs)
                    else:
                        self.model = Baichuan13bTpPartModel(model_kvargs)
                else:
                    raise Exception('can not support baichuan format')
            elif self.model_type == 'gpt_bigcode':
                if any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = StarcoderTpPartModelWQuant(model_kvargs)
                else:
                    self.model = StarcoderTpPartModel(model_kvargs)
            elif self.model_type == 'chatglm':
                self.model = ChatGlm2TpPartModel(model_kvargs)
            elif self.model_type == 'internlm':
                self.model = InternlmTpPartModel(model_kvargs)
            elif self.model_type == "Yi":
                self.model = YiTpPartModel(model_kvargs)
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            import traceback
            traceback.print_exc()
            raise e
        
        set_random_seed(2147483647)
        return
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_add_batch(self, batch_id, reqs, dtype):
        if self.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), self.model.req_manager, self.model.vocab_size)
        self.cache[batch_id] = batch_data
        return
    
    @calculate_time(show=False, min_cost_ms=300)
    def exposed_prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def exposed_decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        if self.world_size != 1:
            batch_id, req_id_list, finished_req_id_list = obtain(batch_id), obtain(req_id_list), obtain(finished_req_id_list)
        # print("filter old size:", len(batch.reqs), "new size:", len(req_id_list))
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    def exposed_pause_reqs(self, batch_id, req_list):
        if self.world_size != 1:
            batch_id, req_list = obtain(batch_id), obtain(req_list)
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.pause_reqs(req_list)
        self.cache[batch_id] = batch2
        del batch1
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def exposed_remove_batch(self, batch_id):
        batch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        # torch.cuda.empty_cache()
        return
    
    # @calculate_time(show=True, min_cost_ms=150)
    def forward(self, batch_id, is_prefill):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_req_ids, not_run_req_ids = prepare_prefill_inputs(batch)
        else:
            kwargs, run_req_ids, not_run_req_ids = prepare_decode_inputs(batch)
        
        if len(run_req_ids) >= 1:
            logits = self.model.forward(**kwargs)
            next_token_ids, next_token_probs = sample(logits, run_req_ids)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            for r_id, next_token_id, next_token_logprob in zip(run_req_ids, next_token_ids, next_token_logprobs):

                req_obj:InferReq = requests_mapping[r_id]
                if is_prefill and req_obj.req_status == ReqRunStatus.RERUNNING_FROM_OFFLOAD: 
                    # prefill 阶段可能有部分offload的请求重新prefill，代码需要保证所有的请求prefill之后 请求状态都回归为 RUNNING
                    if req_obj.offload_kv_len < req_obj.seq_len:
                        req_obj.req_status = ReqRunStatus.RUNNING
                        req_obj.offload_kv_len = None
                        continue # 恢复正常状态, recompute 如果 offload 长度不是全部长度，则不输出下一个token，因为以前已经输出过了
                    else:
                        req_obj.req_status = ReqRunStatus.RUNNING
                        req_obj.offload_kv_len = None
                elif req_obj.req_status != ReqRunStatus.RUNNING:
                    assert False, "error req status"
                
                req_obj.input_token_ids.append(next_token_id)
                req_obj.seq_len += 1
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }
                output_dict[r_id] = (int(next_token_id), metadata)

        for r_id in not_run_req_ids:
            req_obj:InferReq = requests_mapping[r_id]
            if is_prefill and req_obj.req_status == ReqRunStatus.RERUNNING_FROM_KVKEEP:
                req_obj.req_status = ReqRunStatus.RUNNING # 恢复正常RUNNING状态
            else:
                assert False, "error req status"
        
        self.cache[batch.batch_id] = batch
        return output_dict


class ModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: ModelRpcServer = model_rpc
        self.world_size = world_size
        self.rpc_server_process = rpc_server_process
        self.use_rpc = self.world_size != 1
        if self.use_rpc:
            def async_wrap(f):
                f = rpyc.async_(f)
                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value
                return _func
            self._init_model = async_wrap(self.model.init_model)
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)
            self._decode_batch = async_wrap(self.model.decode_batch)
            self._pause_reqs = async_wrap(self.model.pause_reqs)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
        else:
            self._init_model = self.model.exposed_init_model
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch
            self._decode_batch = self.model.exposed_decode_batch
            self._pause_reqs = self.model.exposed_pause_reqs
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
        return

    async def init_model(self, kvargs):
        ans : rpyc.AsyncResult = self._init_model(kvargs)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs, "fp16")
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def prefill_batch(self, batch_id):
        ans = self._prefill_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def decode_batch(self, batch_id):
        ans = self._decode_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        ans = self._filter_batch(batch_id, req_id_list, finished_req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return 

    async def pause_reqs(self, batch_id, reqs_list):
        ans = self._pause_reqs(batch_id, reqs_list)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def remove_batch(self, batch_id):
        ans = self._remove_batch(batch_id)
        if self.use_rpc:
            await ans
            return
        else:
            return


def _init_env(port):
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(ModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, world_size):
    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcClient(ModelRpcServer(), world_size)
    
    import multiprocessing
    proc = multiprocessing.Process(target=_init_env, args=(port,))
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return ModelRpcClient(con.root, world_size, rpc_server_process=proc)
