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
from lightllm.models.llama_awquant.model import LlamaTpPartModelAWQuant
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
from lightllm.models.internlm_wquant.model import InternlmTpPartModelWQuant
from lightllm.models.yi.model import YiTpPartModel
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from .pre_process import prepare_decode_inputs, prepare_prefill_inputs, splitfuse_prepare_decode_inputs
from .post_process import sample
from .infer_batch import requests_mapping
from .infer_batch import InferReq
from lightllm.server.io_struct import ReqRunStatus
from lightllm.utils.log_utils import init_logger


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
        self.is_splitfuse_mode = kvargs.get("is_splitfuse_mode", False)
        self.splitfuse_block_size = kvargs.get("splitfuse_block_size", None)
        self.return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)

        self.cache = {}
        self.logger = init_logger(__name__)

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
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "return_all_prompt_logprobs": self.return_all_prompt_logprobs
        }

        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "bloom":
                self.model = BloomTpPartModel(model_kvargs)
            elif self.model_type == "llama":
                if any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = LlamaTpPartModelWQuant(model_kvargs)
                elif any('int8_activation_weight' in mode_ for mode_ in self.mode):
                    self.model = LlamaTpPartModelAWQuant(model_kvargs)
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
                if any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = InternlmTpPartModelWQuant(model_kvargs)
                else:
                    self.model = InternlmTpPartModel(model_kvargs)
            elif self.model_type == "Yi":
                self.model = YiTpPartModel(model_kvargs)
            elif self.model_type == "mistral":
                self.model = MistralTpPartModel(model_kvargs)
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            self.logger.error(f"load model error: {str(e)} {e} {type(e)}")
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

        # 将更新后的状态返回给调用方用于router中请求的状态
        ans = {}
        for req_id in batch_data.request_ids:
            req_obj : InferReq  = requests_mapping[req_id]
            ans[req_id] = (req_obj.req_status, req_obj.cur_kv_len)
        return ans
    
    @calculate_time(show=False, min_cost_ms=300)
    def exposed_prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def exposed_decode_batch(self, batch_id):
        if self.is_splitfuse_mode:
            return self.splitfuse_forward(batch_id)
        else:
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
        # special code for return all prompt_logprobs
        if self.return_all_prompt_logprobs and is_prefill:
            return self._prefill_to_return_all_prompt_logprobs(batch_id)
        
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs, not_run_reqs = prepare_prefill_inputs(batch)
        else:
            kwargs, run_reqs, not_run_reqs = prepare_decode_inputs(batch)
        
        if len(run_reqs) >= 1:
            logits = self.model.forward(**kwargs)
            next_token_ids, next_token_probs = sample(logits, run_reqs)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
                # prefill and decode is same
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }
                output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata

        for req_obj in not_run_reqs:
            output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None) # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict
    
    @torch.no_grad()
    def _prefill_to_return_all_prompt_logprobs(self, batch_id):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs, not_run_reqs = prepare_prefill_inputs(batch)
        
        if len(run_reqs) >= 1:
            prompt_all_logits = self.model.forward(**kwargs)
            input_ids = kwargs["input_ids"]
            b_start_loc = kwargs["b_start_loc"]
            b_seq_len = kwargs["b_seq_len"]            
            last_index = torch.cumsum(b_seq_len, dim=0, dtype=torch.long) - 1
            logits = prompt_all_logits[last_index, :]

            next_token_ids, next_token_probs = sample(logits, run_reqs)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            
            b_start_loc = b_start_loc.cpu().numpy()
            b_seq_len = b_seq_len.cpu().numpy()
            for req_obj, next_token_id, next_token_logprob, start_loc, seq_len in zip(run_reqs, next_token_ids, next_token_logprobs, b_start_loc, b_seq_len):
                # prefill and decode is same
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }

                cur_ids: torch.Tensor = input_ids[start_loc : start_loc + seq_len]
                cur_logits = prompt_all_logits[start_loc : start_loc + seq_len]
                cur_logprobs = torch.log_softmax(cur_logits, dim=-1, dtype=torch.float)[0:-1, :]
                cur_logprobs = torch.gather(cur_logprobs, dim=1, index=cur_ids[1:].view(-1, 1)).detach().cpu().numpy()

                cur_ids = cur_ids.cpu().numpy()
                all_prompts = []
                for index in range(len(cur_ids) - 1):
                    tmp_dict = {
                        int(cur_ids[index + 1]) : float(cur_logprobs[index, 0])
                    }
                    all_prompts.append([int(cur_ids[index]), tmp_dict])

                metadata["prompt_logprobs"] = all_prompts
                metadata["prompt_token_ids"] = [int(e) for e in cur_ids]
                output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata

        for req_obj in not_run_reqs:
            output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None) # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict

    # @calculate_time(show=True, min_cost_ms=200)
    def splitfuse_forward(self, batch_id):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, decode_reqs, prefill_reqs = splitfuse_prepare_decode_inputs(batch, self.splitfuse_block_size)
        decode_req_num = len(decode_reqs)
        all_reqs = decode_reqs
        all_reqs.extend(prefill_reqs)

        logits = self.model.splitfuse_forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, all_reqs)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        
        index = 0
        for req_obj, next_token_id, next_token_logprob in zip(all_reqs, next_token_ids, next_token_logprobs):
            if index < decode_req_num:
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }
                output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata
            else:
                old_input_token_size = len(req_obj.input_token_ids)
                split_len = min(old_input_token_size - req_obj.cur_kv_len, self.splitfuse_block_size)
                if req_obj.cur_kv_len + split_len == old_input_token_size:
                    # 有输出
                    req_obj.cur_kv_len = old_input_token_size
                    req_obj.input_token_ids.append(next_token_id)
                    req_obj.out_token_id_count[next_token_id] += 1
                    metadata = {
                        'id': int(next_token_id),
                        'logprob': float(next_token_logprob),
                    }
                    output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata)
                elif req_obj.cur_kv_len + split_len < old_input_token_size:
                    # 没输出
                    req_obj.cur_kv_len = req_obj.cur_kv_len + split_len
                    output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None)
                else:
                    assert False, "error state"
            index += 1    

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
            return await ans
        else:
            return ans

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
