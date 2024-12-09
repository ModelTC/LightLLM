import os
import asyncio
import numpy as np
import rpyc
import torch
from datetime import timedelta
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.models.cohere.model import CohereTpPartModel
from lightllm.models.mixtral.model import MixtralTpPartModel
from lightllm.models.qwen2.model import Qwen2TpPartModel
from rpyc.utils.classic import obtain

from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder2.model import Starcoder2TpPartModel
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.chatglm2.model import ChatGlm2TpPartModel
from lightllm.models.internlm.model import InternlmTpPartModel
from lightllm.models.stablelm.model import StablelmTpPartModel
from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.internlm2_reward.model import Internlm2RewardTpPartModel
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.models.minicpm.model import MiniCPMTpPartModel
from lightllm.models.llava.model import LlavaTpPartModel
from lightllm.models.qwen_vl.model import QWenVLTpPartModel
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.models.phi3.model import Phi3TpPartModel
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.internvl.model import InternVLLlamaTpPartModel, InternVLPhi3TpPartModel, InternVLQwen2TpPartModel
from lightllm.models.internvl.model import InternVLInternlm2TpPartModel
from lightllm.models.qwen2_vl.model import Qwen2VLTpPartModel
from lightllm.models.qwen2_reward.model import Qwen2RewardTpPartModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams, requests_mapping
from lightllm.server.router.token_load import TokenLoad
from lightllm.common.basemodel.infer_lock import g_infer_state_lock, InferStateLock

import torch.distributed as dist


class ModeBackend:
    def __init__(self) -> None:
        pass

    def init_model(self, kvargs):
        import torch

        self.args = kvargs.get("args", None)
        # p d 分离模式下会有特殊的一些初始化, 所以需要传递
        # 模式参数到模型的初始化过程中进行控制
        self.run_mode = "normal" if self.args is None else self.args.run_mode
        self.is_multimodal = False
        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.dp_size = kvargs.get("dp_size", 1)
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.is_splitfuse_mode = kvargs.get("is_splitfuse_mode", False)
        self.splitfuse_block_size = kvargs.get("splitfuse_block_size", None)
        self.return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        self.use_dynamic_prompt_cache = kvargs.get("use_dynamic_prompt_cache", False)
        self.eos_id: List[int] = kvargs.get("eos_id", [2])
        self.disable_cudagraph = kvargs.get("disable_cudagraph", False)

        self.cache = {}
        self.logger = init_logger(__name__)

        self.weight_dir = kvargs["weight_dir"]
        nccl_port_str = str(kvargs["nccl_port"])
        self.shared_token_load = TokenLoad(f"{nccl_port_str}_shared_token_load", self.dp_size)
        # p d 分离模式，decode节点才会使用的参数
        self.pd_rpyc_port = kvargs.get("pd_rpyc_port", None)
        max_total_token_num = kvargs["max_total_token_num"]

        torch.cuda.set_device(self.tp_rank)

        dist.init_process_group(
            "nccl",
            init_method=f'tcp://127.0.0.1:{kvargs["nccl_port"]}',
            rank=self.tp_rank,
            world_size=self.world_size,
        )
        # warmup nccl communicator
        _a = torch.zeros([1]).to(f"cuda:{self.tp_rank}")
        dist.all_reduce(_a)
        del _a

        from lightllm.distributed import set_custom_reduce

        set_custom_reduce()

        # 为 p d 分离模式添加的全局锁管理，用于做一些同步操作。 一定需要在
        # init_process_group 之后调用
        g_infer_state_lock.obj = InferStateLock(name=nccl_port_str)
        g_infer_state_lock.dp_size = self.dp_size
        self.infer_state_lock = g_infer_state_lock
        # 防止InferStateLock 中的全局共享信息被重复异常初始化,导致同步异常的问题。
        # 所以做一次barrier等待
        dist.barrier()

        model_cfg, _ = PretrainedConfig.get_config_dict(self.weight_dir)

        model_kvargs = {
            "tp_rank": self.tp_rank,
            "world_size": self.world_size,
            "weight_dir": self.weight_dir,
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "is_token_healing": kvargs.get("is_token_healing", False),
            "return_all_prompt_logics": self.return_all_prompt_logprobs,
            "use_dynamic_prompt_cache": self.use_dynamic_prompt_cache,
            "data_type": kvargs.get("data_type", "float16"),
            "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
            "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
            "disable_cudagraph": kvargs.get("disable_cudagraph", False),
            "mem_fraction": kvargs.get("mem_fraction", 0.9),
            "batch_max_tokens": kvargs.get("batch_max_tokens", None),
            "quant_type": kvargs.get("quant_type", None),
            "quant_cfg": kvargs.get("quant_cfg", None),
            "run_mode": self.run_mode,
        }

        try:
            self.model_type = model_cfg.get("model_type", "")
            if self.model_type == "bloom":
                self.model = BloomTpPartModel(model_kvargs)
            elif self.model_type == "llama":
                self.model = LlamaTpPartModel(model_kvargs)
            elif self.model_type == "qwen":
                if "visual" in model_cfg:
                    self.model = QWenVLTpPartModel(model_kvargs)
                    self.is_multimodal = True
                else:
                    self.model = QWenTpPartModel(model_kvargs)
            elif self.model_type == "gpt_bigcode":
                self.model = StarcoderTpPartModel(model_kvargs)
            elif self.model_type == "starcoder2":
                self.model = Starcoder2TpPartModel(model_kvargs)
            elif self.model_type == "chatglm":
                self.model = ChatGlm2TpPartModel(model_kvargs)
            elif self.model_type == "internlm":
                self.model = InternlmTpPartModel(model_kvargs)
            elif self.model_type == "internlm2":
                if model_cfg["architectures"][0] == "InternLM2ForRewardModel":
                    self.model = Internlm2RewardTpPartModel(model_kvargs)
                else:
                    self.model = Internlm2TpPartModel(model_kvargs)
            elif self.model_type == "mistral":
                self.model = MistralTpPartModel(model_kvargs)
            elif self.model_type == "stablelm":
                self.model = StablelmTpPartModel(model_kvargs)
            elif self.model_type == "mixtral":
                self.model = MixtralTpPartModel(model_kvargs)
            elif self.model_type == "minicpm" or model_cfg["architectures"][0] == "MiniCPMForCausalLM":
                self.model = MiniCPMTpPartModel(model_kvargs)
            elif self.model_type == "llava":
                self.model = LlavaTpPartModel(model_kvargs)
                self.is_multimodal = True
            elif self.model_type == "qwen2":
                if model_cfg["architectures"][0] == "Qwen2ForRewardModel":
                    self.model = Qwen2RewardTpPartModel(model_kvargs)
                else:
                    self.model = Qwen2TpPartModel(model_kvargs)
            elif self.model_type == "qwen2_vl":
                self.model = Qwen2VLTpPartModel(model_kvargs)
                self.is_multimodal = True
            elif self.model_type == "gemma":
                self.model = Gemma_2bTpPartModel(model_kvargs)
            elif self.model_type == "cohere":
                self.model = CohereTpPartModel(model_kvargs)
            elif self.model_type == "phi3":
                self.model = Phi3TpPartModel(model_kvargs)
            elif self.model_type == "deepseek_v2":
                self.model = Deepseek2TpPartModel(model_kvargs)
            elif self.model_type == "internvl_chat":
                llm_model_type = model_cfg.get("llm_config").get("model_type")
                if llm_model_type == "phi3":
                    self.model = InternVLPhi3TpPartModel(model_kvargs)
                elif llm_model_type == "internlm2":
                    self.model = InternVLInternlm2TpPartModel(model_kvargs)
                elif llm_model_type == "llama":
                    self.model = InternVLLlamaTpPartModel(model_kvargs)
                elif llm_model_type == "qwen2":
                    self.model = InternVLQwen2TpPartModel(model_kvargs)
                self.is_multimodal = True
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            self.logger.exception(str(e))
            raise e

        set_random_seed(2147483647)
        self.radix_cache = (
            RadixCache(
                str(kvargs["nccl_port"]), self.model.mem_manager.size, self.tp_rank, mem_manager=self.model.mem_manager
            )
            if self.use_dynamic_prompt_cache
            else None
        )

        self.logger.info(f"loaded model class {self.model.__class__}")
        self.init_custom()

        return

    def init_custom(self):
        pass

    def get_max_total_token_num(self):
        return self.model.mem_manager.size

    # @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        raise NotImplementedError()

    # @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        raise NotImplementedError()

    # @calculate_time(show=True, min_cost_ms=0.1)
    def add_batch(self, batch_id, reqs):
        # 当 dp_size 不等于 1 的时候，需要提前从发来的请求参数中
        # 剔除掉不需要处理的请求参数, deepseekv2 这种特殊的模型
        # 在 dp 模式下 tp_rank == dp_rank
        if self.dp_size != 1:
            cur_dp_index = self.tp_rank
            reqs = [req for req in reqs if req["dp_index"] == cur_dp_index]

        g_infer_state_lock.acquire()
        batch_data = InferBatch.init_batch(
            batch_id,
            reqs,
            self.model.data_type,
            torch.cuda.current_device(),
            self.model.req_manager,
            self.model.vocab_size,
            self.radix_cache,
        )
        self.cache[batch_id] = batch_data
        g_infer_state_lock.release()

        # 将更新后的状态返回给调用方用于router中请求的状态
        ans = {}
        for req_id in batch_data.request_ids:
            req_obj: InferReq = requests_mapping[req_id]
            # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数
            ans[req_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [],
                req_obj.finish_status.value,
                None,
            )
        return ans

    # @calculate_time(show=True, min_cost_ms=0.1)
    def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        if self.dp_size != 1:
            req_id_list = [req_id for req_id in req_id_list if req_id in requests_mapping]
            finished_req_id_list = [req_id for req_id in finished_req_id_list if req_id in requests_mapping]

        g_infer_state_lock.acquire()
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        g_infer_state_lock.release()
        return

    def pause_reqs(self, batch_id, req_list):
        if self.dp_size != 1:
            req_list = [req for req in req_list if req[0] in requests_mapping]

        g_infer_state_lock.acquire()
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.pause_reqs(req_list)
        self.cache[batch_id] = batch2
        del batch1
        g_infer_state_lock.release()
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def remove_batch(self, batch_id):
        g_infer_state_lock.acquire()
        batch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        g_infer_state_lock.release()
        return
