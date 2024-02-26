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
from lightllm.models.qwen_vl.qwen_visual import QWenVisionTransformer
from lightllm.models.llava.llava_visual import LlavaVisionModel
from lightllm.models.internlm_xcomposer.internlm_visual import InternVisionModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end

class VisualModelRpcServer(rpyc.Service):

    def exposed_init_model(self, kvargs):
        # import torch
        # import torch.distributed as dist
        # world_size = kvargs["world_size"]
        # if world_size != 1:
        #     kvargs = obtain(kvargs)
        #     world_size = kvargs["world_size"]
        # dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{kvargs["nccl_port"]}', rank=self.tp_rank, world_size=world_size)
        # torch.cuda.set_device(self.tp_rank)
        
        weight_dir = kvargs["weight_dir"]
        model_cfg, _ = PretrainedConfig.get_config_dict(
            weight_dir
        )
        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "qwen":
                self.model = QWenVisionTransformer(**model_cfg["visual"]).eval().bfloat16()
            elif self.model_type == "llava":
                self.model = LlavaVisionModel()
            elif self.model_type == "internlmxcomposer2":
                self.model = InternVisionModel()
            else:
                raise Exception(f"can not support {self.model_type} now")
            self.model.load_model(weight_dir)
            self.model = self.model.cuda()
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            import traceback
            traceback.print_exc()
            raise e
        
        set_random_seed(2147483647)
        return
    
    # @calculate_time(show=True, min_cost_ms=150)
    @torch.no_grad()
    def forward(self, images):
        return self.model.encode(images)

    # @calculate_time(show=False, min_cost_ms=300)
    def exposed_encode(self, images):
        return self.forward(images)

class VisualModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: VisualModelRpcServer = model_rpc
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
            self._encode = async_wrap(self.model.encode)
        else:
            self._init_model = self.model.exposed_init_model
            self._encode = self.model.exposed_encode
        return

    async def init_model(self, kvargs):
        ans : rpyc.AsyncResult = self._init_model(kvargs)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def encode(self, images):
        ans = self._encode(images)
        if self.use_rpc:
            return await ans
        else:
            return ans

async def start_model_process(world_size):
    if world_size == 1:
        return VisualModelRpcClient(VisualModelRpcServer(), world_size)

