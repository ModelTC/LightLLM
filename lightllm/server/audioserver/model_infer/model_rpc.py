import asyncio
import rpyc
import torch
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.models.whisper.whisper_audio import WhisperAudioModel
from lightllm.server.multimodal_params import AudioItem
from lightllm.utils.infer_utils import set_random_seed
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient


class AudioModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        # 注册graceful 退出的处理
        from lightllm.utils.graceful_utils import graceful_registry
        import inspect

        graceful_registry(inspect.currentframe().f_code.co_name)

        weight_dir = kvargs["weight_dir"]
        model_cfg, _ = PretrainedConfig.get_config_dict(weight_dir)
        audio_config = model_cfg["audio_config"]

        model_kvargs = {"cache_port": kvargs["cache_port"], "data_type": kvargs["data_type"]}
        try:
            self.model_type = audio_config["model_type"]
            if self.model_type == "clap_audio_model" or self.model_type == "whisper":
                self.model = WhisperAudioModel(model_kvargs)
            else:
                raise Exception(f"can not support {self.model_type} now")

            self.model.load_model(weight_dir, model_cfg)
            self.model = self.model.cuda()

            # CpuEmbedCacheClient 的初始化需要依赖这个设置的环境信息。
            from lightllm.utils.dist_utils import set_current_device_id

            set_current_device_id(torch.cuda.current_device())

            self.cpu_embed_cache_client = CpuEmbedCacheClient(create_meta_data=False, init_shm_data=False)
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
    def forward(self, audios):
        return self.model.encode(audios, cpu_embed_cache_client=self.cpu_embed_cache_client)

    # @calculate_time(show=False, min_cost_ms=300)
    def exposed_encode(self, audios):
        return self.forward(audios)


class AudioModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: AudioModelRpcServer = model_rpc
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
        ans: rpyc.AsyncResult = self._init_model(kvargs)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def encode(self, audios: List[AudioItem]):
        ans = self._encode(audios)
        if self.use_rpc:
            return await ans
        else:
            return ans


async def start_model_process(world_size):
    if world_size == 1:
        return AudioModelRpcClient(AudioModelRpcServer(), world_size)
