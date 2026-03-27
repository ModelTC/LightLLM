import asyncio
import rpyc
import socket
import torch
import inspect
from typing import List
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from transformers.configuration_utils import PretrainedConfig
from lightllm.models.whisper.whisper_audio import WhisperAudioModel
from lightllm.models.qwen3_omni_moe_thinker.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from lightllm.server.multimodal_params import AudioItem
from lightllm.utils.infer_utils import set_random_seed
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient
from lightllm.utils.graceful_utils import graceful_registry


class AudioModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        kvargs = obtain(kvargs)
        self.dp_rank_id = kvargs["dp_rank_id"]
        torch.cuda.set_device(self.dp_rank_id)

        weight_dir = kvargs["weight_dir"]
        model_cfg, _ = PretrainedConfig.get_config_dict(weight_dir)
        if model_cfg.get("thinker_config") is not None:
            model_cfg = model_cfg["thinker_config"]

        audio_config = model_cfg["audio_config"]

        model_kvargs = {"cache_port": kvargs["cache_port"], "data_type": kvargs["data_type"]}
        try:
            self.model_type = audio_config["model_type"]
            if self.model_type == "clap_audio_model" or self.model_type == "whisper":
                self.model = WhisperAudioModel(model_kvargs)
            elif self.model_type == "qwen3_omni_moe_audio_encoder":
                self.model = Qwen3OmniMoeAudioEncoder(model_kvargs).eval().bfloat16()
            else:
                raise Exception(f"can not support {self.model_type} now")

            self.model.load_model(weight_dir, model_cfg)
            self.model = self.model.cuda()

            # CpuEmbedCacheClient 的初始化需要依赖这个设置的环境信息。
            from lightllm.utils.dist_utils import set_current_device_id

            set_current_device_id(self.dp_rank_id)

            self.cpu_embed_cache_client = CpuEmbedCacheClient(
                create_meta_data=False,
                init_shm_data=False,
            )
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
        torch.cuda.set_device(self.dp_rank_id)
        audios = obtain(audios)
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
                    return ans.value

                return _func

            self._init_model = async_wrap(self.model.init_model)
            self._encode = async_wrap(self.model.encode)
        else:
            self._init_model = self.model.exposed_init_model
            self._encode = self.model.exposed_encode
        return

    async def init_model(self, kvargs):
        ans = self._init_model(kvargs)
        if self.use_rpc:
            return await ans
        return ans

    async def encode(self, audios: List[AudioItem]):
        ans = self._encode(audios)
        if self.use_rpc:
            return await ans
        return ans


def _init_env(port, device_id):
    graceful_registry(inspect.currentframe().f_code.co_name)
    torch.cuda.set_device(device_id)

    from lightllm.utils.dist_utils import set_current_device_id
    import lightllm.utils.rpyc_fix_utils as _

    set_current_device_id(device_id)
    t = ThreadedServer(AudioModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(world_size, port=None, device_id=None):
    if world_size == 1:
        return AudioModelRpcClient(AudioModelRpcServer(), world_size)

    import multiprocessing

    proc = multiprocessing.Process(target=_init_env, args=(port, device_id))
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            con._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1

    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return AudioModelRpcClient(con.root, world_size, rpc_server_process=proc)
