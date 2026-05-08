import queue
import threading
import time
import rpyc
import socket
import torch
import torch.distributed as dist
from typing import List
from transformers.configuration_utils import PretrainedConfig
from rpyc.utils.classic import obtain
from lightllm.models.whisper.whisper_audio import WhisperAudioModel
from lightllm.models.qwen3_omni_moe_thinker.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from lightllm.server.multimodal_params import AudioItem
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.dist_utils import init_audio_distributed_env
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class AudioModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        kvargs = obtain(kvargs)
        init_audio_distributed_env(kvargs)

        weight_dir = kvargs["weight_dir"]
        self.infer_max_batch_size = kvargs["max_batch_size"]
        self.device_id = kvargs["device_id"]
        self.audio_tp = kvargs["audio_tp"]
        self.dp_rank_id = kvargs["dp_rank_id"]
        self.tp_rank_id = kvargs["tp_rank_id"]
        self.cache_port = kvargs["cache_port"]
        self.data_type = kvargs["data_type"]

        model_cfg, _ = PretrainedConfig.get_config_dict(weight_dir)
        if model_cfg.get("thinker_config") is not None:
            model_cfg = model_cfg["thinker_config"]

        audio_config = model_cfg["audio_config"]
        model_kvargs = {"cache_port": self.cache_port, "data_type": self.data_type}
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
            self.model.check_long_audio_infer()

            self.cache_client = rpyc.connect("localhost", self.cache_port, config={"allow_pickle": True})
            self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.cpu_embed_cache_client = CpuEmbedCacheClient(
                create_meta_data=False,
                init_shm_data=False,
            )
            self._init_taskes()
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            import traceback

            traceback.print_exc()
            raise e

        set_random_seed(2147483647)
        return

    def exposed_run_task(self, audios: List[AudioItem], ref_event_list: List[threading.Event]):
        try:
            audios = obtain(audios)
            for i in range(len(audios)):
                audios[i].event = ref_event_list[i]
                audios[i].start_time = time.time()
                self.infer_queue.put(audios[i])
        except BaseException as e:
            logger.exception(str(e))
            raise e
        return

    def _log_latency(self, audio: AudioItem, stage: str):
        if not hasattr(audio, "start_time"):
            return
        latency = time.time() - audio.start_time
        if latency > 0.02:
            logger.info(f"{stage} latency {latency:.4f} seconds for audio with md5 {audio.md5}")
        audio.start_time = time.time()

    def _init_taskes(self):
        self.infer_queue = queue.Queue()
        self.store_queue = queue.Queue()
        self.sempare = threading.Semaphore(self.infer_max_batch_size * 8)
        self.gloo_group = dist.new_group(ranks=list(range(self.audio_tp)), backend="gloo")

        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()

        self._store_thread = threading.Thread(target=self._store_worker, daemon=True)
        self._store_thread.start()
        return

    def _get_audio_items_from_infer_queue(self, max_num: int, force_same: bool = False) -> List[AudioItem]:
        tasks = []
        self.sempare.acquire()
        task = self.infer_queue.get(block=True)
        tasks.append(task)

        if not force_same:
            while len(tasks) < max_num:
                try:
                    self.sempare.acquire()
                    task = self.infer_queue.get(block=False)
                    tasks.append(task)
                except queue.Empty:
                    self.sempare.release()
                    break
        else:
            while len(tasks) < max_num:
                self.sempare.acquire()
                task = self.infer_queue.get(block=True)
                tasks.append(task)

        return tasks

    def _get_audio_items_from_store_queue(self, max_num: int) -> List[AudioItem]:
        """
        与 visual 的 _get_image_items_from_store_queue 一致：store 队列中单条为 AudioItem，
        按批取出至多 max_num 条。
        """
        tasks = []
        task = self.store_queue.get(block=True)
        tasks.append(task)

        while len(tasks) < max_num:
            try:
                task = self.store_queue.get(block=False)
                tasks.append(task)
            except queue.Empty:
                break

        return tasks

    def _infer_worker(self):
        """
        与 visual _infer_worker 一致：推理后对每个 item 单独放入 store_queue，由 store 线程批处理再 commit。
        """
        torch.cuda.set_device(self.device_id)
        while True:
            try:
                if self.tp_rank_id == 0:
                    audios = self._get_audio_items_from_infer_queue(max_num=self.infer_max_batch_size)
                    dist.broadcast_object_list([len(audios)], src=0, group=self.gloo_group)
                else:
                    ans = [None]
                    dist.broadcast_object_list(ans, src=0, group=self.gloo_group)
                    audios = self._get_audio_items_from_infer_queue(max_num=ans[0], force_same=True)

                for audio in audios:
                    self._log_latency(audio, stage="queue_cost_time")

                all_embeds, audios = self.model.encode(audios)

                self._save_to_cpu_cache(all_embeds=all_embeds, audios=audios)

                # 与 visual _store_to_cpu_cache 相同条入队，便于 store 侧按 infer_max_batch_size 聚合
                for audio in audios:
                    self.store_queue.put(audio)

            except Exception as e:
                logger.exception(str(e))
                raise e

    def _save_to_cpu_cache(self, all_embeds: List[torch.Tensor], audios: List[AudioItem]):
        for _emb, audio in zip(all_embeds, audios):
            assert _emb.shape[0] == audio.token_num, f"audio token num not match {audio.token_num} vs {_emb.shape[0]} "
            self.cpu_embed_cache_client.copy_to_cache(
                embed_tensor=_emb, start_index_in_cache=audio.start_index_in_embed_cache
            )
            audio.cuda_event = torch.cuda.Event()
            audio.cuda_event.record()
        return

    def _commit_to_cpu_cache(self, audios: List[AudioItem]):
        # 与 visual _commit_to_cpu_cache：仅 tp0 通知完成；embed 已在 model.encode 内写入 cache
        if self.tp_rank_id == 0:
            for audio in audios:
                audio.cuda_event.synchronize()
                self._log_latency(audio, stage="inference")

            for audio in audios:
                audio.event.set()
                self._log_latency(audio, stage="set_event")

            self.cache_client.root.set_items_embed([audio.uuid for audio in audios])
            self._log_latency(audios[0], "set_items_embed")

    def _store_worker(self):
        """
        与 visual _store_worker 一致：从 store 队列按批取 AudioItem，再 commit 并释放信号量。
        """
        while True:
            try:
                audios: List[AudioItem] = self._get_audio_items_from_store_queue(max_num=self.infer_max_batch_size)
                self._commit_to_cpu_cache(audios=audios)
                for _ in audios:
                    self.sempare.release()

            except Exception as e:
                logger.exception(str(e))
                raise e
