import queue
import threading
import torch.distributed as dist
import torch
from typing import List, Any, Deque, Tuple
from .model_rpc import VisualModelRpcServer
from lightllm.server.multimodal_params import ImageItem
from lightllm.server.embed_cache.afs_utils import SepEmbedHandler
from rpyc.utils.server import ThreadedServer
from lightllm.utils.envs_utils import get_env_start_args
from rpyc.utils.classic import obtain
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

class VisualOnlyModelRpcServer(VisualModelRpcServer):
    """
    完善这个代码:
    1. 创建一个队列, 用于接受别人放入的task,
    2. 创建一个线程，从队列中取出任务，完成后，修改task中的event，让放入的人得到结果和通知。这是任务循环。
    3. 能不能封装比较易读的流程。
    """

    def __init__(self):
        super().__init__()




    def exposed_run_task(self, images: List["ImageItem"], ref_event_list: List[threading.Event]):
        try:
            images = obtain(images)
            for i in range(len(images)):
                images[i].event = ref_event_list[i]
                self.infer_queue.put(images[i])

        except BaseException as e:
            logger.exception(str(e))
            raise e
        return
    
    def init_taskes(self):
        # 控制每次的最大推理图片数量，防止爆显存
        self.max_infer_batch_size = get_env_start_args().visual_infer_batch_size

        # 异步队列, 用于接受任务
        self.infer_queue = queue.Queue()
        self.infer_queue_lock = threading.Lock()
        # 将计算得到的结果放入 afs 或者 embed cache 的 queue
        self.store_queue = queue.Queue()

        # 限制并发, 主要控制内存用量，防止过多造成内存OOM
        self.sempare = threading.Semaphore(self.max_infer_batch_size * 8)

        # 用于同步各个推理tp每次拿到一样的image数量建立的gloo通信组
        self.gloo_group = dist.new_group(ranks=list(range(self.vit_tp)), backend="gloo")

        self.afs_handler = SepEmbedHandler(
            afs_embed_dir=get_env_start_args().afs_embed_dir,
            redis_host=get_env_start_args().config_server_host,
            redis_port=get_env_start_args().config_server_vit_redis_port,
            capacity=get_env_start_args().afs_embed_capacity,
        )

        # 启动任务处理线程
        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()

        self._store_thread = threading.Thread(target=self._store_worker, daemon=True)
        self._store_thread.start()
        pass


    def _get_image_items_from_infer_queue(self, max_num: int, force_same: bool = False) -> List[ImageItem]:
        """
        从队列中批量获取任务，直到达到 max_num 或队列为空。
        """
        tasks = []
        # 至少获取一个任务，阻塞
        self.sempare.acquire()
        task = self.infer_queue.get(block=True)
        tasks.append(task)  
        
        if not force_same:
            # 尝试继续获取更多任务，直到达到 max_num
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
    
    def _get_image_items_from_store_queue(self, max_num: int) -> List[ImageItem]:
        """
        从队列中批量获取任务，直到达到 max_num 或队列为空。
        """
        tasks = []
        # 至少获取一个任务，阻塞
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
        任务处理循环: 从队列中取出任务, 执行完成后通知调用者
        """
        torch.cuda.set_device(self.device_id)
        while True:
            try:
                # 从队列获取任务, 阻塞等待
                if self.tp_rank_id == 0:
                    images = self._get_image_items_from_infer_queue(max_num=self.max_infer_batch_size)
                    dist.broadcast_object_list([len(images)], src=0, group=self.gloo_group)
                else:
                    ans = [None]
                    dist.broadcast_object_list(ans, src=0, group=self.gloo_group)
                    images = self._get_image_items_from_infer_queue(max_num=ans[0], force_same=True)

                # 执行任务: 调用父类的forward方法处理图像
                all_img_embeds, uuids, valid_ids = self.forward(images)
                all_img_embeds = all_img_embeds.to(torch.device("cuda"))

                if self.is_visual_only_mode:
                    all_img_embeds = all_img_embeds.detach().cpu()
                    for image, valid_id in zip(images, valid_ids):
                        start, end = valid_id
                        gen_embed = all_img_embeds[start:end]
                        image.gen_embed = gen_embed
                        self.store_queue.put(image)
                else:
                    self._store_to_cpu_cache(all_img_embeds, valid_ids, images)
                all_img_embeds = all_img_embeds.detach().cpu()
                for image, valid_id in zip(images, valid_ids):
                    start, end = valid_id
                    self.put_afs_queue.put((image, all_img_embeds[start:end]))

            except Exception as e:
                logger.exception(str(e))
                raise e
            
    def _store_to_cpu_cache(self, all_img_embeds, valid_ids, images):
        for i in range(len(images)):
            start, end = valid_ids[i]
            image = images[i]
            if self.tp_rank_id == 0:
                self.cpu_embed_cache_client.copy_vision_to_cache(
                    embed_tensor=all_img_embeds[start:end], start_index_in_cache=image.start_index_in_embed_cache
                )
            cuda_event = torch.cuda.Event()
            cuda_event.record()
            image.cuda_event = cuda_event
            self.store_queue.put(image)
        
    def _store_worker(self):
        """
        任务处理循环: 从队列中取出ImageItem和embed 放入 afs中, 执行完成后通知调用者
        """
        while True:
            try:
                # 从队列获取任务, 阻塞等待
                images: List[ImageItem] = self._get_image_items_from_store_queue(max_num=self.max_infer_batch_size)
                # 只有 0 rank 执行真的写入操作。
                if self.tp_rank_id == 0:
                    if self.is_visual_only_mode:
                        for image in images:
                            self.afs_handler.insert(image.md5, image.gen_embed)
                            image.event.set()
                    else:
                        for image in images:
                            # 等待拷贝到cpu cache 完成。
                            image.cuda_event.synchronize()

                        uuids = [image.uuid for image in images]
                        self.cache_client.root.set_items_embed(uuids)

                        for image in images:
                            image.event.set()
            
                for _ in images:
                    self.sempare.release()
            except Exception as e:
                logger.exception(str(e))
                raise e