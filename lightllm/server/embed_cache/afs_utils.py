import os
import time
import torch
import uuid
from typing import List, Tuple, Optional
from pathlib import Path
from .redis_utils import RedisMetadataClient

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class AfsUtils:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        # 判断 base_dir 是否存在，不存在则创建并赋予777权限，让其他人也可以写入
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            os.chmod(base_dir, 0o777)
        return

    def _get_afs_path(self, name: str) -> Path:
        return Path(self.base_dir) / name

    def save_tensor_afs(self, name: str, tensor: torch.Tensor) -> None:
        target_path = self._get_afs_path(name)

        try:
            with open(target_path, "wb") as f:
                tensor = tensor.detach().cpu()
                dest = torch.empty_like(tensor)
                dest.copy_(tensor)
                torch.save(dest, f, _use_new_zipfile_serialization=False, pickle_protocol=4)

            os.chmod(target_path, 0o777)
        except Exception as e:
            try:
                target_path.unlink(missing_ok=True)
            except Exception:
                pass
            logger.exception(f"failed to save embed tensor file: {target_path} excetion {str(e)}")
            raise e

    def load_tensor_afs(self, name: str) -> torch.Tensor:
        path = self._get_afs_path(name)
        with open(path, "rb") as f:
            return torch.load(f, weights_only=False)

    def free_afs(self, name: str) -> None:
        path = self._get_afs_path(name)
        path.unlink(missing_ok=True)
        return


class SepEmbedManager:
    def __init__(
        self,
        afs_embed_dir: str,
        redis_host: str,
        redis_port: int,
        capacity: int = 50000,
        evict_fraction: float = 0.1,
    ) -> None:
        if not (0.0 <= evict_fraction <= 1.0):
            raise ValueError("evict_fraction must be 0..1")
        if capacity < 1:
            raise ValueError("capacity must be >=1")

        redis_url = f"redis://{redis_host}:{redis_port}/0"
        self.redis_client = RedisMetadataClient(redis_url=redis_url)
        self.capacity = capacity
        self.remove_count = min(int(self.capacity * evict_fraction), 1000)  # full的时候，每次清理的数量
        self.afs_embed_dir = afs_embed_dir
        self.afs_utils = AfsUtils(self.afs_embed_dir)

    def full_to_clean(self):
        remove_objs: List[str] = self.redis_client.get_eviction_candidates(
            remove_size=self.remove_count, capcity=self.capacity
        )
        for obj in remove_objs:
            _token = str(uuid.uuid4())
            try:
                if self.redis_client.acquire_lock(md5=obj, token=_token, time_out=10):
                    if self.redis_client.remove_ready(md5=obj, token=_token)[0]:
                        self.afs_utils.free_afs(obj)
                    self.redis_client.release_lock(md5=obj, token=_token)
            except BaseException as e:
                logger.warning(f"full_to_clean md5 {obj} error {str(e)}")

    def insert(self, md5: str, tensor: torch.Tensor) -> bool:
        for _ in range(3):
            if self._insert(md5, tensor):
                return True
            else:
                time.sleep(30)
        return False

    def _insert(self, md5: str, tensor: torch.Tensor) -> bool:
        self.full_to_clean()
        try:
            _token = str(uuid.uuid4())
            if self.redis_client.acquire_lock(md5=md5, token=_token, time_out=30):
                self.afs_utils.save_tensor_afs(md5, tensor)
                ret = self.redis_client.mark_ready(md5=md5, token=_token)
                if ret[0]:
                    self.redis_client.release_lock(md5=md5, token=_token)
                    return True
                else:
                    self.redis_client.release_lock(md5=md5, token=_token)
                    logger.warning(f"insert {md5} failed error {ret[1]}")
                    return False
        except:
            return False

    def query_to_lock(self, md5: str) -> Optional[str]:
        """
        返回 None, 或者 token, 返回token代表可以去afs中读取数据了，
        """
        try:
            _token = str(uuid.uuid4())
            if self.redis_client.acquire_lock(md5=md5, token=_token, time_out=60):
                ret = self.redis_client.check_ready_and_touch(md5=md5, token=_token)
                if ret[0]:
                    return _token
                else:
                    logger.warning(f"query_to_lock {md5} failed  {ret[1]}")
                    self.redis_client.release_lock(md5=md5, token=_token)
        except:
            try:
                self.redis_client.release_lock(md5=md5, token=_token)
            except:
                pass
        return None
