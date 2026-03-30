import os
import time
import torch
import uuid
import itertools
from typing import List, Tuple, Optional
from pathlib import Path
from .redis_utils import RedisMetadataLib

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class AfsUtils:
    def __init__(self, base_dir: str, dir_depth: int = 2):
        self.base_dir = base_dir
        # 判断 base_dir 是否存在，不存在则创建并赋予777权限，让其他人也可以写入
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, mode=0o777, exist_ok=True)

        # build sub dirs
        parent_dir = Path(base_dir)
        subdirs = ["".join(p) for p in itertools.product("0123456789abcdef", repeat=dir_depth)]
        for sub in subdirs:
            sub_dir_path = parent_dir / sub
            os.makedirs(sub_dir_path, mode=0o777, exist_ok=True)
        return

    def save_tensor_afs(self, name: str, tensor: torch.Tensor) -> bool:
        try:
            target_path = self._get_afs_path(name)
            if target_path.exists():
                return True
            tmp_path = self._get_afs_path(name=name, uuid_tail_str=str(uuid.uuid4()))
            with open(tmp_path, "wb") as f:
                tensor = tensor.detach().cpu()
                dest = torch.empty_like(tensor)
                dest.copy_(tensor)
                torch.save(dest, f, _use_new_zipfile_serialization=False, pickle_protocol=4)
            os.rename(tmp_path, target_path)
            os.chmod(target_path, 0o777)
            return True
        except Exception as e:
            logger.warning(f"failed to save embed tensor file: {target_path} tmp_path: {tmp_path} excetion {str(e)}")
            return False
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except:
                pass

    def load_tensor_afs(self, name: str) -> Optional[torch.Tensor]:
        try:
            path = self._get_afs_path(name)
            with open(path, "rb") as f:
                return torch.load(f, weights_only=False)
        except Exception as e:
            logger.warning(f"fail to load afs file {name} error: {str(e)}")
            return None

    def free_afs(self, name: str) -> bool:
        try:
            path = self._get_afs_path(name)
            if not path.exists():
                return True
            path.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.warning(f"free_afs name: {name} error: {str(e)}")
            return False
        return

    def exist_afs(self, name: str) -> bool:
        try:
            path = self._get_afs_path(name)
            return path.exists()
        except Exception as e:
            logger.warning(f"exist_afs name: {name} error: {str(e)}")
            return False

    def _get_afs_path(self, name: str, uuid_tail_str: Optional[str] = None) -> Path:
        if uuid_tail_str is None:
            return Path(self.base_dir) / name[0:2] / name
        else:
            return Path(self.base_dir) / name[0:2] / f"{name}.{uuid_tail_str}"


class SepEmbedHandler:
    def __init__(
        self,
        afs_embed_dir: str,
        redis_host: str,
        redis_port: int,
        capacity: int = 250000,
        evict_fraction: float = 0.1,
    ) -> None:
        if not (0.0 <= evict_fraction <= 1.0):
            raise ValueError("evict_fraction must be 0..1")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")

        redis_url = f"redis://{redis_host}:{redis_port}/0"
        self.redis_client = RedisMetadataLib(redis_url=redis_url)
        self.capacity = capacity
        self.remove_count = min(int(self.capacity * evict_fraction), 1000)  # full的时候，每次清理的数量
        self.afs_embed_dir = afs_embed_dir
        self.afs_utils = AfsUtils(self.afs_embed_dir)

    def full_to_clean(self):
        remove_objs: List[str] = self.redis_client.get_eviction_candidates(
            remove_size=self.remove_count, capacity=self.capacity
        )
        for obj in remove_objs:
            try:
                if self.afs_utils.free_afs(obj):
                    self.redis_client.remove([obj])
            except BaseException as e:
                logger.warning(f"full_to_clean md5 {obj} error {str(e)}")

    def insert(self, md5: str, tensor: torch.Tensor) -> bool:
        self.full_to_clean()
        try:
            # 保证一定会有清理的可能性
            self.redis_client.update(md5)
            ans = self.afs_utils.save_tensor_afs(md5, tensor)
            self.redis_client.update(md5)
            return ans
        except:
            return False

    def load(self, md5: str) -> Optional[torch.Tensor]:
        try:
            ans = self.afs_utils.load_tensor_afs(md5)
            if ans is not None:
                self.redis_client.update(md5)
                return ans
            else:
                return None
        except Exception as e:
            logger.warning(f"load md5 {md5} error {str(e)}")
            return None

    def check_ready(self, md5_list: List[str]) -> List[bool]:
        try:
            tmp1 = self.redis_client.check_and_update(md5_list)
            assert len(tmp1) == len(md5_list)
            start = time.time()
            tmp2 = [exists and self.afs_utils.exist_afs(md5) for md5, exists in zip(md5_list, tmp1)]
            cost_time = time.time() - start
            if cost_time > 0.05:
                logger.warning(f"slow afs check exist {cost_time} seconds, md5_list size: {len(md5_list)}")
            assert len(tmp1) == len(tmp2)
            ans = [a and b for a, b in zip(tmp1, tmp2)]
            return ans
        except Exception as e:
            logger.warning(f"check_ready error {str(e)}")
            return [False] * len(md5_list)
