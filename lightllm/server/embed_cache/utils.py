import os
import time
import torch
import redis
import numpy as np
from typing import List, Tuple
from io import BytesIO
from pathlib import Path
import multiprocessing.shared_memory as shm
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
_ENSURED_AFS_DIRS = set()


def _get_afs_path(base_dir: str, name: str) -> Path:
    if not base_dir:
        raise ValueError("image_embed_dir must be set before using disk-backed embed cache")
    return Path(base_dir) / name


def _ensure_afs_dir(base_dir: Path) -> None:
    base_dir_key = str(base_dir)
    if base_dir_key in _ENSURED_AFS_DIRS:
        return
    if base_dir.exists():
        if not base_dir.is_dir():
            raise ValueError(f"image_embed_dir is not a directory: {base_dir}")
        _ENSURED_AFS_DIRS.add(base_dir_key)
        return

    base_dir.mkdir(parents=True, mode=0o777, exist_ok=True)
    os.chmod(base_dir, 0o777)
    _ENSURED_AFS_DIRS.add(base_dir_key)


def tensor2bytes(t: torch.Tensor):
    buf = BytesIO()
    t = t.detach().cpu()
    dest = torch.empty_like(t)
    dest.copy_(t)
    torch.save(dest, buf, _use_new_zipfile_serialization=False, pickle_protocol=4)
    buf.seek(0)
    return buf.read()


def bytes2tensor(b):
    return torch.load(BytesIO(b), weights_only=False)


def save_tensor_afs(name: str, tensor: torch.Tensor, base_dir: str) -> None:
    target_path = _get_afs_path(base_dir, name)
    _ensure_afs_dir(target_path.parent)
    tmp_path = target_path.parent / f".{target_path.name}.tmp-{os.getpid()}-{time.time_ns()}"

    try:
        with open(tmp_path, "wb") as f:
            torch.save(tensor.detach().cpu(), f, _use_new_zipfile_serialization=False, pickle_protocol=4)
        os.chmod(tmp_path, 0o777)
        os.replace(tmp_path, target_path)
        os.chmod(target_path, 0o777)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        logger.exception(f"failed to save embed tensor file: {target_path}")
        raise


def load_tensor_afs(name: str, base_dir: str) -> torch.Tensor:
    path = _get_afs_path(base_dir, name)
    with open(path, "rb") as f:
        return torch.load(f, weights_only=False)


def create_shm(name, data):
    try:
        data_size = len(data)
        shared_memory = shm.SharedMemory(name=name, create=True, size=data_size)
        mem_view = shared_memory.buf
        mem_view[:data_size] = data
    except FileExistsError:
        print("Warning create shm {} failed because of FileExistsError!".format(name))


def create_afs(name, data, path):
    target_path = _get_afs_path(path, name)
    _ensure_afs_dir(target_path.parent)
    data_size = len(data)
    tmp_path = target_path.parent / f".{target_path.name}.tmp-{os.getpid()}-{time.time_ns()}"

    try:
        with open(tmp_path, "wb") as f:
            mem_view = memoryview(data)
            f.write(mem_view[:data_size])
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o777)
        os.replace(tmp_path, target_path)
        os.chmod(target_path, 0o777)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        logger.exception(f"failed to create embed file: {target_path}")
        raise


def read_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    data = shared_memory.buf.tobytes()
    return data


def read_afs(name: str, base_dir) -> bytes:
    path = _get_afs_path(base_dir, name)
    return path.read_bytes()


def free_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    shared_memory.close()
    shared_memory.unlink()


def free_afs(name: str, base_dir) -> None:
    path = _get_afs_path(base_dir, name)
    path.unlink(missing_ok=True)


def get_shm_name_data(uid):
    return str(uid) + "-data"


def get_shm_name_embed(uid):
    return str(uid) + "-embed"


"""
Importable Redis-backed MD5 refcount with LRU eviction.

Public API:
    from md5_refcount import EmbedRefCountRedis

    cache = EmbedRefCountRedis(
        redis_url="redis://localhost:6379/0",
        capacity=10000,
        evict_fraction=0.2
    )

    # Insert a new md5 with default ref_count=0
    success, evicted_list = cache.insert(md5)
    
    # Query if exists and increment ref_count if found
    exists = cache.query_and_incre(md5)
    
    # Decrement ref_count
    rc, deleted = cache.decr(md5)
    
    s = cache.stats()
"""


class EmbedRefCountRedis:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        capacity: int = 50000,
        evict_fraction: float = 0.1,
        key_prefix: str = "md5:",
        image_embed_dir: str = None,
        path_ext: str = "-embed",
        **redis_kwargs,
    ) -> None:
        """
        - capacity: max count of md5 entries allowed in Redis
        - evict_fraction: fraction to evict when inserting a NEW md5 and at capacity
        - image_embed_dir: base directory for image embed files (e.g., "/afs/embeds")
        - path_ext: file extension for embed files (default: "-embed")
        """
        if not (0.0 <= evict_fraction <= 1.0):
            raise ValueError("evict_fraction must be 0..1")
        if capacity < 1:
            raise ValueError("capacity must be >=1")

        self.capacity = int(capacity)
        self.evict_fraction = float(evict_fraction)
        self.zset_key = f"{key_prefix}lru"
        self.ref_prefix = f"{key_prefix}rc:"
        self.lock_key = f"{key_prefix}evict:lock"
        self.image_embed_dir = image_embed_dir
        self.path_ext = path_ext

        self.r = redis.Redis.from_url(redis_url, decode_responses=True, **redis_kwargs)

        # Register Lua scripts
        self._insert_script = self.r.register_script(self._INSERT_LUA)
        self._query_incre_script = self.r.register_script(self._QUERY_INCRE_LUA)
        self._decr_script = self.r.register_script(self._DECR_LUA)
        self._evict_and_insert_script = self.r.register_script(self._EVICT_AND_INSERT_LUA)

    def insert(self, md5: str) -> Tuple[bool, List[str]]:
        """Insert a new md5 with default ref_count=1. May trigger LRU eviction."""
        # 等待任何正在进行的逐出操作
        self._wait_if_eviction()

        res = self._insert_script(
            keys=[self.zset_key, self.ref_prefix],
            args=[md5, self.capacity, self.evict_fraction],
        )

        if res[0] == 0:  # No eviction needed
            return True, []

        # Need eviction - use atomic eviction script
        try:
            if self._try_acquire_lock():
                try:
                    # 原子执行逐出和插入
                    evict_res = self._evict_and_insert_script(
                        keys=[self.zset_key, self.ref_prefix],
                        args=[md5, self.capacity, self.evict_fraction],
                    )
                    success = bool(evict_res[0])
                    victims = evict_res[1:] if len(evict_res) > 1 else []

                    if success:
                        # 删除被逐出md5对应的AFS文件
                        if victims and self.image_embed_dir:
                            self._delete_afs_files(victims)
                        return True, victims
                    else:
                        # 逐出失败，短暂退避后重试
                        time.sleep(0.01)
                        return self.insert(md5)
                finally:
                    self._release_lock()
            else:
                # 等待锁释放后重试
                time.sleep(0.01)
                return self.insert(md5)
        except Exception as e:
            self._release_lock()
            raise e

    def query(self, md5: str) -> bool:
        """Quert if md5 exists."""
        self._wait_if_eviction()
        return bool(self.r.exists(self.ref_prefix + md5))

    def query_and_incre(self, md5: str) -> bool:
        """Query if md5 exists and increment ref_count if found."""
        self._wait_if_eviction()
        res = self._query_incre_script(
            keys=[self.zset_key, self.ref_prefix],
            args=[md5],
        )
        return bool(res[0])

    def decr(self, md5: str) -> Tuple[int, bool]:
        """Decrement ref_count for md5. Returns (ref_count, deleted)."""
        self._wait_if_eviction()

        res = self._decr_script(
            keys=[self.zset_key, self.ref_prefix],
            args=[md5],
        )
        if res[0] == -1:
            raise KeyError("md5 not found")
        return int(res[0]), bool(res[1])

    def stats(self) -> dict:
        self._wait_if_eviction()

        size = self.r.zcard(self.zset_key)
        return {
            "items": size,
            "capacity": self.capacity,
            "evict_fraction": self.evict_fraction,
        }

    def get_ref(self, md5: str) -> int | None:
        self._wait_if_eviction()
        val = self.r.get(self.ref_prefix + md5)
        return int(val) if val is not None else None

    def _wait_if_eviction(self) -> None:
        max_wait = 30
        start_time = time.time()

        while self.r.exists(self.lock_key):
            if time.time() - start_time > max_wait:
                raise TimeoutError("Eviction operation timeout, waited too long")
            time.sleep(0.01)  # 短暂等待

    def _try_acquire_lock(self) -> bool:
        return bool(self.r.set(self.lock_key, "1", nx=True, ex=30))

    def _release_lock(self) -> None:
        try:
            self.r.delete(self.lock_key)
        except Exception:
            pass

    def _md5_to_afs_path(self, md5: str) -> str:
        """Convert md5 to AFS file path."""
        if not self.image_embed_dir:
            return None
        return str(_get_afs_path(self.image_embed_dir, f"{md5}{self.path_ext}"))

    def _delete_afs_files(self, victims: List[str]) -> None:
        """Delete AFS files for evicted md5s."""
        if not self.image_embed_dir:
            return

        for md5 in victims:
            try:
                file_path = self._md5_to_afs_path(md5)
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Deleted AFS file: {file_path}")
            except Exception as e:
                logger.debug(f"Warning: Failed to delete AFS file for {md5}: {e}")

    # ---------------- Lua scripts ----------------
    _INSERT_LUA = r"""
-- KEYS[1] = zset key, KEYS[2] = ref_prefix
-- ARGV[1] = md5, ARGV[2] = capacity, ARGV[3] = evict_fraction
local zset = KEYS[1]
local ref_prefix = KEYS[2]
local md5 = ARGV[1]
local capacity = tonumber(ARGV[2])

local unpack = unpack or table.unpack
local ref_key = ref_prefix .. md5
if redis.call('GET', ref_key) then
  return {0}  -- Already exists
end

local size = redis.call('ZCARD', zset)
if size < capacity then
  -- Insert with ref_count=1
  redis.call('SET', ref_key, 1)
  local now = redis.call('TIME')[1] * 1000
  redis.call('ZADD', zset, now, md5)
  return {0}  -- Success, no eviction
end

return {1}  -- Need eviction
"""

    _QUERY_INCRE_LUA = r"""
-- KEYS[1] = zset key, KEYS[2] = ref_prefix
-- ARGV[1] = md5
local zset = KEYS[1]
local ref_prefix = KEYS[2]
local md5 = ARGV[1]

local ref_key = ref_prefix .. md5
local val = redis.call('GET', ref_key)

if not val then
  return {0}  -- Not found
end

-- Found, increment ref_count and update LRU
local rc = tonumber(val) + 1
redis.call('SET', ref_key, rc)
local now = redis.call('TIME')[1] * 1000
redis.call('ZADD', zset, now, md5)
return {1}  -- Found and incremented
"""

    _DECR_LUA = r"""
-- KEYS[1] = zset key, KEYS[2] = ref_prefix
-- ARGV[1] = md5
local zset = KEYS[1]
local ref_prefix = KEYS[2]
local md5 = ARGV[1]

local ref_key = ref_prefix .. md5
local val = redis.call('GET', ref_key)

if not val then
  return {-1, 0}  -- Not found
end

--ref 递减到 0 时保留键，只更新计数与 LRU
local rc = tonumber(val) - 1
if rc < 0 then rc = 0 end
redis.call('SET', ref_key, rc)

if rc > 0 then
  -- 只有仍被引用时才更新 LRU
  local now = redis.call('TIME')[1] * 1000
  redis.call('ZADD', zset, now, md5)
end

return {rc, 0}
"""

    _EVICT_AND_INSERT_LUA = r"""
-- KEYS[1] = zset key, KEYS[2] = ref_prefix
-- ARGV[1] = new_md5, ARGV[2] = capacity, ARGV[3] = evict_fraction
local zset = KEYS[1]
local ref_prefix = KEYS[2]
local new_md5 = ARGV[1]
local capacity = tonumber(ARGV[2])
local evict_fraction = tonumber(ARGV[3])

local unpack = unpack or table.unpack

-- helper: now millis
local function now_ms()
  local t = redis.call('TIME')
  return t[1] * 1000 + math.floor(t[2] / 1000)
end

local new_ref_key = ref_prefix .. new_md5

-- If already exists, treat as a hit: bump ref_count and refresh LRU
local cur = redis.call('GET', new_ref_key)
if cur then
  local rc = tonumber(cur) + 1
  redis.call('SET', new_ref_key, rc)
  redis.call('ZADD', zset, now_ms(), new_md5)
  return {1}  -- success, no victims
end

-- If not at capacity, just insert
local size = redis.call('ZCARD', zset)
if size < capacity then
  redis.call('SET', new_ref_key, 1)
  redis.call('ZADD', zset, now_ms(), new_md5)
  return {1}  -- success, no victims
end

-- At capacity: try to evict up to max_try items with rc==0, but success if at least 1 is freed
local max_try = math.max(1, math.floor(size * evict_fraction + 0.5))
local victims = {}
local freed = 0

-- Scan from LRU (smallest score) to MRU
local all_keys = redis.call('ZRANGE', zset, 0, -1, 'WITHSCORES')
local i = 1
while freed < 1 and i <= #all_keys and #victims < max_try do
  local md5 = all_keys[i]
  local ref_key = ref_prefix .. md5
  local v = redis.call('GET', ref_key)
  if v and tonumber(v) <= 0 then
    table.insert(victims, md5)
    freed = freed + 1
  end
  i = i + 2  -- skip score
end

if freed >= 1 then
  -- delete victims
  for _, v in ipairs(victims) do
    redis.call('DEL', ref_prefix .. v)
    redis.call('ZREM', zset, v)
  end
  -- insert new
  redis.call('SET', new_ref_key, 1)
  redis.call('ZADD', zset, now_ms(), new_md5)
  return {1, unpack(victims)}
else
  -- no zero-ref items found
  return {0}
end
"""
