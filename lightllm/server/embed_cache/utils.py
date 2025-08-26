import os
import time
import torch
import redis
import numpy as np
from typing import List, Tuple
from io import BytesIO
from pathlib import Path
import multiprocessing.shared_memory as shm
from lightllm.utils.envs_utils import get_env_start_args


def tensor2bytes(t: torch.Tensor):
    # t = t.cpu().numpy().tobytes()
    # return t
    buf = BytesIO()
    t = t.detach().cpu()
    # 这个地方进行新的empty并复制是因为，torch的tensor save的机制存在问题
    # 如果 t 是从一个大 tensor 上切片复制下来的的tensor， 在save的时候，其
    # 会保存大tensor的所有数据，所以会导致存储开销较大，需要申请一个新的tensor
    # 并进行复制，来打断这种联系。
    dest = torch.empty_like(t)
    dest.copy_(t)
    torch.save(dest, buf, _use_new_zipfile_serialization=False, pickle_protocol=4)
    buf.seek(0)
    return buf.read()


def bytes2tensor(b):
    # return torch.from_numpy(np.frombuffer(b, dtype=np.float16)).cuda()
    return torch.load(BytesIO(b), weights_only=False)


def create_shm(name, data):
    try:
        data_size = len(data)
        shared_memory = shm.SharedMemory(name=name, create=True, size=data_size)
        mem_view = shared_memory.buf
        mem_view[:data_size] = data
    except FileExistsError:
        print("Warning create shm {} failed because of FileExistsError!".format(name))


def create_afs(name, data):
    try:
        data_size = len(data)
        path = os.path.join(get_env_start_args().visual_embed_path, name)
        with open(path, "xb") as f:
            mem_view = memoryview(data)
            f.write(mem_view[:data_size])
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        print("Warning create afs {} failed because of FileExistsError!".format(name))


def read_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    data = shared_memory.buf.tobytes()
    return data


def read_afs(name: str, base_dir: str = "/mtc/sangchengmeng/afs") -> bytes:

    path = Path(base_dir) / name
    return path.read_bytes()


def free_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    shared_memory.close()
    shared_memory.unlink()


def get_shm_name_data(uid):
    return str(uid) + "-data"


def get_shm_name_embed(uid):
    return str(uid) + "-embed"


def afs_embed_exists(md5sum: str):
    uid_int = int(md5sum, 16)
    filename = f"{uid_int}-embed"
    fullpath = os.path.join(get_env_start_args().visual_embed_path, filename)
    return True if os.path.isfile(fullpath) else False


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
        capacity: int = 50_000,
        evict_fraction: float = 0.2,
        key_prefix: str = "md5:",
        image_embed_dir: str = None,
        path_ext: str = ".embed",
        **redis_kwargs,
    ) -> None:
        """
        - capacity: max count of md5 entries allowed in Redis
        - evict_fraction: fraction to evict when inserting a NEW md5 and at capacity
        - image_embed_dir: base directory for image embed files (e.g., "/afs/embeds")
        - path_ext: file extension for embed files (default: ".embed")
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
        """Insert a new md5 with default ref_count=0. May trigger LRU eviction."""
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

                    # 删除被逐出md5对应的AFS文件
                    if victims and self.image_embed_dir:
                        self._delete_afs_files(victims)

                    return success, victims
                finally:
                    self._release_lock()
            else:
                # 等待锁释放后重试
                time.sleep(0.1)
                return self.insert(md5)
        except Exception as e:
            self._release_lock()
            raise e

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
        filename = md5 + self.path_ext
        return filename

    def _delete_afs_files(self, victims: List[str]) -> None:
        """Delete AFS files for evicted md5s."""
        if not self.image_embed_dir:
            return

        for md5 in victims:
            try:
                file_path = self._md5_to_afs_path(md5)
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted AFS file: {file_path}")
            except Exception as e:
                print(f"Warning: Failed to delete AFS file for {md5}: {e}")

    # ---------------- Lua scripts ----------------
    _INSERT_LUA = r"""
-- KEYS[1] = zset key, KEYS[2] = ref_prefix
-- ARGV[1] = md5, ARGV[2] = capacity, ARGV[3] = evict_fraction
local zset = KEYS[1]
local ref_prefix = KEYS[2]
local md5 = ARGV[1]
local capacity = tonumber(ARGV[2])

local ref_key = ref_prefix .. md5
if redis.call('GET', ref_key) then
  return {0}  -- Already exists
end

local size = redis.call('ZCARD', zset)
if size < capacity then
  -- Insert with ref_count=0
  redis.call('SET', ref_key, 0)
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

local rc = tonumber(val) - 1
if rc <= 0 then
  redis.call('DEL', ref_key)
  redis.call('ZREM', zset, md5)
  return {0, 1}  -- Deleted
else
  redis.call('SET', ref_key, rc)
  local now = redis.call('TIME')[1] * 1000
  redis.call('ZADD', zset, now, md5)
  return {rc, 0}  -- Updated
end
"""

    _EVICT_AND_INSERT_LUA = r"""
-- KEYS[1] = zset key, KEYS[2] = ref_prefix
-- ARGV[1] = new_md5, ARGV[2] = capacity, ARGV[3] = evict_fraction
local zset = KEYS[1]
local ref_prefix = KEYS[2]
local new_md5 = ARGV[1]
local capacity = tonumber(ARGV[2])
local evict_fraction = tonumber(ARGV[3])

-- 计算需要逐出的数量
local need = math.max(1, math.floor(capacity * evict_fraction + 0.5))
local victims = {}

-- 获取所有键并按LRU排序
local all_keys = redis.call('ZRANGE', zset, 0, -1, 'WITHSCORES')
local i = 1

-- 查找引用计数为0的键作为逐出候选
while #victims < need and i <= #all_keys do
    local md5 = all_keys[i]
    local ref_key = ref_prefix .. md5
    local rc = redis.call('GET', ref_key)
    
    if rc and tonumber(rc) <= 0 then
        table.insert(victims, md5)
    end
    i = i + 2  -- 跳过分数
end

-- 如果找到足够的候选，执行逐出
if #victims >= need then
    -- 删除受害者
    for _, v in ipairs(victims) do
        local ref_key = ref_prefix .. v
        redis.call('DEL', ref_key)
        redis.call('ZREM', zset, v)
    end
    
    -- 插入新的md5
    local ref_key = ref_prefix .. new_md5
    redis.call('SET', ref_key, 0)
    local now = redis.call('TIME')[1] * 1000
    redis.call('ZADD', zset, now, new_md5)
    
    return {1, table.unpack(victims)}  -- success + victims
else
    return {0}  -- 逐出失败，没有足够的候选
end
"""
