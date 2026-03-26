import redis
from typing import List, Tuple, Union, Optional


class RedisMetadataClient:
    """
    # 代码任务
    创建一个基于redis 管理的元数据操作库代码。
    要求：
    2. 提供一个包装的 redis 操作client 库，提供以下功能：
    (1) 提供输入为（md5, token, time_out） 为其创建一个零时具有超时时间的记录，同时提供输入为(md5, token)的解锁接口，防止多线程的异步操作出现问题。
    (2) 提供一个时间排序队列，当出现对md5的任何操作的时候，向队列中插入md5，并更新时间错(单位为s即可)， 当时创建锁和解锁不更新时间错。
    (3) 输入为(md5，token)， 先校验 md5锁对应的内容为token, 然后标记 md5 对应的资源已经准备就绪， 向时间排序队列插入更新md5的时间错。
    (4) 输入为(md5, token)  先校验 md5锁对应的内容为token, 当 md5 对应的资源存在的时候，同时更新排序队列中的时间错,同时返回True， 否则返回False，不更新时间错。
    (5) 输入为(md5, token)， 先校验 md5锁对应的内容为token, 移除标记 md5 对应的资源已经准备就绪，并同时从时间排序队列中移除对应的md5。
    (6) 输入为(remove_size, capcity), 当时间排序队列中的元素数量大于等于capcity， 返回时间排序队列中排在前面的 remove_size 个元素,其内容为 md5。
    (7) 所有操作都使用lua 脚本，以实现原子化操作，同时返回的错误要能区分具体错误的原因，注意lua脚本的可读性，和相关函数的输入输出测试。
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "meta"):
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
        self.lru_key = f"{prefix}:queue:lru"
        self._register_scripts()

    def _register_scripts(self):
        """注册 Lua 脚本"""

        # (1) 解锁脚本 (不更新时间戳)
        self._lua_unlock = self.r.register_script(
            """
            local lock_key = KEYS[1]
            local token = ARGV[1]
            if redis.call("GET", lock_key) == token then
                return redis.call("DEL", lock_key)
            elseif redis.call("EXISTS", lock_key) == 0 then
                return -2
            else
                return -1
            end
        """
        )

        # (3, 4, 5) 元数据操作脚本
        # 内部通过 redis.call('TIME') 获取服务器时间
        self._lua_meta_op = self.r.register_script(
            """
            local lock_key = KEYS[1]
            local ready_key = KEYS[2]
            local lru_key = KEYS[3]
            local op = ARGV[1]
            local token = ARGV[2]
            local md5 = ARGV[3]

            -- 校验锁
            local current_token = redis.call("GET", lock_key)
            if not current_token then return -2 end
            if current_token ~= token then return -1 end

            -- 获取服务器时间 (秒)
            local server_time = redis.call('TIME')[1]

            if op == "mark_ready" then
                redis.call("SET", ready_key, "1")
                redis.call("ZADD", lru_key, server_time, md5)
                return 1
            elseif op == "check_touch" then
                if redis.call("EXISTS", ready_key) == 1 then
                    redis.call("ZADD", lru_key, server_time, md5)
                    return 1
                else
                    return 0
                end
            elseif op == "remove_ready" then
                redis.call("DEL", ready_key)
                redis.call("ZREM", lru_key, md5)
                return 1
            end
        """
        )

        # (6) 逐出检查脚本
        self._lua_evict = self.r.register_script(
            """
            local lru_key = KEYS[1]
            local remove_size = tonumber(ARGV[1])
            local capacity = tonumber(ARGV[2])
            
            local current_size = redis.call("ZCARD", lru_key)
            if current_size >= capacity then
                return redis.call("ZRANGE", lru_key, 0, remove_size - 1)
            else
                return {}
            end
        """
        )

    def _get_keys(self, md5: str):
        return [f"{self.prefix}:lock:{md5}", f"{self.prefix}:ready:{md5}", self.lru_key]

    def _handle_res(self, res: int):
        """映射错误原因"""
        errors = {
            1: (True, "Success"),
            0: (False, "Resource not ready"),
            -1: (False, "Error: Token mismatch (Permission denied)"),
            -2: (False, "Error: Lock missing or expired"),
        }
        return errors.get(res, (False, f"Unknown error code: {res}"))

    # (1) 创建锁
    def acquire_lock(self, md5: str, token: str, time_out: int) -> bool:
        """创建临时超时记录 (不更新排序队列)"""
        lock_key = self._get_keys(md5)[0]
        return bool(self.r.set(lock_key, token, nx=True, ex=time_out))

    # (1) 解锁
    def release_lock(self, md5: str, token: str) -> Tuple[bool, str]:
        """解锁 (不更新排序队列)"""
        res = self._lua_unlock(keys=[self._get_keys(md5)[0]], args=[token])
        return self._handle_res(res)

    # (3) 标记就绪
    def mark_ready(self, md5: str, token: str) -> Tuple[bool, str]:
        """标记就绪并在 Lua 内部更新服务器时间戳"""
        keys = self._get_keys(md5)
        # 不再传入 now，Lua 脚本内部自行获取
        res = self._lua_meta_op(keys=keys, args=["mark_ready", token, md5])
        return self._handle_res(res)

    # (4) 检查就绪并 Touch
    def check_ready_and_touch(self, md5: str, token: str) -> Tuple[bool, str]:
        """校验锁和就绪状态，并在 Lua 内部更新服务器时间戳"""
        keys = self._get_keys(md5)
        res = self._lua_meta_op(keys=keys, args=["check_touch", token, md5])
        return self._handle_res(res)

    # (5) 移除就绪
    def remove_ready(self, md5: str, token: str) -> Tuple[bool, str]:
        """移除就绪状态并从队列删除"""
        keys = self._get_keys(md5)
        res = self._lua_meta_op(keys=keys, args=["remove_ready", token, md5])
        return self._handle_res(res)

    # (6) 获取逐出列表
    def get_eviction_candidates(self, remove_size: int, capacity: int) -> List[str]:
        """当数量达到上限，返回最旧的元素"""
        return self._lua_evict(keys=[self.lru_key], args=[remove_size, capacity])


# ---------------- 测试验证 ----------------


def test_client():

    client = RedisMetadataClient()
    md5 = "test_file_server_time"
    token = "secure_token_123"

    print("Step 1: Acquire Lock")
    client.acquire_lock(md5, token, 60)

    print("Step 2: Mark Ready (Updates time inside Lua)")
    ok, msg = client.mark_ready(md5, token)
    print(f"Result: {ok}, {msg}")

    # 检查 Redis 内部 ZSet 存储的时间戳
    score = client.r.zscore(client.lru_key, md5)
    print(f"Server Timestamp in ZSet: {score}")

    print("\nStep 3: Check and Touch (Updates time inside Lua)")
    ok, msg = client.check_ready_and_touch(md5, token)
    print(f"Result: {ok}, {msg}")

    new_score = client.r.zscore(client.lru_key, md5)
    print(f"Updated Server Timestamp: {new_score}")


if __name__ == "__main__":
    test_client()
