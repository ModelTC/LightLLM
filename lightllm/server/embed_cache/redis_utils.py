import redis
from typing import List, Tuple, Union, Optional

class RedisMetadataLib:
    """
    # 代码任务
    创建一个基于redis 管理的元数据操作库代码。
    要求：
    2. 提供一个包装的 redis 操作client 库，提供以下功能：
    (1) 提供一个时间排序队列，向队列中插入md5，并更新时间错(单位为s即可).
    (2) 输入为(md5_list,), 向队列中插入所有的md5, 并更新其对应时间错。
    (3) 输入为(md5_list,), 将队列中的md5进行删除。
    (4) 输入为(md5_list,), 返回 md5_list 中所有md5 每个是否在链表中存在，返回一个bool list来标识，同时对所有存在的md5，更新时间错到最新。
    (5) 输入为(remove_size, capcity), 当时间排序队列中的元素数量大于等于capcity， 返回时间排序队列中排在前面的 remove_size 个元素,其内容为 md5。
    (6) 所有操作都使用lua 脚本，以实现原子化操作，同时返回的错误要能区分具体错误的原因，注意lua脚本的可读性，和相关函数的输入输出测试。时间错为server端s级别的参数。
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "meta"):
        # decode_responses=True 确保返回的是字符串而非字节
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        self.lru_key = f"{prefix}:queue:lru"
        self._register_scripts()

    def _register_scripts(self):
        """注册 Lua 脚本实现原子化操作"""

        # (1) & (2) 更新/插入：支持传入单个或多个 MD5
        # 逻辑：获取服务器时间，循环执行 ZADD
        self._lua_update = self.r.register_script("""
            local lru_key = KEYS[1]
            local now = redis.call('TIME')[1]
            local count = 0
            for i, md5 in ipairs(ARGV) do
                redis.call('ZADD', lru_key, now, md5)
                count = count + 1
            end
            return count
        """)

        # (3) 删除：从队列中移除指定的 MD5
        self._lua_remove = self.r.register_script("""
            local lru_key = KEYS[1]
            local count = 0
            for i, md5 in ipairs(ARGV) do
                count = count + redis.call('ZREM', lru_key, md5)
            end
            return count
        """)

        # (4) 检查并更新：判断是否存在，存在则刷新时间，返回 bool 状态列表
        self._lua_check_update = self.r.register_script("""
            local lru_key = KEYS[1]
            local now = redis.call('TIME')[1]
            local results = {}
            for i, md5 in ipairs(ARGV) do
                if redis.call('ZSCORE', lru_key, md5) then
                    redis.call('ZADD', lru_key, now, md5)
                    table.insert(results, 1)
                else
                    table.insert(results, 0)
                end
            end
            return results
        """)

        # (5) 容量清理：检查容量并获取候选列表
        self._lua_evict = self.r.register_script("""
            local lru_key = KEYS[1]
            local remove_size = tonumber(ARGV[1])
            local capacity = tonumber(ARGV[2])
            
            local current_size = redis.call('ZCARD', lru_key)
            if current_size >= capacity then
                -- 按照分数（时间戳）从小到大排列，获取最旧的 N 个
                return redis.call('ZRANGE', lru_key, 0, remove_size - 1)
            else
                return {}
            end
        """)

    def _to_list(self, data: Union[str, List[str]]) -> List[str]:
        """内部工具：将输入统一转为列表形式"""
        if isinstance(data, str):
            return [data]
        return data

    def update(self, md5_list: Union[str, List[str]]) -> int:
        """
        功能 (1) & (2)：插入或更新 md5 的时间戳。
        支持传入单个字符串或字符串列表。
        """
        items = self._to_list(md5_list)
        if not items: return 0
        return self._lua_update(keys=[self.lru_key], args=items)

    def remove(self, md5_list: Union[str, List[str]]) -> int:
        """
        功能 (3)：将队列中的 md5 进行删除。
        支持传入单个字符串或字符串列表。
        """
        items = self._to_list(md5_list)
        if not items: return 0
        return self._lua_remove(keys=[self.lru_key], args=items)

    def check_and_update(self, md5_list: List[str]) -> List[bool]:
        """
        功能 (4)：返回 md5_list 中每个 md5 是否在队列中存在。
        对存在的 md5 会同时更新时间戳到最新。
        """
        if not md5_list: return []
        raw_res = self._lua_check_update(keys=[self.lru_key], args=md5_list)
        return [res == 1 for res in raw_res]

    def get_eviction_candidates(self, remove_size: int, capacity: int) -> List[str]:
        """
        功能 (5)：当队列数量 >= capacity 时，返回排在前面的 remove_size 个 md5。
        """
        return self._lua_evict(keys=[self.lru_key], args=[remove_size, capacity])

# ---------------- 功能测试 ----------------

def test_meta_lib():
    lib = RedisMetadataLib(prefix="test_service")
    # 清理历史数据
    lib.r.delete(lib.lru_key)

    print("1. 测试更新 (update)")
    lib.update("file_0")  # 单个
    lib.update(["file_1", "file_2", "file_3"])  # 批量
    print(f"当前队列大小: {lib.r.zcard(lib.lru_key)}")

    print("\n2. 测试检查并更新 (check_and_update)")
    # file_1 存在，file_none 不存在，file_3 存在
    check_list = ["file_1", "file_none", "file_3"]
    exists_results = lib.check_and_update(check_list)
    for m, exists in zip(check_list, exists_results):
        print(f"MD5: {m}, 存在状态: {exists}")

    print("\n3. 测试容量逐出 (get_eviction_candidates)")
    # 当前有 4 个元素，设容量为 3，要求返回最旧的 2 个
    candidates = lib.get_eviction_candidates(remove_size=2, capacity=3)
    print(f"容量达到3时，建议删除的最旧2个元素: {candidates}")

    print("\n4. 测试删除 (remove)")
    removed_count = lib.remove(["file_0", "file_1"])
    print(f"成功移除数量: {removed_count}")
    
    final_check = lib.check_and_update(["file_1", "file_2"])
    print(f"最终检查 [file_1, file_2]: {final_check}")

if __name__ == "__main__":
    test_meta_lib()