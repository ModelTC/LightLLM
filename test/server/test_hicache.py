# test_hicache.py
import torch
import time
import random
from threading import Thread, Event
from queue import Queue
from lightllm.server.router.dynamic_prompt.cache_controller import HiCacheController, CacheNode, BLOCK_SIZE

class MockMemoryManager:
    """模拟内存管理器，仅返回连续的索引值"""
    def __init__(self):
        self.current_idx = 0
        self.kvcache_store = {}

    def alloc(self, size):
        indices = list(range(self.current_idx, self.current_idx + size))
        self.current_idx += size
        self.store(indices, torch.tensor([[0] * 512 for _ in range(size)]))
        return indices
    
    def to_kvcache(self, indices):
        return torch.tensor([self.kvcache_store[idx].tolist() for idx in indices])
    
    def store(self, indices, value):
        for idx, val in zip(indices, value):
            self.kvcache_store[idx] = val
    
    def free(self, indices):
        for idx in indices:
            del self.kvcache_store[idx]

class MockTask:
    def __init__(self, hashs, mode, value=None):
        self.hashs = hashs
        self.mode = mode
        self._ready = Event()
        self.data = value
    
    def ready(self):
        return self._ready.is_set()
    
    def set_ready(self):
        self._ready.set()

class MockService:
    def __init__(self):
        self.tasks = Queue()
        self.running = True
        self.worker = Thread(target=self.process_tasks)
        self.worker.daemon = True
        self.worker.start()
    
    def process_tasks(self):
        while self.running:
            if not self.tasks.empty():
                task = self.tasks.get()
                # 模拟随机延迟后完成任务
                delay = random.uniform(0.01, 0.1)
                time.sleep(delay)
                task.set_ready()
                print(f"Task for {task.hashs} completed after {delay:.2f}s")
            else:
                time.sleep(0.01)
    
    def create(self, hashs, mode, value=None):
        task = MockTask(hashs, mode, value)
        self.tasks.put(task)
        return task
    
    def commit(self, task):
        pass  # 在Mock中不需要实现
    
    def shutdown(self):
        self.running = False
        self.worker.join()

def setup():
    mem_manager = MockMemoryManager()
    service = MockService()
    hicache = HiCacheController(mem_manager)
    hicache.service = service  # 注入模拟服务
    
    # 预先计算单token大小
    dummy_indices = mem_manager.alloc(1)
    kvcache = mem_manager.to_kvcache(dummy_indices[:1])
    token_size = kvcache.nelement() * kvcache.element_size()
    print(f"[TEST] Single token KV cache size: {token_size} bytes, Block size: {BLOCK_SIZE}")
    
    return mem_manager, service, hicache, token_size

def test_basic_write_read(mem_manager, hicache, token_size):
    # 计算每个块可容纳的token数量
    tokens_per_block = BLOCK_SIZE // token_size
    print(f"[TEST] Each block can hold {tokens_per_block} tokens")
    
    # 生成测试数据：刚好占满一个块
    token_ids = list(range(tokens_per_block))
    indices = mem_manager.alloc(len(token_ids))
    
    # 写入缓存
    hicache.write(torch.tensor(token_ids), torch.tensor(indices))
    
    # 等待任务完成
    time.sleep(0.5)  # 确保后台线程处理完成
    
    # 读取验证
    result = hicache.read(torch.tensor(token_ids))
    assert result.tolist() == indices, f"Retrieved indices: {result.tolist()}, Expected indices: {indices}"
    print(f"[TEST] Basic test passed. Retrieved indices: {result.tolist()}")

def test_node_splitting(mem_manager, hicache, token_size):
    tokens_per_block = BLOCK_SIZE // token_size
    # 生成超过一个块的数据
    token_ids = list(range(tokens_per_block + 1))
    indices = mem_manager.alloc(len(token_ids))
    
    hicache.write(torch.tensor(token_ids), torch.tensor(indices))
    time.sleep(0.5)
    
    # 验证根节点应该有子节点
    root = hicache.root
    assert len(root.children) > 0
    print(f"\nRoot node has {len(root.children)} children")
    
    # 读取完整序列
    result = hicache.read(torch.tensor(token_ids))
    assert result.tolist() == indices
    print(f"[TEST] Node splitting test passed. Retrieved indices: {result.tolist()}")

def test_partial_read(mem_manager, hicache):
    token_ids = [1,2,3,4,5]
    indices = mem_manager.alloc(len(token_ids))
    hicache.write(torch.tensor(token_ids), torch.tensor(indices))
    time.sleep(0.2)
    
    # 查询存在的部分前缀
    result = hicache.read(torch.tensor([1,2,3]))
    assert result.tolist() == indices[:3]
    print(f"[TEST] Partial read result: {result.tolist()}")
    
    # 查询不存在的前缀
    result = hicache.read(torch.tensor([1,2,9]))
    assert len(result) == 0
    print(f"[TEST] Non-existent prefix returned: {result.tolist()}")

def main():
    mem_manager, service, hicache, token_size = setup()
    try:
        test_basic_write_read(mem_manager, hicache, token_size)
        test_node_splitting(mem_manager, hicache, token_size)
        test_partial_read(mem_manager, hicache)
    finally:
        service.shutdown()

if __name__ == "__main__":
    main()