import torch
import threading
import time
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from queue import Queue
from lightllm.common.mem_manager import MemoryManager

BLOCK_SIZE = 16384

def get_torch_tensor_size(tensor: torch.Tensor):
    return tensor.nelement() * tensor.element_size()

class CacheNode:
    def __init__(self, parent=None, split_token_idx=None):
        self.parent = parent  # 父节点
        self.split_token_idx = split_token_idx  # 从父节点分裂的位置
        self.children = {}  # (token_id, split_position) -> (child_node, split_position)
        self.cache_indices = []  # 存储kv cache在mem_manager中的索引
        self.token_ids = []  # 当前节点存储的token ids
        self.hash = None  # 存储在磁盘上的唯一标识
    
    def serialize(self):
        """将节点数据序列化为JSON"""
        data = {
            "children": {f"{k[0]}_{k[1]}": [c.hash, p] for k, (c, p) in self.children.items()},
            "cache_indices": self.cache_indices,
            "token_ids": self.token_ids,
            "split_token_idx": self.split_token_idx
        }
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data_str, parent=None):
        """从JSON反序列化节点数据"""
        data = json.loads(data_str)
        node = cls(parent=parent, split_token_idx=data["split_token_idx"])
        node.cache_indices = data["cache_indices"]
        node.token_ids = data["token_ids"]
        # 子节点需要单独加载
        return node, {(int(k.split('_')[0]), int(k.split('_')[1])): (v[0], v[1]) for k, v in data["children"].items()}


class HiCacheController:
    def __init__(self, mem_manager: MemoryManager):
        self.mem_manager = mem_manager
        self.service = None  # 将由外部代码初始化
        
        self.root = CacheNode()
        self.root.hash = "root"
        
        self.node_cache = {self.root.hash: self.root}  # hash -> node
        self.read_queue = Queue()
        self.write_queue = Queue()
        
        self.token_kvcache_size = None  # 每个token的kvcache大小
        
        # 启动后台线程处理读写任务
        self.running = True
        self.poll_thread = threading.Thread(target=self._poll_tasks)
        self.poll_thread.daemon = True
        self.poll_thread.start()
    
    def reset(self):
        """重置缓存控制器"""
        self.running = False
        self.poll_thread.join(timeout=1)
        
        self.root = CacheNode()
        self.root.hash = "root"
        self.node_cache = {self.root.hash: self.root}
        
        self.read_queue = Queue()
        self.write_queue = Queue()
        
        self.running = True
        self.poll_thread = threading.Thread(target=self._poll_tasks)
        self.poll_thread.daemon = True
        self.poll_thread.start()
    
    def _poll_tasks(self):
        """轮询读写任务，检查是否完成"""
        while self.running:
            # 处理读任务
            pending_reads = []
            while not self.read_queue.empty():
                task = self.read_queue.get()
                if task.ready():
                    # TODO: 将读到的内容存入 memory manager 中
                    node_hash = task.hashs[0]
                    if node_hash in self.node_cache:
                        node = self.node_cache[node_hash]
                        node.cache_indices = self.mem_manager.store(node.cache_indices, task.value)
                        print(f"Node {node_hash} loaded with {len(node.cache_indices)} cache indices")
                else:
                    pending_reads.append(task)
            
            for task in pending_reads:
                self.read_queue.put(task)
            
            # 处理写任务
            pending_writes = []
            while not self.write_queue.empty():
                task = self.write_queue.get()
                if not task.ready():
                    pending_writes.append(task)
            
            for task in pending_writes:
                self.write_queue.put(task)
            
            time.sleep(0.01)  # 避免CPU过度使用
    
    def _ensure_node_loaded(self, node_hash):
        """确保节点已加载到内存中"""
        if node_hash not in self.node_cache and node_hash != "root":
            task = self.service.create(hashs=[node_hash], mode="r")
            self.service.commit(task)
            self.read_queue.put(task)
            # 需要等待节点加载完成
            while not task.ready() or node_hash not in self.node_cache:
                time.sleep(0.01)
    
    def _persist_node(self, node):
        """将节点持久化到磁盘"""
        print(f"Persisting node {node.hash} with {len(node.token_ids)} tokens")
        if not node.hash:
            # 为新节点生成hash
            node.hash = f"node_{id(node)}_{time.time()}"
        
        # TODO: 将对应的kvcache写入磁盘
        task = self.service.create(hashs=[node.hash], value=self.mem_manager.to_kvcache(node.cache_indices), mode="w")
        self.service.commit(task)
        self.write_queue.put(task)
        self.node_cache[node.hash] = node
    
    def write(self, key: torch.Tensor, value: torch.Tensor):
        """
        写入token序列及其对应的KV缓存索引
        key: token_ids序列
        value: 对应的KV缓存索引
        """
        token_ids = key.cpu().tolist()
        indices = value.cpu().tolist()
        
        # 首次计算每个token的kvcache大小
        if self.token_kvcache_size is None:
            kvcache = self.mem_manager.to_kvcache(indices[:1])  # 计算单个token的kvcache
            self.token_kvcache_size = get_torch_tensor_size(kvcache)
            print(f"Single token KV cache size: {self.token_kvcache_size} bytes, Block size: {BLOCK_SIZE}")
        
        current = self.root
        position = 0
        relative_position = 0
        
        while position < len(token_ids):
            token_id = token_ids[position]
            print(f"Writing token {token_id} at position {position}, current node has {len(current.token_ids)} tokens")
            child_key = (token_id, relative_position)
            
            if child_key in current.children:
                print(f"Child key {child_key} found in current.children")
                child_info = current.children[child_key]
                assert isinstance(child_info[0], CacheNode)
                child_hash = child_info[0].hash
                self._ensure_node_loaded(child_hash)
                current = self.node_cache[child_hash]
                position += 1
                relative_position = 0 # next time relative pos is 0
            else:
                # 计算当前节点剩余空间
                remaining_space = BLOCK_SIZE - len(current.cache_indices) * self.token_kvcache_size
                
                if self.token_kvcache_size <= remaining_space:
                    # 当前节点有足够空间
                    current.token_ids.append(token_ids[position])
                    current.cache_indices.append(indices[position])
                    position += 1
                    relative_position += 1
                    self._persist_node(current)
                else:
                    # 当前节点已满，需要创建新节点
                    new_node = CacheNode(parent=current, split_token_idx=len(current.token_ids))
                    print(f"Creating new node at split position {new_node.split_token_idx}, parent hash: {current.hash}")
                    
                    # 将token添加到新节点
                    new_node.token_ids.append(token_ids[position])
                    new_node.cache_indices.append(indices[position])
                    position += 1
                    relative_position = 0 # next time relative pos is 0, not affecting child_key
                    
                    # 建立父子关系
                    current.children[child_key] = (new_node, len(current.cache_indices))
                    
                    # 持久化
                    self._persist_node(new_node)
                    self._persist_node(current)
                    
                    current = new_node
        
        # 确保最后修改的节点被持久化
        self._persist_node(current)
    
    def read(self, key: torch.Tensor) -> torch.Tensor:
        """
        读取token序列对应的KV缓存索引
        key: token_ids序列
        返回: 对应的KV缓存索引
        """
        token_ids = key.cpu().tolist()
        result_indices = []
        
        current = self.root
        position = 0
        relative_position = 0
        
        while position < len(token_ids):
            token_id = token_ids[position]
            print(f"Reading token {token_id} at position {position}, current node has {len(current.token_ids)} tokens")
            
            # 检查当前节点的token
            if relative_position < len(current.token_ids) and current.token_ids[relative_position] == token_id:
                # TODO: 将读到的东西存到 result_indices 中
                position += 1
                relative_position += 1
                continue
            
            # 查找子节点
            child_key = (token_id, relative_position)
            if child_key in current.children:
                child_info = current.children[child_key]
                assert isinstance(child_info[0], CacheNode)
                child_hash = child_info[0].hash
                self._ensure_node_loaded(child_hash)
                current = self.node_cache[child_hash]
                relative_position = 0
            else:
                # 未找到匹配的路径
                return torch.tensor(result_indices, dtype=torch.int64)
        
        return torch.tensor(result_indices, dtype=torch.int64)
