import random
import time
import os

import torch
from cache import PyLocalCacheService, PyTask

FILE_SIZE = 128 * (1024**3) # 128GB
VOCABS = 180000

PAGE_SIZE = 16384
NUM_PAGES = 128
NUM_LAYERS = 60
TOKENS_PER_BLOCK = 64
BLOCK_SIZE = PAGE_SIZE * NUM_LAYERS * TOKENS_PER_BLOCK

element_bytes = torch.tensor([], dtype=torch.half).element_size()
if PAGE_SIZE % element_bytes != 0:
    raise ValueError(
        f"PAGE_SIZE={PAGE_SIZE} 必须是 dtype torch.half 单元素字节数 {element_bytes} 的整数倍"
    )

page_elements = PAGE_SIZE // element_bytes
kvcache = torch.rand(size=[NUM_PAGES, NUM_LAYERS, page_elements], dtype=torch.half, device="cpu")

# ensure storage directory exists for LocalStorageEngine shards
os.makedirs("cache", exist_ok=True)

service = PyLocalCacheService(
    kvcache_tensor=kvcache,
    file="cache/cache_file",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
)

for num_of_page in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
    tokens  = [random.randint(0, VOCABS) for _ in range(num_of_page)]
    indexer = [random.randint(0, NUM_PAGES - 1) for _ in range(num_of_page)]
    indexer = torch.tensor(indexer, device="cpu", dtype=torch.int32)
    start = time.time()
    t: PyTask = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")

    # wait until t.ready()
    while not t.ready(): pass

    end = time.time()
    size = num_of_page * PAGE_SIZE * NUM_LAYERS / (1e9)
    bandwidth = size / (end - start)

    print(f"Size: {size:.4f} GB, Time Cost: {(end - start) * 1000:.2f} ms, Bandwidth: {bandwidth:.2f} GB/Sec ")

