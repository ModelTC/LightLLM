import os
import random
import time

import torch

from cache import PyLocalCacheService, PyTask

FILE_SIZE = 256 * (1024**3)  # 128GB
VOCABS = 180000

PAGE_SIZE = 16384
NUM_PAGES = 128
NUM_LAYERS = 60

DTYPE = torch.half
ELEMENT_BYTES = torch.tensor([], dtype=DTYPE).element_size()
if PAGE_SIZE % ELEMENT_BYTES != 0:
    raise ValueError(f"PAGE_SIZE={PAGE_SIZE} 必须是 dtype {DTYPE} 单元素字节数 {ELEMENT_BYTES} 的整数倍")

PAGE_ELEMENTS = PAGE_SIZE // ELEMENT_BYTES

kvcache_src = torch.rand(size=[NUM_PAGES, NUM_LAYERS, PAGE_ELEMENTS], dtype=DTYPE, device="cpu")
kvcache_dst = torch.zeros_like(kvcache_src)

# ensure storage directory exists for LocalStorageEngine shards
os.makedirs("cache", exist_ok=True)

service_write = PyLocalCacheService(
    kvcache_tensor=kvcache_src,
    file="cache/cache_file",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
)
service_read = PyLocalCacheService(
    kvcache_tensor=kvcache_dst,
    file="cache/cache_file",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
)

actual_page_size = service_write._page_size
actual_block_size = service_write._block_size

# share hash mapping and shard locks so different services can collaborate
hash_info = service_write.get_hash_info()
service_read.set_hash_info(hash_info)

for num_of_page in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
    tokens = [random.randint(0, VOCABS) for _ in range(num_of_page)]
    indexer = [random.randint(0, NUM_PAGES - 1) for _ in range(num_of_page)]
    indexer = torch.tensor(indexer, device="cpu", dtype=torch.int32)
    size_gb = num_of_page * actual_page_size * NUM_LAYERS / 1e9
    start = time.time()
    t: PyTask = service_write.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not t.ready():
        pass
    end = time.time()

    bandwidth = size_gb / (end - start)

    start = time.time()
    t = service_read.create(tokens=tokens, kv_page_indexer=indexer, mode="r")
    while not t.ready():
        pass
    end = time.time()

    bandwidth = size_gb / (end - start)
    print(
        f"Read Size: {size_gb:.2f} GB, Time Cost: {(end - start) * 1000:.2f} ms, "
        f"Bandwidth: {bandwidth:.2f} GB/Sec"
    )

assert torch.allclose(kvcache_dst, kvcache_src)
