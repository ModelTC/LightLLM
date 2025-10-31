import os
import torch
from cache import PyLocalCacheService, PyTask, PyState

FILE_SIZE = 32 * (1024**3) # 32GB
VOCABS = 180000

PAGE_SIZE = 64
NUM_PAGES = 12800
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

os.makedirs("cache", exist_ok=True)
service = PyLocalCacheService(
    kvcache_tensor=kvcache,
    file="cache/cache_file",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
)

tokens  = [_ for _ in range(NUM_PAGES)]
indexer = [_ for _ in range(NUM_PAGES)]
indexer = torch.tensor(indexer, device="cpu", dtype=torch.int32)

t: PyTask = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
service.az5(t)

if t.ready() != True:
    raise Exception("Unexpected task state.")

F, A = 0, 0
for state in t.state():
    if state not in {PyState.Finished, PyState.Aborted}:
        raise Exception("Unexpected task state.")
    elif state == PyState.Finished:
        F += 1
    else: A += 1

print(f"{F} Finished Job, {A} Aborted Job.")