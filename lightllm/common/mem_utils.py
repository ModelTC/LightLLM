from lightllm.common.mem_manager import MemoryManager
from lightllm.common.int8kv_mem_manager import INT8KVMemoryManager
from lightllm.common.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager

def select_mem_manager_class(mode):
    if "ppl_int8kv" in mode:
        memory_manager_class = PPLINT8KVMemoryManager
        print("Model using mode ppl int8kv")
    elif "triton_int8kv" in mode:
        memory_manager_class = INT8KVMemoryManager
        print("Model using mode triton int8kv")
    else:
        memory_manager_class = MemoryManager
        print("Model using mode normal")
    return memory_manager_class