# 这个里面声明了一个全局变量，主要用于推理进程缓存发送给其他进程的Kv move 任务的缓存数据
# 为了减少一些调用时候的序列化开销。有些调用就只需要传输一个请求id就可以了，不用传输特别的
# 数据了，提升rpyc 调用的速度, 只用在 prefill_impl.py 和 prefill_infer_rpyc.py 文件中
from typing import Dict, Tuple
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.server.router.dynamic_prompt.radix_cache import TreeNode

g_kv_move_task_cache: Dict[int, Tuple[KVMoveTask, TreeNode]] = {}
