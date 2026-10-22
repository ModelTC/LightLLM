"""在轻量生命周期测试中复用生产 ReqTokenIndex；矩阵仅作为测试数据源。"""
from types import SimpleNamespace

import torch


def make_cpu_index_manager(token_table, page_size=1, **attributes):
    from lightllm.common.req_manager.base import ReqManager
    from lightllm.server.router.model_infer.infer_req_ext1 import PinMemCacheReqTokenIndex

    class CpuOnlyTokenIndex(PinMemCacheReqTokenIndex):
        def _reserve(self, end):
            # CPU 生命周期测试只模拟缓冲区分配，追加、快照和 reset 仍复用生产实现。
            # 按测试请求表容量申请普通内存，不依赖 CUDA 或 pinned allocator。
            if self._cpu_buffer is not None and self._cpu_buffer.numel() >= end:
                return
            buffer = torch.empty_like(self.gpu_indexes, device="cpu", requires_grad=False)
            if self.token_index_len:
                buffer[: self.token_index_len].copy_(self._cpu_buffer[: self.token_index_len])
            self._cpu_buffer = buffer

    class TestReqManager(ReqManager):
        def get_req_token_index(self, req_idx):
            if req_idx not in self._req_token_indexes:
                req = SimpleNamespace(
                    req_idx=req_idx,
                    req_manager=self,
                    args=SimpleNamespace(req_token_index_mode="pin_mem_cache"),
                    shm_req=SimpleNamespace(input_len=self.token_indexes_cpu.shape[1]),
                )
                if self.req_to_token_indexs.is_cuda:
                    helper = PinMemCacheReqTokenIndex(
                        req, index_stream=torch.cuda.default_stream(self.req_to_token_indexs.device)
                    )
                else:
                    # CPU 测试不调用 GPU 接口，无需构造索引流。
                    helper = CpuOnlyTokenIndex(req, index_stream=None)
                # 既有底层 CUDA 测试直接布置 source；生产实现按请求独立申请。
                helper._cpu_buffer = self.token_indexes_cpu[req_idx]
                self._req_token_indexes[req_idx] = helper
            return self._req_token_indexes[req_idx]

        def get_cpu_token_indexes(self, req_idx, start, end):
            return self.get_req_token_index(req_idx).get_cpu_token_indexes(start, end)

        def append_token_indexes(self, req_idx, indexes):
            self.get_req_token_index(req_idx).append_token_indexes(indexes)

    manager = TestReqManager.__new__(TestReqManager)
    manager.page_size = page_size
    manager.req_token_index_mode = "pin_mem_cache"
    manager.token_indexes_cpu = token_table
    manager.req_to_token_indexs = torch.full_like(token_table, -1)
    manager._req_token_indexes = {}
    for name, value in attributes.items():
        setattr(manager, name, value)
    return manager


def bind_cpu_indexes(req, manager, gpu_index_submitted_len=0, token_index_len=None):
    req.req_manager = manager
    if not hasattr(req, "args"):
        req.args = SimpleNamespace(req_token_index_mode=manager.req_token_index_mode)
    req.token_index = manager.get_req_token_index(req.req_idx)
    # 矩阵只提供容量和预置数据；追加位置取请求实际持有的长度，不能使用整行容量。
    if token_index_len is None:
        token_index_len = req.hold_kv_len if hasattr(req, "hold_kv_len") else 0
    req.token_index.token_index_len = token_index_len
    req.token_index.gpu_index_submitted_len = gpu_index_submitted_len
    return req
