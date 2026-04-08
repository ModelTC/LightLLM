from typing import Any, Optional

import torch


class KvBuffer:
    """KV cache 的数据封装类。

    这个类的职责是管理 kv buffer 本身的存储与访问语义，关注点是
    "这块缓存里存了什么、怎么按层读写、怎么导入导出"。
    因此这里的方法应当主要围绕 kv buffer 自身的数据操作展开，
    不承载 page io、cpu cache、dp 传输这类业务流程逻辑。
    """

    def __init__(self, buffer: torch.Tensor, head_num: int):
        self._buffer = buffer
        self._head_num = head_num

    def create_adapter(self):
        # 业务逻辑由 adapter 承接，KvBuffer 只负责提供底层存储对象。
        from .kv_buffer_adapter import KvBufferAdapter

        return KvBufferAdapter(self)

    def __getitem__(self, item):
        return self._buffer[item]

    @property
    def shape(self):
        return self._buffer.shape

    def get_storage_tensor(self) -> torch.Tensor:
        return self._buffer

    def get_storage_data_ptr(self) -> int:
        return self._buffer.data_ptr()

    def get_scale_buffer(self) -> Optional[torch.Tensor]:
        return None

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor) -> None:
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        destindex_copy_kv(kv, mem_index, self._buffer[layer_index])

    def get_att_input_params(self, layer_index: int) -> Any:
        layer_buffer = self._buffer[layer_index]
        k = layer_buffer[:, : self._head_num, :]
        v = layer_buffer[:, self._head_num :, :]
        return k, v

    def get_index_kv_buffer(self, index: Any) -> dict:
        return {"kv_buffer": self._buffer[:, index]}

    def load_index_kv_buffer(self, index: Any, payload: dict) -> None:
        self._buffer[:, index].copy_(payload["kv_buffer"])

    def get_device(self) -> int:
        return self._buffer.get_device()

    def find_layer_index(self, k: torch.Tensor, v: torch.Tensor) -> int:
        key = min(k.data_ptr(), v.data_ptr())
        find_dict = {self._buffer[i].data_ptr(): i for i in range(len(self._buffer))}
        assert key in find_dict
        return find_dict[key]
