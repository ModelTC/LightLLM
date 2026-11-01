from abc import ABC, abstractmethod


class AttentionCompute(ABC):
    """把编码存储解释为算子视图，不分配 token 槽位或决定传输格式。

    prepare 绑定设备上的辅助参数，随后由 State 分配 payload。
    layer_view 借用 StorageSpec 定义的字段；融合算子可直接写入这些视图。
    """

    def __init__(self, config):
        self.config = config

    def prepare(self, device):
        pass

    @abstractmethod
    def layer_view(self, storage, logical_layer_id):
        ...


class KVWrite(ABC):
    """可选的独立 KV 写入能力，只由确实支持此操作的计算实现提供。

    目标物理 token 槽位必须已预留。实现负责编码与动态 scale，调用方负责
    输入就绪、CUDA 流顺序及目标存活；返回不表示异步写入完成。
    """

    @abstractmethod
    def write(self, storage, logical_layer_id, token_indices, kv):
        ...
