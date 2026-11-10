from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityResult:
    """后端对运行环境或 LayerSpec 的支持结论，供初始化时的 selector 使用。

    不支持是正常的候选筛选结果；reason 用于汇总各候选的拒绝原因。
    auto 可以继续尝试候选，显式选择不支持的后端则报错，不静默回退。
    """

    supported: bool
    reason: str = ""

    @classmethod
    def ok(cls):
        return cls(True)

    @classmethod
    def reject(cls, reason: str):
        if not reason:
            raise ValueError("a rejected capability needs a reason")
        return cls(False, reason)
