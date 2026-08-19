from abc import ABC, abstractmethod


class BaseDpOverlapPlanner(ABC):
    """DP overlap draft 配置的基础规划接口。"""

    @abstractmethod
    def get_draft_step(self) -> int:
        """返回当前 overlap proposal 使用的 draft step。"""

        raise NotImplementedError
