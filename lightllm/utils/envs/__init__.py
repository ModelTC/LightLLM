# 环境变量管理框架
# 提供集中式、类型安全的环境变量管理

from lightllm.utils.envs.core import EnvVar, EnvGroup, EnvManager
from lightllm.utils.envs.registry import env

__all__ = ["EnvVar", "EnvGroup", "EnvManager", "env"]
