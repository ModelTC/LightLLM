"""
LightLLM 环境变量管理核心模块

提供类型安全、集中管理的环境变量框架。

设计目标：
1. 集中定义：所有环境变量在一处定义
2. 类型安全：自动类型转换和验证
3. 文档化：每个变量都有描述和元数据
4. 向后兼容：支持别名和废弃警告
5. 分组管理：按功能分类
"""

from __future__ import annotations
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

T = TypeVar("T")


class EnvVarType(Enum):
    """环境变量类型枚举"""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST_INT = "list_int"
    LIST_STR = "list_str"


@dataclass
class EnvVar(Generic[T]):
    """
    环境变量描述符

    用于定义单个环境变量的所有元数据，包括：
    - 名称、类型、默认值
    - 描述、单位、有效范围
    - 别名（用于向后兼容）
    - 验证器

    示例用法：
        >>> LOG_LEVEL = EnvVar(
        ...     name="LIGHTLLM_LOG_LEVEL",
        ...     var_type=EnvVarType.STRING,
        ...     default="info",
        ...     description="日志级别",
        ...     choices=["debug", "info", "warning", "error"],
        ... )
        >>> LOG_LEVEL.get()  # 返回当前值
        'info'
    """

    name: str
    var_type: EnvVarType
    default: Optional[T] = None
    description: str = ""
    choices: Optional[List[T]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    unit: str = ""
    required: bool = False
    deprecated: bool = False
    deprecated_message: str = ""
    aliases: List[str] = field(default_factory=list)
    validator: Optional[Callable[[T], bool]] = None
    _cached_value: Optional[T] = field(default=None, repr=False, compare=False)
    _is_cached: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        """验证配置的有效性"""
        if self.required and self.default is not None:
            warnings.warn(f"EnvVar {self.name}: required=True 但设置了默认值，这可能是配置错误")

        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(f"EnvVar {self.name}: min_value > max_value")

    def _convert_value(self, raw_value: str) -> T:
        """将字符串值转换为目标类型"""
        if self.var_type == EnvVarType.STRING:
            return raw_value

        elif self.var_type == EnvVarType.INT:
            return int(raw_value)

        elif self.var_type == EnvVarType.FLOAT:
            return float(raw_value)

        elif self.var_type == EnvVarType.BOOL:
            return raw_value.upper() in ("1", "TRUE", "ON", "YES")

        elif self.var_type == EnvVarType.LIST_INT:
            if not raw_value.strip():
                return []
            return [int(x.strip()) for x in raw_value.split(",")]

        elif self.var_type == EnvVarType.LIST_STR:
            if not raw_value.strip():
                return []
            return [x.strip() for x in raw_value.split(",")]

        raise ValueError(f"未知的变量类型: {self.var_type}")

    def _validate_value(self, value: T) -> None:
        """验证值的有效性"""
        if value is None:
            if self.required:
                raise ValueError(f"环境变量 {self.name} 是必需的但未设置")
            return

        if self.choices is not None and value not in self.choices:
            raise ValueError(f"环境变量 {self.name}={value} 不在有效选项中: {self.choices}")

        if self.var_type in (EnvVarType.INT, EnvVarType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"环境变量 {self.name}={value} 小于最小值 {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"环境变量 {self.name}={value} 大于最大值 {self.max_value}")

        if self.validator is not None and not self.validator(value):
            raise ValueError(f"环境变量 {self.name}={value} 未通过自定义验证")

    def _get_raw_value(self) -> Optional[str]:
        """获取原始字符串值，支持别名"""
        # 首先检查主名称
        value = os.environ.get(self.name)
        if value is not None:
            return value

        # 检查别名（向后兼容）
        for alias in self.aliases:
            value = os.environ.get(alias)
            if value is not None:
                if self.deprecated or alias != self.name:
                    warnings.warn(
                        f"环境变量 {alias} 已废弃，请使用 {self.name}",
                        DeprecationWarning,
                        stacklevel=4,
                    )
                return value

        return None

    def get(self, use_cache: bool = True) -> T:
        """
        获取环境变量的值

        Args:
            use_cache: 是否使用缓存值（默认True）

        Returns:
            转换后的值

        Raises:
            ValueError: 如果值无效或必需变量未设置
        """
        if use_cache and self._is_cached:
            return self._cached_value

        raw_value = self._get_raw_value()

        if raw_value is None:
            value = self.default
        else:
            try:
                value = self._convert_value(raw_value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"环境变量 {self.name}={raw_value} 类型转换失败: {e}")

        self._validate_value(value)

        if self.deprecated and raw_value is not None:
            warnings.warn(
                f"环境变量 {self.name} 已废弃. {self.deprecated_message}",
                DeprecationWarning,
                stacklevel=2,
            )

        # 缓存值
        self._cached_value = value
        self._is_cached = True

        return value

    def set(self, value: T) -> None:
        """
        设置环境变量的值

        Args:
            value: 要设置的值

        Raises:
            ValueError: 如果值无效
        """
        self._validate_value(value)

        if self.var_type == EnvVarType.BOOL:
            str_value = "1" if value else "0"
        elif self.var_type in (EnvVarType.LIST_INT, EnvVarType.LIST_STR):
            str_value = ",".join(str(x) for x in value)
        else:
            str_value = str(value)

        os.environ[self.name] = str_value

        # 更新缓存
        self._cached_value = value
        self._is_cached = True

    def clear_cache(self) -> None:
        """清除缓存"""
        self._cached_value = None
        self._is_cached = False

    def is_set(self) -> bool:
        """检查环境变量是否已设置"""
        return self._get_raw_value() is not None

    def reset(self) -> None:
        """重置为默认值"""
        if self.name in os.environ:
            del os.environ[self.name]
        self.clear_cache()

    def __str__(self) -> str:
        return f"{self.name}={self.get()}"

    def __repr__(self) -> str:
        return f"EnvVar({self.name}, type={self.var_type.value}, default={self.default})"


class EnvGroup:
    """
    环境变量分组

    用于按功能组织相关的环境变量，便于管理和文档化。

    示例用法：
        >>> logging = EnvGroup("logging", "日志相关配置")
        >>> logging.add(LOG_LEVEL)
        >>> logging.add(LOG_DIR)
        >>> logging.validate_all()  # 验证所有变量
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._vars: Dict[str, EnvVar] = {}

    def add(self, var: EnvVar) -> EnvVar:
        """添加环境变量到组"""
        if var.name in self._vars:
            raise ValueError(f"环境变量 {var.name} 已存在于组 {self.name} 中")
        self._vars[var.name] = var
        return var

    def get(self, name: str) -> EnvVar:
        """获取环境变量描述符"""
        if name not in self._vars:
            raise KeyError(f"环境变量 {name} 不在组 {self.name} 中")
        return self._vars[name]

    def get_value(self, name: str) -> Any:
        """获取环境变量的值"""
        return self.get(name).get()

    def validate_all(self) -> List[str]:
        """
        验证组内所有变量

        Returns:
            错误消息列表，如果全部有效则返回空列表
        """
        errors = []
        for var in self._vars.values():
            try:
                var.get()
            except ValueError as e:
                errors.append(str(e))
        return errors

    def clear_all_caches(self) -> None:
        """清除所有缓存"""
        for var in self._vars.values():
            var.clear_cache()

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（用于调试和序列化）"""
        return {name: var.get() for name, var in self._vars.items()}

    def to_markdown(self) -> str:
        """生成 Markdown 格式的文档"""
        lines = [
            f"### {self.name}",
            "",
            self.description,
            "",
            "| 变量名 | 类型 | 默认值 | 描述 |",
            "|--------|------|--------|------|",
        ]

        for var in sorted(self._vars.values(), key=lambda v: v.name):
            default_str = str(var.default) if var.default is not None else "-"
            desc = var.description
            if var.unit:
                desc += f" ({var.unit})"
            if var.deprecated:
                desc = f"⚠️ 已废弃: {desc}"
            lines.append(f"| `{var.name}` | {var.var_type.value} | {default_str} | {desc} |")

        return "\n".join(lines)

    def __iter__(self):
        return iter(self._vars.values())

    def __len__(self):
        return len(self._vars)

    def __contains__(self, name: str):
        return name in self._vars


class EnvManager:
    """
    环境变量中央管理器

    单例模式，管理所有环境变量组和变量。

    示例用法：
        >>> manager = EnvManager.get_instance()
        >>> manager.register_group(logging_group)
        >>> manager.validate_all()
        >>> print(manager.generate_docs())
    """

    _instance: Optional[EnvManager] = None

    def __init__(self):
        self._groups: Dict[str, EnvGroup] = {}
        self._all_vars: Dict[str, EnvVar] = {}
        self._alias_map: Dict[str, str] = {}  # alias -> canonical name

    @classmethod
    def get_instance(cls) -> EnvManager:
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        cls._instance = None

    def register_group(self, group: EnvGroup) -> EnvGroup:
        """注册一个环境变量组"""
        if group.name in self._groups:
            raise ValueError(f"组 {group.name} 已注册")

        self._groups[group.name] = group

        # 注册组内所有变量
        for var in group:
            if var.name in self._all_vars:
                raise ValueError(f"环境变量 {var.name} 已在其他组中注册")
            self._all_vars[var.name] = var

            # 注册别名映射
            for alias in var.aliases:
                if alias in self._alias_map:
                    raise ValueError(f"别名 {alias} 已被使用")
                self._alias_map[alias] = var.name

        return group

    def get_var(self, name: str) -> EnvVar:
        """通过名称或别名获取环境变量"""
        # 首先检查是否是别名
        canonical_name = self._alias_map.get(name, name)
        if canonical_name not in self._all_vars:
            raise KeyError(f"未知的环境变量: {name}")
        return self._all_vars[canonical_name]

    def get_value(self, name: str) -> Any:
        """获取环境变量的值"""
        return self.get_var(name).get()

    def set_value(self, name: str, value: Any) -> None:
        """设置环境变量的值"""
        self.get_var(name).set(value)

    def get_group(self, name: str) -> EnvGroup:
        """获取环境变量组"""
        if name not in self._groups:
            raise KeyError(f"未知的组: {name}")
        return self._groups[name]

    def validate_all(self) -> Dict[str, List[str]]:
        """
        验证所有环境变量

        Returns:
            字典：{组名: [错误消息列表]}
        """
        errors = {}
        for group_name, group in self._groups.items():
            group_errors = group.validate_all()
            if group_errors:
                errors[group_name] = group_errors
        return errors

    def validate_required(self) -> List[str]:
        """验证所有必需的环境变量是否已设置"""
        errors = []
        for var in self._all_vars.values():
            if var.required and not var.is_set():
                errors.append(f"必需的环境变量 {var.name} 未设置")
        return errors

    def clear_all_caches(self) -> None:
        """清除所有缓存"""
        for group in self._groups.values():
            group.clear_all_caches()

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """导出所有变量为嵌套字典"""
        return {group_name: group.to_dict() for group_name, group in self._groups.items()}

    def generate_docs(self, format: str = "markdown") -> str:
        """
        生成文档

        Args:
            format: 输出格式，目前支持 "markdown"

        Returns:
            格式化的文档字符串
        """
        if format != "markdown":
            raise ValueError(f"不支持的格式: {format}")

        lines = [
            "# LightLLM 环境变量参考",
            "",
            "本文档列出所有支持的环境变量及其配置。",
            "",
        ]

        for group in sorted(self._groups.values(), key=lambda g: g.name):
            lines.append(group.to_markdown())
            lines.append("")

        return "\n".join(lines)

    def list_all_vars(self) -> List[str]:
        """列出所有已注册的环境变量名"""
        return sorted(self._all_vars.keys())

    def list_groups(self) -> List[str]:
        """列出所有组名"""
        return sorted(self._groups.keys())

    def find_vars(self, pattern: str) -> List[EnvVar]:
        """根据名称模式查找变量"""
        import re

        regex = re.compile(pattern, re.IGNORECASE)
        return [var for name, var in self._all_vars.items() if regex.search(name)]

    def print_summary(self) -> None:
        """打印环境变量摘要"""
        print(f"LightLLM 环境变量摘要")
        print(f"=" * 50)
        print(f"总组数: {len(self._groups)}")
        print(f"总变量数: {len(self._all_vars)}")
        print()
        for group_name, group in sorted(self._groups.items()):
            print(f"  [{group_name}] {len(group)} 个变量")
            for var in group:
                value = var.get() if var.is_set() else f"(默认: {var.default})"
                print(f"    - {var.name}: {value}")

    def __contains__(self, name: str):
        return name in self._all_vars or name in self._alias_map

    def __getitem__(self, name: str) -> Any:
        return self.get_value(name)

    def __setitem__(self, name: str, value: Any) -> None:
        self.set_value(name, value)
