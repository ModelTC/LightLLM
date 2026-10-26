from dataclasses import dataclass, fields
from typing import Any, Callable

from lightllm.platform.ops import PlatformOps
from lightllm.platform.plugin.common import is_plugin_registering
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

OpKey = tuple[str, type, str]


@dataclass(frozen=True)
class _OpImpl:
    fn: Callable[..., Any]
    plugin: bool


OP_IMPLS: dict[OpKey, _OpImpl] = {}


def register_ops(
    group: type,
    *,
    platforms: tuple[str, ...],
    **ops: Callable[..., Any],
) -> None:
    unknown = sorted(set(ops) - {field.name for field in fields(group)})
    if unknown:
        raise ValueError(f"Unknown op(s) {unknown} on {group.__name__}.")

    plugin = is_plugin_registering()
    group_name = next((field.name for field in fields(PlatformOps) if field.type is group), group.__name__)
    staged: dict[OpKey, _OpImpl] = {}
    for name, fn in ops.items():
        for platform in platforms:
            key = (platform, group, name)
            current = staged.get(key, OP_IMPLS.get(key))
            # 当前 key 不存在，则直接注册
            if current is None:
                staged[key] = _OpImpl(fn, plugin)
                continue
            # 当前是插件且已有内置实现，则替换
            if plugin and not current.plugin:
                logger.info(
                    "Op %s.%s on platform %s: plugin replaced builtin implementation.",
                    group_name,
                    name,
                    platform,
                )
                staged[key] = _OpImpl(fn, plugin=True)
                continue
            # 按照同样 key 注册多次，报错
            raise ValueError(f"Op {group.__name__}.{name} is already registered for platform {platform!r}.")
    OP_IMPLS.update(staged)


def build_platform_ops(platform: str) -> PlatformOps:
    # 按平台把已登记的算子组装成 PlatformOps，无对应实现置为 unsupported op 在运行时报错
    groups = {}
    for group_field in fields(PlatformOps):
        group_type = group_field.type
        impls = {}
        for op_field in fields(group_type):
            full_name = f"{group_field.name}.{op_field.name}"
            impls[op_field.name] = _lookup(platform, group_type, op_field.name, full_name)
        groups[group_field.name] = group_type(**impls)
    return PlatformOps(**groups)


def _lookup(platform: str, group: type, name: str, full_name: str) -> Callable[..., Any]:
    entry = OP_IMPLS.get((platform, group, name))
    if entry is None:
        return _unsupported_op(platform, full_name)
    return entry.fn


def _unsupported_op(platform: str, full_name: str) -> Callable[..., Any]:

    def op(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"Op {full_name!r} is not implemented for platform {platform!r}.")

    op.__name__ = full_name.rsplit(".", 1)[-1]
    return op
