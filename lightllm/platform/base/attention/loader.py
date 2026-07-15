import importlib
import pkgutil
from lightllm.platform.plugins import ATT


_ATT_ROOT = "lightllm.common.basemodel.attention"

_BUILTIN_ATT_DIRS: tuple[str, ...] = (
    "fa3",
    "flashinfer",
    "nsa",
    "paged_fa3",
    "triton",
)


def _load_builtin_att_backends() -> None:
    for backend_dir in _BUILTIN_ATT_DIRS:
        package = importlib.import_module(f"{_ATT_ROOT}.{backend_dir}")
        for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{package.__name__}."):
            importlib.import_module(module_info.name)


def _load_att_backends(modules: tuple[str, ...] = ()) -> None:
    _load_builtin_att_backends()
    for module_name in modules:
        importlib.import_module(module_name)

    if not modules:
        return

    from lightllm.platform.base.attention.registry import att_backend_registry

    registered_modules = {
        spec.backend_cls.__module__ for spec in att_backend_registry.list_specs()
    }
    missing = [
        module_name for module_name in modules if module_name not in registered_modules
    ]
    if missing:
        raise RuntimeError(
            f"{ATT.name}: {ATT.cli_flag('modules')} modules not registered "
            f"by {ATT.register_decorator}: {missing}. "
            f"Check {ATT.cli_flag('modules')} / {ATT.cli_flag('plugins')} "
            f"and that they call {ATT.register_decorator}."
        )


_att_backends_loaded = False


def ensure_att_backends_loaded() -> None:
    global _att_backends_loaded
    if _att_backends_loaded:
        return

    _load_att_backends(ATT.get().modules)
    _att_backends_loaded = True
