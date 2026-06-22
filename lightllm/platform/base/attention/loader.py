import importlib
import pkgutil

from lightllm.platform.plugins import ATT
from lightllm.platform.plugins.registration_validation import validate_extra_modules_registered


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


def _load_att_backends(extra_modules: tuple[str, ...] = ()) -> None:
    _load_builtin_att_backends()
    for module_name in extra_modules:
        importlib.import_module(module_name)

    if not extra_modules:
        return

    from lightllm.platform.base.attention.registry import att_backend_registry

    registered_modules = {
        spec.backend_cls.__module__ for spec in att_backend_registry.list_specs()
    }
    validate_extra_modules_registered(
        kind_spec=ATT.spec,
        register_decorator="@register_att_backend",
        extra_modules=extra_modules,
        registered_module_names=registered_modules,
        installed_plugins=ATT.installed(),
    )


_att_backends_loaded = False


def ensure_att_backends_loaded() -> None:
    global _att_backends_loaded
    if _att_backends_loaded:
        return

    _load_att_backends(ATT.get().extra_modules)
    _att_backends_loaded = True
