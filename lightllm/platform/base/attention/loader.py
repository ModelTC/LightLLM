import importlib
import pkgutil

from lightllm.platform.plugins.att import ENTRY_POINT_GROUP
from lightllm.platform.plugins.common import list_installed_plugin_names


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
    _validate_extra_att_modules_loaded(extra_modules)


def _validate_extra_att_modules_loaded(extra_modules: tuple[str, ...]) -> None:
    if not extra_modules:
        return

    from lightllm.platform.base.attention.registry import att_backend_registry

    registered_modules = {
        spec.backend_cls.__module__ for spec in att_backend_registry.list_specs()
    }
    missing = [
        module_name
        for module_name in extra_modules
        if module_name not in registered_modules
    ]
    if not missing:
        return

    hints = _format_extra_att_module_hints()
    raise RuntimeError(
        "External attention modules configured in extra_att_modules / extra_att_plugins "
        f"did not register any @register_att_backend implementations after loading: "
        f"{missing}. {hints}"
    )


def _format_extra_att_module_hints() -> str:
    hints = [
        "Check that --extra_att_modules / plugin extra_modules import paths are correct "
        "and modules define @register_att_backend."
    ]
    available = list_installed_plugin_names(ENTRY_POINT_GROUP)
    if available:
        hints.append(f"Installed att plugins: {available}.")
    else:
        hints.append(
            "No att plugins installed; use --extra_att_plugins <name> with a package "
            f"registered under entry point group {ENTRY_POINT_GROUP!r}."
        )
    return " ".join(hints)


_att_backends_loaded = False


def ensure_att_backends_loaded() -> None:
    """ Load attention backends once per process (idempotent). """
    global _att_backends_loaded
    if _att_backends_loaded:
        return

    from lightllm.platform.plugins import configure_att_plugins, get_att_plugin_config

    configure_att_plugins()
    _load_att_backends(get_att_plugin_config().extra_modules)
    _att_backends_loaded = True
