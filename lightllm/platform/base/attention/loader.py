import importlib

_BUILTIN_ATT_MODULES: tuple[str, ...] = (
    "lightllm.common.basemodel.attention.triton.fp",
    "lightllm.common.basemodel.attention.triton.int4kv",
    "lightllm.common.basemodel.attention.triton.int8kv",
    "lightllm.common.basemodel.attention.triton.mla",
    "lightllm.common.basemodel.attention.fa3.fp",
    "lightllm.common.basemodel.attention.paged_fa3.fp",
    "lightllm.common.basemodel.attention.fa3.fp8",
    "lightllm.common.basemodel.attention.fa3.mla",
    "lightllm.common.basemodel.attention.flashinfer.fp",
    "lightllm.common.basemodel.attention.flashinfer.fp8",
    "lightllm.common.basemodel.attention.flashinfer.mla",
    "lightllm.common.basemodel.attention.nsa.flashmla_sparse",
    "lightllm.common.basemodel.attention.nsa.fp8_flashmla_sparse",
)

_att_backends_loaded = False


def _load_builtin_att_backends() -> None:
    for module_name in _BUILTIN_ATT_MODULES:
        importlib.import_module(module_name)


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
    try:
        from importlib.metadata import entry_points as eps_fn

        available = sorted(ep.name for ep in eps_fn(group="lightllm.att_plugins"))
    except Exception:
        available = []
    if available:
        hints.append(f"Installed att plugins: {available}.")
    else:
        hints.append(
            "No att plugins installed; use --extra_att_plugins <name> with a package "
            "registered under entry point group 'lightllm.att_plugins'."
        )
    return " ".join(hints)


def ensure_att_backends_loaded() -> None:
    """ Load attention backends once per process (idempotent). """
    global _att_backends_loaded
    if _att_backends_loaded:
        return

    from lightllm.platform.plugins import configure_att_plugins, get_att_plugin_config

    configure_att_plugins()
    _load_att_backends(get_att_plugin_config().extra_modules)
    _att_backends_loaded = True
