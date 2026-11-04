from importlib import import_module


def load_backend_class(backend):
    module_name, class_name = backend.split(":")
    return getattr(import_module(f"{__package__}.{module_name}"), class_name)
