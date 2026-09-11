"""Resolve an output capability to its autonomous server runtime."""

import importlib


_MODE_MODULES = {
    "action": "lightllm.server.generationserver.action",
    "image": "lightllm.server.generationserver.image",
}


def _load(mode: str):
    path = _MODE_MODULES.get(mode)
    if path is None:
        raise ValueError(f"unsupported generation mode: {mode!r}")
    return importlib.import_module(path)


def configure_generation(args, mode: str) -> None:
    _load(mode).configure_defaults(args)


def create_generation_client(args, mode: str):
    return _load(mode).create_client(args)


def get_generation_start_process(mode: str):
    return _load(mode).get_start_process()
