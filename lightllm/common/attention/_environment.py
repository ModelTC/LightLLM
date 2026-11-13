import importlib

from .api.backend import UnsupportedAttention


def check_attention_environment(caps, *, provider, supported_platforms, min_device_capability, dependency):
    if caps.platform not in supported_platforms:
        raise UnsupportedAttention(f"{provider} does not support platform {caps.platform.value}")
    if not caps.available:
        raise UnsupportedAttention(f"device {caps.device} is unavailable")
    if min_device_capability is not None and (caps.capability is None or caps.capability < min_device_capability):
        raise UnsupportedAttention(f"requires device capability {min_device_capability}")
    try:
        if dependency:
            importlib.import_module(dependency)
    except (ImportError, OSError) as exc:
        raise UnsupportedAttention(str(exc)) from exc
