"""Write-buffer limits for PD control WebSockets."""

import inspect
from typing import Any, Optional

from starlette.websockets import WebSocket

from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size

# One 12k control message is about 100 KiB. 256 KiB lets that send return
# without waiting for the socket, and the 32 KiB gap keeps a full buffer from
# holding token traffic for megabytes.
PD_WEBSOCKET_WRITE_LIMIT = (256 * 1024, 256 * 1024 - 32 * 1024)


def websocket_message_exceeds_limit(payload: bytes) -> bool:
    return len(payload) > get_lightllm_websocket_max_message_size()


def set_pd_websocket_write_buffer(websocket: WebSocket) -> bool:
    """Raise the write buffer when the server exposes the transport.

    Hypercorn 0.18 doesn't expose the accepted transport through ASGI. The
    transport is discovered on a best-effort basis without assuming a fixed
    middleware depth. If a custom stack hides it, keep the default limits and
    let the WebSocket continue normally.
    """
    send = getattr(websocket, "_send", None)
    if send is None:
        return False
    transport = _find_transport(send)
    if transport is None:
        return False
    high, low = PD_WEBSOCKET_WRITE_LIMIT
    transport.set_write_buffer_limits(high=high, low=low)
    return True


def _find_transport(value: Any, seen: Optional[set[int]] = None, depth: int = 0):
    """Find an asyncio transport without depending on wrapper depth."""
    if value is None or depth > 12:
        return None

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return None
    seen.add(value_id)

    if callable(getattr(value, "set_write_buffer_limits", None)) and callable(
        getattr(value, "get_write_buffer_size", None)
    ):
        return value

    if inspect.ismethod(value):
        transport = _find_transport(value.__self__, seen, depth + 1)
        if transport is not None:
            return transport
    elif inspect.isfunction(value):
        for cell in value.__closure__ or ():
            transport = _find_transport(cell.cell_contents, seen, depth + 1)
            if transport is not None:
                return transport

    if isinstance(value, dict):
        for item in value.values():
            transport = _find_transport(item, seen, depth + 1)
            if transport is not None:
                return transport
        return None
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            transport = _find_transport(item, seen, depth + 1)
            if transport is not None:
                return transport
        return None

    for attribute in (
        "__self__",
        "send",
        "app_send",
        "writer",
        "transport",
        "protocol",
    ):
        try:
            child = getattr(value, attribute)
        except (AttributeError, RuntimeError):
            continue
        transport = _find_transport(child, seen, depth + 1)
        if transport is not None:
            return transport
    return None
