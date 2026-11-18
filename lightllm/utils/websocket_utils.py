"""Write-buffer limits for PD control WebSockets."""

from starlette.websockets import WebSocket

# One 12k control message is about 100 KiB. 256 KiB lets that send return
# without waiting for the socket, and the 32 KiB gap keeps a full buffer from
# holding token traffic for megabytes.
PD_WEBSOCKET_WRITE_LIMIT = (256 * 1024, 256 * 1024 - 32 * 1024)


def set_pd_websocket_write_buffer(websocket: WebSocket) -> None:
    """Raise the write buffer on this PD control socket. HTTP sockets stay unchanged."""
    send = getattr(websocket, "_send", None)
    if send is None:
        return
    transport = _pd_websocket_transport(send)
    high, low = PD_WEBSOCKET_WRITE_LIMIT
    transport.set_write_buffer_limits(high=high, low=low)


def _unwrap_send(send: object) -> object:
    for cell in send.__closure__:
        inner = cell.cell_contents
        if callable(inner):
            return inner
    raise LookupError(send)


def _pd_websocket_transport(send: object):
    # ExceptionMiddleware.sender -> _AccessLogMiddleware.send_wrapper
    # -> WSStream.app_send -> protocol.stream_send -> TCPServer.protocol_send.
    try:
        access_log_send = _unwrap_send(send=send)
        app_send = _unwrap_send(send=access_log_send)
        return app_send.__self__.send.__self__.send.__self__.writer.transport
    except Exception as error:
        raise RuntimeError("PD control websocket transport is unavailable") from error
