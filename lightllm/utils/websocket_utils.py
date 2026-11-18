"""Write-buffer limits shared by PD WebSocket clients and masters."""

# websockets 默认 write_limit 是 32 KiB，低水位是高水位的 1/4。
PD_WEBSOCKET_WRITE_LIMIT = (32 * 1024, 8 * 1024)


def configure_pd_master_write_buffer():
    """Configure Hypercorn asyncio connections before the master starts listening."""
    from hypercorn.asyncio import run
    from hypercorn.asyncio.tcp_server import TCPServer

    class PDMasterTCPServer(TCPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            high, low = PD_WEBSOCKET_WRITE_LIMIT
            self.writer.transport.set_write_buffer_limits(high=high, low=low)

    # Hypercorn 0.18 has no write-limit setting. worker_serve looks up this
    # connection class after ASGI lifespan startup, for every accepted socket.
    run.TCPServer = PDMasterTCPServer
