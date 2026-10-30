import socket
import zlib
from typing import Any

from rpyc.core.channel import Channel
from rpyc.core.stream import SocketStream
from rpyc.utils.server import ThreadedServer

from .rpc_payload import MAX_RPC_FRAME_BYTES


class BoundedChannel(Channel):
    """Reject oversized RPyC frames before reading or decompressing their payloads."""

    __slots__ = ("_max_message_size",)

    def __init__(self, stream: SocketStream, max_message_size: int) -> None:
        """Create a channel with a maximum serialized frame size."""
        if type(max_message_size) is not int or max_message_size <= 0:
            raise ValueError("max_message_size must be a positive integer")
        super().__init__(stream)
        self._max_message_size = max_message_size

    def recv(self) -> bytes:
        """Read one frame while bounding both compressed and expanded data."""
        header = self.stream.read(self.FRAME_HEADER.size)
        length, compressed = self.FRAME_HEADER.unpack(header)
        if length > self._max_message_size:
            self.stream.close()
            raise EOFError("visual RPC frame exceeds the configured byte limit")

        data = self.stream.read(length + len(self.FLUSHER))[: -len(self.FLUSHER)]
        if compressed:
            decompressor = zlib.decompressobj()
            try:
                data = decompressor.decompress(data, self._max_message_size + 1)
            except zlib.error as error:
                self.stream.close()
                raise EOFError("invalid compressed visual RPC frame") from error
            if (
                len(data) > self._max_message_size
                or decompressor.unconsumed_tail
                or not decompressor.eof
                or decompressor.unused_data
            ):
                self.stream.close()
                raise EOFError("visual RPC frame exceeds the configured byte limit or is invalid")
        return data


class VisualOnlyThreadedServer(ThreadedServer):
    """Threaded RPyC server that applies the visual endpoint's frame limit."""

    def _serve_client(self, sock: socket.socket, credentials: Any) -> None:
        """Serve one client using the bounded channel instead of RPyC's default channel."""
        addrinfo = sock.getpeername()
        if credentials:
            self.logger.info(f"welcome {addrinfo} ({credentials!r})")
        else:
            self.logger.info(f"welcome {addrinfo}")

        try:
            config = dict(
                self.protocol_config,
                credentials=credentials,
                endpoints=(sock.getsockname(), addrinfo),
                logger=self.logger,
            )
            # RPyC 5.3.1 hard-codes Channel in Server._serve_client.
            channel = BoundedChannel(SocketStream(sock), config.get("max_message_size", MAX_RPC_FRAME_BYTES))
            conn = self.service._connect(channel, config)
            self._handle_connection(conn)
        finally:
            self.logger.info(f"goodbye {addrinfo}")
