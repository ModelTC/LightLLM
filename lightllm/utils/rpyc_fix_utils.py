import sys
import errno
import socket
from rpyc.lib.compat import get_exc_errno
from rpyc.core.stream import SocketStream
from rpyc.utils.server import Server


def fix_connect(cls, host, port, **kwargs):
    if kwargs.pop("ipv6", False):
        kwargs["family"] = socket.AF_INET6
    kwargs["nodelay"] = True
    return cls(cls._connect(host, port, **kwargs))


SocketStream.connect = classmethod(fix_connect)


def fix_accept(self):
    """accepts an incoming socket connection (blocking)"""
    while self.active:
        try:
            sock, addrinfo = self.listener.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except socket.timeout:
            pass
        except socket.error:
            ex = sys.exc_info()[1]
            if get_exc_errno(ex) in (errno.EINTR, errno.EAGAIN):
                pass
            else:
                raise EOFError()
        else:
            break

    if not self.active:
        return

    sock.setblocking(True)
    self.logger.info(f"accepted {addrinfo} with fd {sock.fileno()}")
    self.clients.add(sock)
    self._accept_method(sock)


Server.accept = fix_accept
