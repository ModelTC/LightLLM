import sys
import errno
import socket
from rpyc.lib import socket_backoff_connect
from rpyc.lib.compat import get_exc_errno
from rpyc.core.stream import SocketStream
from rpyc.utils.server import Server
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_BUFF_SIZE = 4 * 1024 * 1024


def _fix_connect(
    cls,
    host,
    port,
    family=socket.AF_INET,
    socktype=socket.SOCK_STREAM,
    proto=0,
    timeout=3,
    nodelay=False,
    keepalive=False,
    attempts=6,
):
    family, socktype, proto, _, sockaddr = socket.getaddrinfo(host, port, family, socktype, proto)[0]
    s = socket_backoff_connect(family, socktype, proto, sockaddr, timeout, attempts)
    try:
        if nodelay:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        old_s_buf = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        old_r_buf = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

        logger.info(f"change socket buffer from {old_s_buf} {old_r_buf} change to {_BUFF_SIZE}")

        # set buffer
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _BUFF_SIZE)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _BUFF_SIZE)

        if keepalive:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # Linux specific: after <keepalive> idle seconds, start sending keepalives every <keepalive> seconds.
            is_linux_socket = hasattr(socket, "TCP_KEEPIDLE")
            is_linux_socket &= hasattr(socket, "TCP_KEEPINTVL")
            is_linux_socket &= hasattr(socket, "TCP_KEEPCNT")
            if is_linux_socket:
                # Drop connection after 5 failed keepalives
                # `keepalive` may be a bool or an integer
                if keepalive is True:
                    keepalive = 60
                if keepalive < 1:
                    raise ValueError("Keepalive minimal value is 1 second")

                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, keepalive)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, keepalive)
        return s
    except BaseException:
        s.close()
        raise


SocketStream._connect = classmethod(_fix_connect)


def fix_connect(cls, host, port, **kwargs):
    if kwargs.pop("ipv6", False):
        kwargs["family"] = socket.AF_INET6
    kwargs["nodelay"] = True
    return cls(cls._connect(host, port, **kwargs))


SocketStream.connect = classmethod(fix_connect)


def fix_unix_connect(cls, path, timeout=3):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout)
        old_s_buf = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        old_r_buf = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

        logger.info(f"change socket buffer from {old_s_buf} {old_r_buf} change to {_BUFF_SIZE}")

        # set buffer
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _BUFF_SIZE)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _BUFF_SIZE)

        s.connect(path)
        return cls(s)
    except BaseException:
        s.close()
        raise


SocketStream.unix_connect = classmethod(fix_unix_connect)


def fix_accept(self):
    """accepts an incoming socket connection (blocking)"""
    while self.active:
        try:
            sock, addrinfo = self.listener.accept()
            if str(sock.family) != "AddressFamily.AF_UNIX":
                logger.info("set nodelay mode")
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            old_s_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            old_r_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

            logger.info(f"change socket buffer from {old_s_buf} {old_r_buf} change to {_BUFF_SIZE}")

            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _BUFF_SIZE)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _BUFF_SIZE)
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
