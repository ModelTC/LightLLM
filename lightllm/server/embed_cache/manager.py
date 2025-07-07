import rpyc
import uuid
import inspect
from typing import Union
from lightllm.utils.graceful_utils import graceful_registry
from .interface import CacheManager
from rpyc.utils.classic import obtain


class CacheServer(rpyc.Service):
    def __init__(self, manager_impl: CacheManager) -> None:
        super().__init__()
        self._impl = manager_impl

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_alloc_batch(self, md5sum_list: list[str], token_num_list: list[int]) -> dict:
        md5sum_list = obtain(md5sum_list)
        token_num_list = obtain(token_num_list)
        record = self._impl.alloc(md5sum_list, token_num_list)
        return record

    def exposed_release(self, id: int) -> None:
        id = obtain(id)
        return self._impl.release(id)

    def exposed_set_items_data(self, ids: list[int]) -> None:
        ids = obtain(ids)
        return self._impl.set_items_data(ids=ids)

    def exposed_get_items_data(self, ids: list[int]) -> list[bool]:
        ids = obtain(ids)
        return self._impl.get_items_data(ids=ids)

    def exposed_set_item_embed(self, id: int) -> None:
        id = obtain(id)
        return self._impl.set_item_embed(id=id)

    def exposed_get_item_embed(self, id: int) -> bool:
        id = obtain(id)
        return self._impl.get_item_embed(id=id)


def start_cache_manager(port: int, args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    from .interface import CacheManagerFactory

    manager_cls = CacheManagerFactory.get_impl("naive")
    manager = manager_cls(args)
    service = CacheServer(manager)
    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(service, port=port)
    pipe_writer.send("init ok")
    t.start()


if __name__ == "__main__":
    start_cache_manager(2233)
