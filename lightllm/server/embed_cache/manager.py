import rpyc
import uuid
import inspect
from typing import Union, Optional
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.embed_cache.impl.naive_memory_cache import InMemoryCache
from lightllm.server.embed_cache.impl.memory_cache_with_redis import MemoryCacheWithRedis
from rpyc.utils.classic import obtain


class CacheServer(rpyc.Service):
    def __init__(self, manager_impl: InMemoryCache) -> None:
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

    def exposed__check_and_set_new_id_range(self, token_num: int) -> int:
        token_num = obtain(token_num)
        return self._impl._check_and_set_new_id_range(token_num)

    def exposed_alloc(self, md5sum_list: list[str], token_num_list: list[int]) -> Optional[list[dict]]:
        md5sum_list = obtain(md5sum_list)
        token_num_list = obtain(token_num_list)
        record = self._impl.alloc(md5sum_list, token_num_list)
        return record

    def exposed_release(self, ids: list[int]) -> None:
        ids = obtain(ids)
        return self._impl.release(ids)

    def exposed_set_items_data(self, ids: list[int]) -> None:
        ids = obtain(ids)
        return self._impl.set_items_data(ids)

    def exposed_get_items_data(self, ids: list[int]) -> list[bool]:
        ids = obtain(ids)
        return self._impl.get_items_data(ids)

    def exposed_set_items_embed(self, ids: list[int]) -> None:
        ids = obtain(ids)
        return self._impl.set_items_embed(ids)

    def exposed_get_items_embed(self, ids: list[int]) -> list[bool]:
        ids = obtain(ids)
        return self._impl.get_items_embed(ids)


def get_cache_manager(args):
    if args.enable_remote_vit or args.run_mode == "visual":
        return MemoryCacheWithRedis(args)
    else:
        return InMemoryCache(args)


def start_cache_manager(port: int, args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    manager = get_cache_manager(args)
    service = CacheServer(manager)
    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(service, port=port, protocol_config={"allow_pickle": True})
    pipe_writer.send("init ok")
    t.start()


if __name__ == "__main__":
    start_cache_manager(2233)
