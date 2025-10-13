import rpyc
import uuid
import inspect
import setproctitle
from typing import Union, Optional
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.embed_cache.impl.naive_memory_cache import InMemoryCache
from rpyc.utils.classic import obtain
from lightllm.utils.envs_utils import get_unique_server_name
import pickle


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

    def exposed_alloc_v2(self, batch_md5_token_nums: bytes) -> bytes:
        """
        batch_md5_token_nums: pickle.dumps([(md5sum, token_num), ...])
        返回: pickle.dumps(records)
        """
        batch_requests = pickle.loads(batch_md5_token_nums)
        md5sum_list = [obtain(md5) for md5, num in batch_requests]
        token_num_list = [obtain(num) for md5, num in batch_requests]
        record = self._impl.alloc(md5sum_list, token_num_list)
        return pickle.dumps(record)

    def exposed_release_v2(self, ids_blob: bytes) -> None:
        ids = pickle.loads(ids_blob)
        ids = [obtain(id) for id in ids]
        return self._impl.release(ids)

    def exposed_set_items_data_v2(self, ids_blob: bytes) -> bytes:
        ids = pickle.loads(ids_blob)
        ids = [obtain(id) for id in ids]
        status_list = self._impl.set_items_data(ids)
        return pickle.dumps(status_list)

    def exposed_get_items_data_v2(self, ids_blob: bytes) -> bytes:
        ids = pickle.loads(ids_blob)
        ids = [obtain(id) for id in ids]
        status_list = self._impl.get_items_data(ids)
        return pickle.dumps(status_list)

    def exposed_set_items_embed_v2(self, ids_blob: bytes) -> None:

        ids = pickle.loads(ids_blob)
        ids = [obtain(id) for id in ids]
        status_list = self._impl.set_items_embed(ids)
        return pickle.dumps(status_list)

    def exposed_get_items_embed_v2(self, ids_blob: bytes) -> bytes:
        ids = pickle.loads(ids_blob)
        ids = [obtain(id) for id in ids]
        status_list = self._impl.get_items_embed(ids)
        return pickle.dumps(status_list)


def start_cache_manager(port: int, args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::cache_manager")
    manager = InMemoryCache(args)
    service = CacheServer(manager)
    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(service, port=port, protocol_config={"allow_pickle": True})
    pipe_writer.send("init ok")
    t.start()


if __name__ == "__main__":
    start_cache_manager(2233)
