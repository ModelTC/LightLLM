from collections import OrderedDict
from lightllm.utils.dist_utils import get_current_device_id


class ImageCacheManager:
    def __init__(self):
        """
        Initialize the image cache manager with a simple GPU cache and an LRU CPU cache.
        """
        self._gpu_cache = dict()
        self._cpu_cache = OrderedDict()

    def set_max_size(self, max_size: int):
        """
        Set the maximum number of items to keep in the CPU cache.
        :param max_size: Maximum number of items to keep in the CPU cache.
        """
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")
        self._max_size = max_size

    def set_embed(self, uuid, embed):
        """
        Store the embedding for the given uuid in the GPU cache.
        :param uuid: Unique identifier for the image
        :param embed: Embedding vector for the image (on GPU)
        """
        self._gpu_cache[uuid] = embed

    def get_embed(self, uuid):
        """
        Retrieve the embedding for the given uuid. Prefer GPU cache,
        otherwise return CPU cache and move to GPU (simulate .cuda()).
        :param uuid: Unique identifier for the image
        :return: Embedding vector (on GPU if possible, else move from CPU to GPU)
        """
        if uuid in self._gpu_cache:
            return self._gpu_cache[uuid]
        elif uuid in self._cpu_cache:
            self._cpu_cache.move_to_end(uuid)
            embed = self._cpu_cache[uuid].cuda(get_current_device_id())
            return embed
        return None

    def query_embed(self, uuid):
        """
        Query if the embedding for the given uuid is in the cache.
        :param uuid: Unique identifier for the image
        :return: True if the embedding is in the cache, False otherwise
        """
        return uuid in self._gpu_cache or uuid in self._cpu_cache

    def filter(self, uuid_list):
        """
        Given a list of uuids, move their embeddings from GPU cache to CPU cache if present,
        and return a dict of those found in the cache and their embeddings (on CPU).
        :param uuid_list: List of uuids
        """
        for uuid in uuid_list:
            if uuid in self._gpu_cache:
                embed_cpu = self._gpu_cache[uuid].cpu()
                # Move to CPU cache and remove from GPU cache
                self._gpu_cache.pop(uuid)
                if uuid in self._cpu_cache:
                    self._cpu_cache.move_to_end(uuid)
                self._cpu_cache[uuid] = embed_cpu
                if len(self._cpu_cache) > self._max_size:
                    self._cpu_cache.popitem(last=False)
            elif uuid in self._cpu_cache:
                self._cpu_cache.move_to_end(uuid)
        print(self._gpu_cache.keys())
        print(self._cpu_cache.keys())
        return


image_cache_manager = ImageCacheManager()
