from collections import OrderedDict


class ImageCacheManager:
    def __init__(self):
        """
        Initialize the image cache manager with a simple LRU CPU cache.
        """
        self._cpu_cache = OrderedDict()
        self._max_size = 10000

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
        if len(self._cpu_cache) >= self._max_size:
            self._cpu_cache.popitem(last=False)
        self._cpu_cache[uuid] = embed.to("cpu", non_blocking=True)

    def get_embed(self, uuid):
        """
        Retrieve the embedding for the given uuid. Prefer GPU cache,
        otherwise return CPU cache and move to GPU (simulate .cuda()).
        :param uuid: Unique identifier for the image
        :return: Embedding vector (on GPU if possible, else move from CPU to GPU)
        """
        if uuid in self._cpu_cache:
            self._cpu_cache.move_to_end(uuid)
            embed = self._cpu_cache[uuid]
            return embed.cuda(non_blocking=True)
        return None

    def query_embed(self, uuid):
        """
        Query if the embedding for the given uuid is in the cache.
        :param uuid: Unique identifier for the image
        :return: True if the embedding is in the cache, False otherwise
        """
        return uuid in self._cpu_cache


image_cache_manager = ImageCacheManager()
