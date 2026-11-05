"""Instance-owned metadata caches; copies rebuild queries from their own resources."""

from functools import wraps
from weakref import ref


class CachedQueries:
    def clear_query_cache(self):
        self.__dict__.pop("_query_cache", None)

    def __getstate__(self):
        return {key: value for key, value in vars(self).items() if key != "_query_cache"}


def cached_query(function):
    @wraps(function)
    def query(instance, *args, **kwargs):
        cache = instance.__dict__.setdefault("_query_cache", {})
        key = (function.__qualname__, args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = function(instance, *args, **kwargs)
        return cache[key]

    return query


class ResourceOperator(CachedQueries):
    def __init__(self, owner):
        self._owner_ref = ref(owner)

    @property
    def owner(self):
        owner = self._owner_ref()
        if owner is None:
            raise ReferenceError("the resource manager has been released")
        return owner
