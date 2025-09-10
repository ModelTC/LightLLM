from time import perf_counter
from contextlib import contextmanager


@contextmanager
def timer(tag):
    t0 = perf_counter()
    try:
        yield
    finally:
        took_time = (perf_counter() - t0) * 1000
        print(f"[load-timer] {tag}: {took_time:.1f} ms", flush=True)
