import threading

from tqdm import tqdm


class ThreadProgress(tqdm):
    """Progress owned by one process, without a multiprocessing semaphore."""

    _lock = threading.RLock()
