import threading
import torch.multiprocessing as mp
from queue import Empty


def join_if_alive(thread: threading.Thread):
    if thread is not None and thread.is_alive():
        try:
            thread.join()
        except Exception:
            pass
    return


def clear_queue(queue: mp.Queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except Empty:
            break
