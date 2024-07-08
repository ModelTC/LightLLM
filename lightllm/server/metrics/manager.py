import asyncio
import rpyc
import time
import threading
from typing import Union
from rpyc.utils.classic import obtain
from .metrics import Monitor
from prometheus_client import generate_latest
import multiprocessing.shared_memory as shm
from concurrent.futures import ThreadPoolExecutor

async_metric_server = None
from rpyc import async_


class MetricServer(rpyc.Service):
    def __init__(self, args) -> None:
        super().__init__()
        self.monitor = Monitor(args)
        self.interval = args.push_interval

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_counter_inc(self, name: str, label: str = None) -> None:
        return self.monitor.counter_inc(name, label)

    def exposed_histogram_observe(self, name: str, value: float, label: str = None) -> None:
        return self.monitor.histogram_observe(name, value, label)

    def exposed_gauge_set(self, name: str, value: float) -> None:
        return self.monitor.gauge_set(name, value)

    def exposed_generate_latest(self) -> bytes:
        data = generate_latest(self.monitor.registry)
        return data

    def push_metrics(self):
        while True:
            self.monitor.push_metrices()
            time.sleep(self.interval)


class MetricClient:
    def __init__(self, port):
        self.port = port
        self.conn = rpyc.connect("localhost", self.port)
        self.counter_inc = async_(self.conn.root.counter_inc)
        self.histogram_observe = async_(self.conn.root.histogram_observe)
        self.gauge_set = async_(self.conn.root.gauge_set)

        def async_wrap(f):
            f = rpyc.async_(f)

            async def _func(*args, **kwargs):
                ans = f(*args, **kwargs)
                await asyncio.to_thread(ans.wait)
                return ans.value

            return _func

        self._generate_latest = async_wrap(self.conn.root.generate_latest)

    async def generate_latest(self):
        ans = await self._generate_latest()
        return ans


def start_metric_manager(port: int, args, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    service = MetricServer(args)
    if args.metric_gateway is not None:
        push_thread = threading.Thread(target=service.push_metrics)
        push_thread.start()  # 启动推送任务线程

    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(service, port=port)
    pipe_writer.send("init ok")
    t.start()
