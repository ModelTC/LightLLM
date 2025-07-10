import dataclasses
import threading
from threading import Event
from typing import Optional

# 通过开启两个线程进行推理，通过overlapEvent的管理，实现batch的折叠推理效果，对于不支持
# 折叠的模式，只需要调整调用的位置即可。将推理分为三个阶段进行事件同步处理
# 1. forward (包含生成模型输入和模型推理)
# 2. pre_post_handle (预更新InferReq上的部分状态，使可以完成下一个batch的模型输入初始化)
# 3. post_handle (包含复杂的各种停止判断等处理)


class OverlapEventManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.overlap_pack: OverlapEventPack = OverlapEventPack(
            wait_forward_event=Event(),
            notify_post_handle_event=Event(),
            wait_pre_post_handle_event=Event(),
            notify_forward_event=Event(),
            wait_post_handle_event=Event(),
            notify_pre_post_handle_event=Event(),
        )
        self.overlap_pack.wait_forward_event.set()
        self.overlap_pack.wait_pre_post_handle_event.set()
        return

    def get_overlap_event_pack(self):
        with self.lock:
            new_pack = OverlapEventPack(
                wait_forward_event=self.overlap_pack.notify_forward_event,
                notify_post_handle_event=self.overlap_pack.wait_post_handle_event,
                wait_pre_post_handle_event=self.overlap_pack.notify_pre_post_handle_event,
                notify_forward_event=Event(),
                wait_post_handle_event=Event(),
                notify_pre_post_handle_event=Event(),
            )
            ans_pack = self.overlap_pack
            self.overlap_pack = new_pack

            return ans_pack


@dataclasses.dataclass
class OverlapEventPack:
    wait_forward_event: Optional[threading.Event] = None

    notify_post_handle_event: Optional[threading.Event] = None
    wait_pre_post_handle_event: Optional[threading.Event] = None

    notify_forward_event: Optional[threading.Event] = None
    wait_post_handle_event: Optional[threading.Event] = None

    notify_pre_post_handle_event: Optional[threading.Event] = None

    def wait_to_forward(self):
        self.wait_forward_event.wait()
        return

    def notify_post_handle_and_wait_pre_post_handle(self):
        self.notify_post_handle_event.set()
        self.wait_pre_post_handle_event.wait()
        return

    def notify_forward_and_wait_post_handle(self):
        self.notify_forward_event.set()
        self.wait_post_handle_event.wait()
        return

    def notify_pre_post_handle(self):
        self.notify_pre_post_handle_event.set()
        return
