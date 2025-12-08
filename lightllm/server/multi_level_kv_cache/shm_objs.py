import ctypes
import numpy as np
from multiprocessing import shared_memory
from typing import List, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.auto_shm_cleanup import register_posix_shm_for_cleanup

logger = init_logger(__name__)


class IntList(object):
    def __init__(self, name: str, capacity: int, init_shm_data: bool):
        self.capacity: int = capacity
        byte_size = np.dtype(np.int32).itemsize * (self.capacity + 1)
        shm_name = name
        shm = _create_shm(name=shm_name, byte_size=byte_size)
        self.shm = shm

        if self.shm.size != byte_size:
            logger.info(f"size not same, unlink lock shm {self.shm.name} and create again")
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            self.shm = _create_shm(name=shm_name, byte_size=byte_size)

        self.arr = np.ndarray((self.capacity + 1), dtype=np.int32, buffer=self.shm.buf)
        if init_shm_data:
            self.arr.fill(0)
        return

    def size(self):
        return self.arr[-1]

    def add_item(self, value: int):
        write_index = self.arr[-1]
        self.arr[write_index] = value
        self.arr[-1] += 1
        return

    def add_items(self, values: List[int]):
        write_index = self.arr[-1]
        n = len(values)
        self.arr[write_index : write_index + n] = values
        self.arr[-1] += n
        return

    def pop_all_item(self) -> Optional[List[int]]:
        if self.size() == 0:
            return None

        ans = self.arr[0 : self.size()].tolist()
        self.arr[-1] = 0
        return ans


class ShmLinkedList(object):
    def __init__(self, name: str, item_class: "_LinkedListItem.__class__", capacity: int, init_shm_data: bool):
        self.capacity: int = capacity
        # add head and tail node.
        byte_size = ctypes.sizeof(item_class) * (self.capacity + 2)
        shm_name = name
        shm = _create_shm(name=shm_name, byte_size=byte_size)
        self.shm = shm

        if self.shm.size != byte_size:
            logger.info(f"size not same, unlink lock shm {self.shm.name} and create again")
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            self.shm = _create_shm(name=shm_name, byte_size=byte_size)
        # 构建 hash table 表
        self.linked_items: List[_LinkedListItem] = (item_class * (self.capacity + 2)).from_buffer(self.shm.buf)
        # 如果不转变存储，set_list_obj 的对象上绑定的非shm信息在下一次从 shm 中获取对象时将丢失
        self.linked_items = [item for item in self.linked_items]
        for e in self.linked_items:
            e.set_list_obj(self)

        self.head = self.linked_items[self.capacity]
        self.tail = self.linked_items[self.capacity + 1]

        if init_shm_data:
            for e in self.linked_items:
                e.init()

            self.head.self_index = self.capacity
            self.tail.self_index = self.capacity + 1
            self.head.next_index = self.tail.self_index
            self.tail.pre_index = self.head.self_index

            for i in range(self.capacity):
                item = self.linked_items[i]
                item.self_index = i
                self.add_item_to_tail(i)
        return

    def add_item_to_tail(self, index: int):
        item = self.linked_items[index]
        pre_node = self.linked_items[self.tail.pre_index]
        pre_node.next_index = item.self_index
        item.pre_index = pre_node.self_index
        item.next_index = self.tail.self_index
        self.tail.pre_index = item.self_index
        return

    def add_item_to_head(self, index: int):
        item = self.linked_items[index]
        next_node = self.linked_items[self.head.next_index]
        next_node.pre_index = item.self_index
        item.next_index = next_node.self_index
        item.pre_index = self.head.self_index
        self.head.next_index = item.self_index
        return

    def get_item_by_index(self, index: int) -> "_LinkedListItem":
        item = self.linked_items[index]
        return item

    def pop_head_item(self) -> "_LinkedListItem":
        head_item = self.linked_items[self.head.next_index]
        if head_item.self_index == self.tail.self_index:
            return None
        head_item.del_self_from_list()
        return head_item


class ShmDict(object):
    def __init__(self, name: str, capacity: int, init_shm_data: bool):
        self.capacity: int = capacity
        self.link_items: ShmLinkedList = ShmLinkedList(
            name=name, item_class=_HashLinkItem, capacity=self.capacity * 2, init_shm_data=init_shm_data
        )
        # 将前capacity个item,作为hash item的链表头。
        if init_shm_data:
            for i in range(self.capacity):
                self.link_items.pop_head_item()
                item: _HashLinkItem = self.link_items.get_item_by_index(i)
                item.pre_index = -1
                item.next_index = -1
        return

    def put(self, key: int, value: int):
        dest_index = key % self.capacity
        hash_item: _HashLinkItem = self.link_items.get_item_by_index(dest_index)
        if hash_item.next_index == -1:  # 空的
            add_link_item: _HashLinkItem = self.link_items.pop_head_item()
            add_link_item.key = key
            add_link_item.value = value
            hash_item.next_index = add_link_item.self_index
            add_link_item.pre_index = hash_item.self_index
            add_link_item.next_index = -1
            return

        # 存在元素，先遍历是否已经存在
        start_link_item: _HashLinkItem = hash_item.get_next_item()
        cur_link_item = start_link_item
        # 找到对应key的元素，并设置对应的value
        while True:
            if cur_link_item.key == key:
                cur_link_item.value = value
                return
            else:
                next_item = cur_link_item.get_next_item()
                if next_item is None:
                    break
                else:
                    cur_link_item = next_item

        # 没有找到时候，直接插入一个新的节点
        add_link_item: _HashLinkItem = self.link_items.pop_head_item()
        add_link_item.key = key
        add_link_item.value = value

        cur_link_item.next_index = add_link_item.self_index
        add_link_item.pre_index = cur_link_item.self_index
        add_link_item.next_index = -1
        return

    def get(self, key: int) -> Optional[int]:
        dest_index = key % self.capacity
        hash_item: _HashLinkItem = self.link_items.get_item_by_index(dest_index)
        if hash_item.next_index == -1:
            return None
        else:
            start_link_item: _HashLinkItem = hash_item.get_next_item()
            cur_link_item = start_link_item
            # 找到对应key的元素，并设置对应的value
            while cur_link_item is not None:
                if cur_link_item.key == key:
                    return cur_link_item.value
                else:
                    cur_link_item = cur_link_item.get_next_item()
            return None

    def remove(self, key: int):
        dest_index = key % self.capacity
        hash_item: _HashLinkItem = self.link_items.get_item_by_index(dest_index)
        if hash_item.next_index == -1:
            logger.warning(f"shm dict not contain key {key}")
            return

        start_link_item: _HashLinkItem = hash_item.get_next_item()
        cur_link_item = start_link_item

        # 找到对应key的元素，并设置对应的value
        while cur_link_item is not None:
            if cur_link_item.key == key:
                break
            else:
                cur_link_item = cur_link_item.get_next_item()

        if cur_link_item is not None:
            # remove item
            pre_item = cur_link_item.get_pre_item()
            pre_item.next_index = cur_link_item.next_index
            if cur_link_item.next_index != -1:
                next_item = cur_link_item.get_next_item()
                next_item.pre_index = pre_item.self_index

            self.link_items.add_item_to_tail(index=cur_link_item.self_index)
        else:
            logger.warning(f"shm dict not contain key {key}")
        return


class _LinkedListItem(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("self_index", ctypes.c_int),
        ("pre_index", ctypes.c_int),
        ("next_index", ctypes.c_int),
    ]

    def __init__(self):
        self.init()

    def init(self):
        self.self_index = -1
        self.pre_index = -1
        self.next_index = -1
        return

    def set_list_obj(self, parent_list: ShmLinkedList):
        self.linked_items = parent_list.linked_items
        return

    def get_next_item(self) -> "_LinkedListItem":
        if self.next_index == -1:
            return None
        return self.linked_items[self.next_index]

    def get_pre_item(self) -> "_LinkedListItem":
        if self.pre_index == -1:
            return None
        return self.linked_items[self.pre_index]

    def del_self_from_list(self):
        pre_node = self.get_pre_item()
        next_node = self.get_next_item()
        pre_node.next_index = next_node.self_index
        next_node.pre_index = pre_node.self_index
        return


class _HashLinkItem(_LinkedListItem):
    _pack_ = 4
    _fields_ = [
        ("key_low", ctypes.c_uint64),  # 128位key的低64位
        ("key_high", ctypes.c_uint64),  # 128位key的高64位
        ("value", ctypes.c_int),
    ]

    def __init__(self):
        self.init()

    def init(self):
        super().init()
        self.key_low = 0
        self.key_high = 0
        self.value = -1

    @property
    def key(self) -> int:
        """获取完整的128位key"""
        return (self.key_high << 64) | self.key_low

    @key.setter
    def key(self, value: int):
        """设置128位key"""
        self.key_low = value & 0xFFFFFFFFFFFFFFFF
        self.key_high = (value >> 64) & 0xFFFFFFFFFFFFFFFF


def _create_shm(name: str, byte_size: int, auto_cleanup: bool = False):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=byte_size)
        if auto_cleanup:
            register_posix_shm_for_cleanup(name)
        logger.info(f"create lock shm {name}")
    except:
        shm = shared_memory.SharedMemory(name=name, create=False, size=byte_size)
        logger.info(f"link lock shm {name}")
    return shm
