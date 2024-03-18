# import faulthandler
# faulthandler.enable()

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory


class SharedArray:
    def __init__(self, name, shape, dtype):
        dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=np.prod(shape) * dtype_byte_num)
            print(f"create shm {name}")
        except:
            shm = shared_memory.SharedMemory(name=name, create=False, size=np.prod(shape) * dtype_byte_num)
            print(f"link shm {name}")
        self.shm = shm  # SharedMemory 对象一定要被持有，否则会被释放
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)


class SharedTreeInfoNode:
    def __init__(self, manager, idx) -> None:
        self.manager = manager
        self.idx = idx

    def get_idx(self):
        return self.idx

    def get_parent_idx(self):
        return self.manager._values[self.idx, 3]

    def set_parent_idx(self, p_idx):
        self.manager._values[self.idx, 3] = p_idx

    def get_parent_idx_shared_node(self):
        return SharedTreeInfoNode(self.manager, self.manager._values[self.idx, 3])

    def get_node_value_len(self):
        return self.manager._values[self.idx, 4]

    def set_node_value_len(self, value_len):
        self.manager._values[self.idx, 4] = value_len

    def get_node_prefix_total_len(self):
        return self.manager._values[self.idx, 5]

    def set_node_prefix_total_len(self, prefix_total_len):
        self.manager._values[self.idx, 5] = prefix_total_len


class SharedLinkedListManager:
    VALUE_INDEX = 0
    PRE_INDEX = 1
    NEXT_INDEX = 2

    def __init__(self, unique_name, total_token_num, tp_id) -> None:
        self.size = total_token_num + 2  # 因为 0 号节点不分配，所以为了满足充分可用性需要 + 2.
        # 第二维对应信息 0 idx 1 pre index 2 next index 用于链表管理  3 tree_node parent node idx
        # 4 tree_node value len 5 tree node prefix total len
        self._shm_array = SharedArray(f"{unique_name} SharedLinkedList_{tp_id}", shape=(self.size, 6), dtype=np.int64)
        self._values = self._shm_array.arr
        # idx
        self._values[:, self.VALUE_INDEX] = np.arange(0, self.size, 1)
        # pre
        self._values[0, self.PRE_INDEX] = -1
        self._values[1:, self.PRE_INDEX] = np.arange(0, self.size - 1, 1)
        # next
        self._values[0 : self.size - 1, self.NEXT_INDEX] = np.arange(1, self.size, 1)
        self._values[self.size - 1, self.NEXT_INDEX] = -1

        # tree node value
        self._values[:, 3] = -1
        self._values[:, 4] = 0
        self._values[:, 5] = 0

    def alloc(self):
        if self._values[0, self.NEXT_INDEX] != -1:
            alloc_idx = self._values[0, self.NEXT_INDEX]
            if self._values[alloc_idx, self.NEXT_INDEX] == -1:
                self._values[0, self.NEXT_INDEX] = -1
                ans = SharedTreeInfoNode(self, alloc_idx)

            nn_idx = self._values[alloc_idx, self.NEXT_INDEX]
            self._values[0, self.NEXT_INDEX] = nn_idx
            self._values[nn_idx, self.PRE_INDEX] = 0
            ans = SharedTreeInfoNode(self, alloc_idx)
            # 初始化值
            ans.set_parent_idx(-1)
            ans.set_node_value_len(0)
            ans.set_node_prefix_total_len(0)
            return ans

        assert False, "error cannot alloc"

    def free(self, idx):
        nn_idx = self._values[0, self.NEXT_INDEX]
        self._values[0, self.NEXT_INDEX] = idx
        self._values[idx, self.PRE_INDEX] = 0
        self._values[idx, self.NEXT_INDEX] = nn_idx
        if nn_idx != -1:
            self._values[nn_idx, self.PRE_INDEX] = idx
        return

    def can_alloc_num(self):
        num = 0
        cur_loc = 0
        while self._values[cur_loc, self.NEXT_INDEX] != -1:
            num += 1
            cur_loc = self._values[cur_loc, self.NEXT_INDEX]
        return num

    def get_shared_node(self, idx):
        return SharedTreeInfoNode(self, idx)


if __name__ == "__main__":
    # test SharedArray
    a = SharedArray("sb_abc", (1,), dtype=np.int32)
    a.arr[0] = 10
    assert a.arr[0] == 10
    a.arr[0] += 10
    assert a.arr[0] == 20

    # test SharedTreeIdxManager
    mananger = SharedLinkedListManager("unique_name", 100, 0)
    node1 = mananger.alloc()
    node1.set_parent_idx(10)
    assert node1.get_parent_idx() == 10
    node1.set_node_value_len(10)
    assert node1.get_node_value_len() == 10
    node1.set_node_prefix_total_len(100)
    assert node1.get_node_prefix_total_len() == 100
    mananger.free(node1.get_idx())
    alloc_nodes = []
    for _ in range(101):
        node1 = alloc_nodes.append(mananger.alloc())

    try:
        node_tmp = mananger.alloc()
    except:
        assert True

    for e in alloc_nodes:
        mananger.free(e.get_idx())

    alloc_nodes = []
    for _ in range(101):
        node1 = alloc_nodes.append(mananger.alloc())

    try:
        node_tmp = mananger.alloc()
    except:
        assert True
