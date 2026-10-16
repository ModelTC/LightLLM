from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Optional, Tuple

from sortedcontainers import SortedDict


@dataclass(slots=True)
class PromptCacheMatchResult:
    """prefix_match 的返回结果。"""

    # 匹配终点关联的 prefill 连接标识（connection_key），可区分同一服务的不同 DP 连接；
    # 匹配停在 root 时为 None。该标识记录历史处理位置，不保证 KV cache 仍然驻留。
    prefill_node: Optional[str]
    # 匹配到的原始 prompt 字符数：从 text[0] 到最后一个命中采样点（含）的连续前缀长度。
    matched_char_count: int
    # 输入 prompt 的原始字符数 len(text)。
    input_char_count: int
    # 命中节点最近一次由 prompt 插入更新的单调时钟时间戳；未命中时为 None。
    last_insert_time: Optional[float]


@dataclass(slots=True)
class _PromptCacheNode:
    """压缩边节点；key 中所有采样字符共享处理位置与插入时间。"""

    children: Dict[str, "_PromptCacheNode"] = field(default_factory=dict)
    parent: Optional["_PromptCacheNode"] = None
    key: str = ""
    last_prefill_node: Optional[str] = None
    last_insert_time: Optional[float] = None
    # 边末端字符的 LRU 序号；向边起点回退时每个字符加 1。
    last_time_mark: int = 0


class PromptCacheTree:
    """
    用于 cache-aware 选点的 prompt 前缀缓存树。

    prompt 先按 sample_stride 抽稀成 key，再按压缩边建树/匹配。
    recursion_limit 在初始化时通过 sys.setrecursionlimit 调大 Python 调用栈深度。
    容量按采样字符数计算；超限时按 LRU 删除或裁剪叶边。
    """

    def __init__(
        self,
        sample_stride: int = 512,
        max_node_count: int = 1_000_000,
        evict_node_batch: int = 10_000,
        recursion_limit: int = 4000,
    ) -> None:
        """
        Args:
            sample_stride: 每隔多少个字符抽 1 个作为 trie key。
            max_node_count: 树中允许的最大采样字符数（不含 root）；超限时触发 LRU 驱逐。
            evict_node_batch: 每次驱逐时在超出量基础上额外腾出的节点缓冲数。
            recursion_limit: 初始化时通过 sys.setrecursionlimit 设置的调用栈深度上限。
        """
        if sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")
        if max_node_count < 0:
            raise ValueError(f"max_node_count must be >= 0, got {max_node_count}")
        if evict_node_batch < 1:
            raise ValueError(f"evict_node_batch must be >= 1, got {evict_node_batch}")
        if recursion_limit < 1:
            raise ValueError(f"recursion_limit must be >= 1, got {recursion_limit}")
        self.sample_stride = sample_stride
        self.max_node_count = max_node_count
        self.evict_node_batch = evict_node_batch
        self.recursion_limit = recursion_limit
        if recursion_limit > sys.getrecursionlimit():
            sys.setrecursionlimit(recursion_limit)
        self.root = _PromptCacheNode()
        self._node_count = 0
        self._leaf_lru: SortedDict[int, _PromptCacheNode] = SortedDict()
        self._lock = RLock()
        self._time_mark = -1

    def _to_key(self, text: str) -> str:
        return text[:: self.sample_stride]

    def _is_leaf(self, node: _PromptCacheNode) -> bool:
        return node is not self.root and not node.children

    def insert(self, text: str, prefill_node: str) -> None:
        """将 text 的前缀路径写入树，并关联到 prefill_node。

        路径上的节点会更新 last_prefill_node 与 LRU 时间戳；写入后若超
        max_node_count 会立即按 LRU 删除或裁剪叶边。

        Args:
            text: 原始 prompt 文本。
            prefill_node: 处理该 prompt 的 prefill 连接标识（connection_key），不可为 None。

        Raises:
            ValueError: prefill_node 为 None 时抛出。
        """
        if prefill_node is None:
            raise ValueError("prefill_node must not be None")
        key = self._to_key(text)
        with self._lock:
            self._insert_at(self.root, key, 0, prefill_node)
            self._evict_if_needed()

    def _insert_at(self, node: _PromptCacheNode, key: str, depth: int, prefill_node: str) -> None:
        start = node
        time_mark = self._time_mark
        insert_time = time.monotonic()
        root = self.root
        leaf_lru = self._leaf_lru
        while depth < len(key):
            child = node.children.get(key[depth])
            if child is None:
                if node is not root and not node.children:
                    leaf_lru.pop(node.last_time_mark, None)
                suffix = key[depth:]
                child = _PromptCacheNode(parent=node, key=suffix)
                node.children[suffix[0]] = child
                self._node_count += len(suffix)
                time_mark += len(suffix)
                node = child
                break
            matched = self._common_prefix_length(key, depth, child.key)
            if matched < len(child.key):
                child = self._split_node(child, matched)
            depth += matched
            node = child

        # insert holds the tree lock throughout all time-mark and LRU updates.
        # Refresh leaf to root to preserve the original eviction order.
        leaf = node if node is not root and not node.children else None
        if leaf is not None:
            leaf_lru.pop(leaf.last_time_mark, None)
        while True:
            if node is not root:
                node.last_prefill_node = prefill_node
                node.last_insert_time = insert_time
                node.last_time_mark = time_mark + 1
                time_mark += len(node.key)
            else:
                time_mark += 1
                node.last_time_mark = time_mark
            if node is start:
                break
            parent = node.parent
            # Prefix endpoints cease to be metadata boundaries once the whole
            # path is refreshed. Coalesce unary edges, including growing chats.
            while parent is not start and len(parent.children) == 1:
                node.key = parent.key + node.key
                time_mark += len(parent.key)
                node.parent = parent.parent
                node.parent.children[node.key[0]] = node
                parent = node.parent
            node = node.parent
        if leaf is not None:
            leaf_lru[leaf.last_time_mark] = leaf
        self._time_mark = time_mark

    @staticmethod
    def _common_prefix_length(key: str, depth: int, edge: str) -> int:
        if key.startswith(edge, depth):
            return len(edge)
        remaining = len(key) - depth
        if remaining == 1 or key[depth + 1] != edge[1]:
            return 1
        if remaining < len(edge) and edge.startswith(key[depth:]):
            return remaining
        # The child lookup already matched the first character. Use C string
        # comparisons to locate a mismatch without walking characters in Python.
        low, high = 1, min(remaining, len(edge))
        while low < high:
            mid = (low + high + 1) // 2
            if key.startswith(edge[:mid], depth):
                low = mid
            else:
                high = mid - 1
        return low

    def _split_node(self, node: _PromptCacheNode, prefix_len: int) -> _PromptCacheNode:
        parent = node.parent
        prefix, suffix = node.key[:prefix_len], node.key[prefix_len:]
        middle = _PromptCacheNode(
            children={suffix[0]: node},
            parent=parent,
            key=prefix,
            last_prefill_node=node.last_prefill_node,
            last_insert_time=node.last_insert_time,
            last_time_mark=node.last_time_mark + len(suffix),
        )
        parent.children[prefix[0]] = middle
        node.parent = middle
        node.key = suffix
        return middle

    def _evict_if_needed(self) -> None:
        if self._node_count <= self.max_node_count:
            return
        need = self._node_count - self.max_node_count + self.evict_node_batch
        self._evict_chars(need)

    def prefix_match(self, text: str) -> PromptCacheMatchResult:
        """
        对 text 做前缀匹配。

        返回说明：
          - matched_char_count：匹配到的原始 prompt 字符数；
          - input_char_count：原始 prompt 字符数 len(text)；
          - prefill_node 为 None：匹配停在 root（无任何 key 字符命中）；
          - prefill_node 为非空 str：匹配停在非 root 节点，取该节点的 last_prefill_node。
          - last_insert_time：命中节点最近一次由 prompt 插入更新的单调时钟时间戳。
        """
        key = self._to_key(text)
        with self._lock:
            node, matched = self._match_at(self.root, key, 0)
            # The last matched sample always falls within the input text.
            matched_char_count = (matched - 1) * self.sample_stride + 1 if matched else 0
            return PromptCacheMatchResult(node.last_prefill_node, matched_char_count, len(text), node.last_insert_time)

    def _match_at(self, node: _PromptCacheNode, key: str, depth: int) -> Tuple[_PromptCacheNode, int]:
        key_len = len(key)
        while depth < key_len:
            child = node.children.get(key[depth])
            if child is None:
                break
            node = child
            edge = child.key
            if len(edge) == 1:
                depth += 1
                continue
            if key.startswith(edge, depth):
                depth += len(edge)
                continue
            # Keep shallow misses local without scanning long edges in Python.
            matched = 1
            while matched < 4 and depth + matched < key_len:
                if key[depth + matched] != edge[matched]:
                    return node, depth + matched
                matched += 1
            if depth + matched == key_len:
                return node, key_len
            if key.startswith(edge[:32], depth):
                matched = 32
            else:
                while matched < 32 and depth + matched < key_len:
                    if key[depth + matched] != edge[matched]:
                        return node, depth + matched
                    matched += 1
            if depth + matched == key_len:
                return node, key_len
            remaining = key_len - depth
            if remaining < len(edge) and edge.startswith(key[depth:]):
                return node, key_len
            low, high = matched, min(remaining, len(edge)) - 1
            while low < high:
                mid = (low + high + 1) // 2
                if key.startswith(edge[:mid], depth):
                    low = mid
                else:
                    high = mid - 1
            return node, depth + low
        return node, depth

    def evict_lru_nodes(self) -> int:
        """采样字符数超上限时，按 LRU 删除或裁剪叶边。

        删除数量为 ``_node_count - max_node_count + evict_node_batch``，
        使采样字符数降到 max_node_count 以下并留出缓冲。

        Returns:
            实际删除的采样字符数；未超上限时返回 0。
        """
        with self._lock:
            if self._node_count <= self.max_node_count:
                return 0
            need = self._node_count - self.max_node_count + self.evict_node_batch
            return self._evict_chars(need)

    def _evict_chars(self, need: int) -> int:
        removed = 0
        while removed < need and self._leaf_lru:
            _, node = self._leaf_lru.popitem(0)
            size = len(node.key)
            count = min(size, need - removed)
            self._node_count -= count
            removed += count
            if count < size:
                node.key = node.key[:-count]
                node.last_time_mark += count
                self._leaf_lru[node.last_time_mark] = node
            else:
                parent = node.parent
                parent.children.pop(node.key[0])
                if self._is_leaf(parent):
                    self._leaf_lru[parent.last_time_mark] = parent
        return removed
