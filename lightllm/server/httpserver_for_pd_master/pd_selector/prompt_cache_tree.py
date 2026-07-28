from __future__ import annotations

import sys
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Dict, Optional, Tuple

from sortedcontainers import SortedDict


@dataclass(slots=True)
class PromptCacheMatchResult:
    """prefix_match 的返回结果。"""

    # 匹配终点上的 prefill 节点（client_ip_port）；仅当匹配停在 root 时为 None。
    prefill_node: Optional[str]
    # 匹配到的原始 prompt 字符数：从 text[0] 到最后一个命中采样点（含）的连续前缀长度。
    matched_char_count: int
    # 输入 prompt 的原始字符数 len(text)。
    input_char_count: int


@dataclass(slots=True)
class _PromptCacheNode:
    """单字符边的 trie 节点：parent --edge_char--> self。"""

    children: Dict[str, "_PromptCacheNode"] = field(default_factory=dict)
    parent: Optional["_PromptCacheNode"] = None
    edge_char: Optional[str] = None
    last_prefill_node: Optional[str] = None
    last_time_mark: int = 0


class PromptCacheTree:
    """
    用于 cache-aware 选点的 prompt 前缀缓存树。

    prompt 先按 sample_stride 抽稀成 key，再递归按单字符建树/匹配。
    recursion_limit 在初始化时通过 sys.setrecursionlimit 调大 Python 调用栈深度。
    整棵树节点数有上限；超限时按 LRU 从叶节点批量删除。
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
            max_node_count: 树中允许的最大节点数（不含 root）；超限时触发 LRU 驱逐。
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
        self._time_mark_lock = Lock()

    def _to_key(self, text: str) -> str:
        return text[:: self.sample_stride]

    def _is_leaf(self, node: _PromptCacheNode) -> bool:
        return node is not self.root and not node.children

    def insert(self, text: str, prefill_node: str) -> None:
        """将 text 的前缀路径写入树，并关联到 prefill_node。

        路径上的节点会更新 last_prefill_node 与 LRU 时间戳；写入后若超
        max_node_count 会立即按 LRU 驱逐一批叶节点。

        Args:
            text: 原始 prompt 文本。
            prefill_node: 处理该 prompt 的 prefill 节点标识（client_ip_port），不可为 None。

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
        try:
            if depth >= len(key):
                return

            ch = key[depth]
            child = node.children.get(ch)
            if child is None:
                child = _PromptCacheNode(parent=node, edge_char=ch)
                child.last_time_mark = self._gen_time_mark()
                node.children[ch] = child
                self._node_count += 1

            self._insert_at(child, key, depth + 1, prefill_node)
        finally:
            if node is not self.root:
                node.last_prefill_node = prefill_node
            if node.last_time_mark in self._leaf_lru:
                self._leaf_lru.pop(node.last_time_mark, None)
            node.last_time_mark = self._gen_time_mark()
            if self._is_leaf(node):
                self._leaf_lru[node.last_time_mark] = node

    def _gen_time_mark(self) -> int:
        with self._time_mark_lock:
            time_mark = getattr(self, "_time_mark", -1) + 1
            self._time_mark = time_mark
            return time_mark

    def _evict_if_needed(self) -> None:
        if self._node_count <= self.max_node_count:
            return
        need = self._node_count - self.max_node_count + self.evict_node_batch
        removed = 0
        while removed < need and self._leaf_lru:
            _, node = self._leaf_lru.peekitem(0)
            self._remove_leaf_node(node)
            removed += 1

    def prefix_match(self, text: str) -> PromptCacheMatchResult:
        """
        对 text 做前缀匹配。

        返回说明：
          - matched_char_count：匹配到的原始 prompt 字符数；
          - input_char_count：原始 prompt 字符数 len(text)；
          - prefill_node 为 None：匹配停在 root（无任何 key 字符命中）；
          - prefill_node 为非空 str：匹配停在非 root 节点，取该节点的 last_prefill_node。
        """
        key = self._to_key(text)
        with self._lock:
            node, matched = self._match_at(self.root, key, 0)
            if node is self.root:
                prefill_node = None
            else:
                prefill_node = node.last_prefill_node
            if matched <= 0:
                matched_char_count = 0
            else:
                matched_char_count = min((matched - 1) * self.sample_stride + 1, len(text))
            return PromptCacheMatchResult(
                prefill_node=prefill_node,
                matched_char_count=matched_char_count,
                input_char_count=len(text),
            )

    def _match_at(self, node: _PromptCacheNode, key: str, depth: int) -> Tuple[_PromptCacheNode, int]:
        if depth >= len(key):
            return node, depth

        child = node.children.get(key[depth])
        if child is None:
            return node, depth

        return self._match_at(child, key, depth + 1)

    def evict_lru_nodes(self) -> int:
        """节点数超上限时，按 LRU 从叶节点批量删除。

        删除数量为 ``_node_count - max_node_count + evict_node_batch``，
        使节点数降到 max_node_count 以下并留出缓冲。

        Returns:
            实际删除的叶节点数量；未超上限时返回 0。
        """
        with self._lock:
            if self._node_count <= self.max_node_count:
                return 0
            need = self._node_count - self.max_node_count + self.evict_node_batch
            removed = 0
            while removed < need and self._leaf_lru:
                _, node = self._leaf_lru.peekitem(0)
                self._remove_leaf_node(node)
                removed += 1
            return removed

    def _remove_leaf_node(self, node: _PromptCacheNode) -> None:
        assert self._is_leaf(node)
        parent = node.parent
        assert parent is not None and node.edge_char is not None

        self._leaf_lru.pop(node.last_time_mark, None)
        parent.children.pop(node.edge_char, None)
        self._node_count -= 1
        if self._is_leaf(parent):
            self._leaf_lru[parent.last_time_mark] = parent
