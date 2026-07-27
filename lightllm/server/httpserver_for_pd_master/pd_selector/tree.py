from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from threading import RLock
from typing import Dict, List, Optional, Tuple


_EPOCH_COUNTER = count()

# 每隔 sample_stride 个字符取 1 个作为前缀树 key，压缩树深度与内存。
DEFAULT_SAMPLE_STRIDE = 16


def _get_epoch() -> int:
    return next(_EPOCH_COUNTER)


def _shared_prefix_count(a: str, b: str) -> int:
    matched = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        matched += 1
    return matched


def sample_prompt_key(text: str, sample_stride: int) -> str:
    """从 prompt 中按固定步长抽字符，作为前缀匹配 key。"""
    if sample_stride <= 1 or not text:
        return text
    return text[::sample_stride]


@dataclass(slots=True)
class PrefixMatchResult:
    tenant: str
    matched_char_count: int
    input_char_count: int


@dataclass(slots=True)
class _Node:
    text: str
    children: Dict[str, "_Node"] = field(default_factory=dict)
    tenant_last_access_time: Dict[str, int] = field(default_factory=dict)
    parent: Optional["_Node"] = None
    last_tenant: Optional[str] = None


class Tree:
    """
    Cache-aware 前缀树（Python）。

    对原始 prompt 按 sample_stride 抽稀后再建树/匹配，key 长度约为
    len(prompt) / sample_stride，从而降低匹配开销与树节点内存。
    matched_char_count / input_char_count 均基于抽稀后的 key 长度计算，
    比值仍可近似反映原始前缀重合比例。
    """

    def __init__(self, sample_stride: int = DEFAULT_SAMPLE_STRIDE) -> None:
        if sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")
        self.sample_stride = sample_stride
        self.root = _Node(text="")
        self.tenant_char_count: Dict[str, int] = {}
        self._lock = RLock()

    def _to_key(self, text: str) -> str:
        return sample_prompt_key(text, self.sample_stride)

    def insert(self, text: str, tenant: str) -> None:
        key = self._to_key(text)
        with self._lock:
            self.root.tenant_last_access_time.setdefault(tenant, 0)
            self.tenant_char_count.setdefault(tenant, 0)

            remaining = key
            prev = self.root

            while remaining:
                first_char = remaining[0]
                child = prev.children.get(first_char)

                if child is None:
                    remaining_char_count = len(remaining)
                    epoch = _get_epoch()
                    new_node = _Node(
                        text=remaining,
                        tenant_last_access_time={tenant: epoch},
                        parent=prev,
                        last_tenant=tenant,
                    )
                    self.tenant_char_count[tenant] = self.tenant_char_count.get(tenant, 0) + remaining_char_count
                    prev.children[first_char] = new_node
                    return

                shared_count = _shared_prefix_count(remaining, child.text)
                child_len = len(child.text)

                if shared_count < child_len:
                    matched_text = child.text[:shared_count]
                    contracted_text = child.text[shared_count:]
                    matched_text_count = shared_count

                    new_node = _Node(
                        text=matched_text,
                        tenant_last_access_time=dict(child.tenant_last_access_time),
                        parent=prev,
                        last_tenant=child.last_tenant,
                    )
                    new_node.children[contracted_text[0]] = child

                    child.text = contracted_text
                    child.parent = new_node
                    prev.children[first_char] = new_node

                    if tenant not in new_node.tenant_last_access_time:
                        self.tenant_char_count[tenant] = self.tenant_char_count.get(tenant, 0) + matched_text_count
                        new_node.tenant_last_access_time[tenant] = 0

                    prev = new_node
                    remaining = remaining[shared_count:]
                else:
                    if tenant not in child.tenant_last_access_time:
                        self.tenant_char_count[tenant] = self.tenant_char_count.get(tenant, 0) + child_len
                        child.tenant_last_access_time[tenant] = 0
                    prev = child
                    remaining = remaining[shared_count:]

            epoch = _get_epoch()
            prev.tenant_last_access_time[tenant] = epoch
            prev.last_tenant = tenant

    def prefix_match_with_counts(self, text: str) -> PrefixMatchResult:
        key = self._to_key(text)
        with self._lock:
            remaining = key
            matched_chars = 0
            prev = self.root

            while remaining:
                first_char = remaining[0]
                child = prev.children.get(first_char)
                if child is None:
                    break

                shared_count = _shared_prefix_count(remaining, child.text)
                child_len = len(child.text)

                if shared_count == child_len:
                    matched_chars += shared_count
                    remaining = remaining[shared_count:]
                    prev = child
                else:
                    matched_chars += shared_count
                    prev = child
                    break

            curr = prev

            if curr.last_tenant and curr.last_tenant in curr.tenant_last_access_time:
                tenant = curr.last_tenant
            else:
                tenant = next(iter(curr.tenant_last_access_time), "empty")
                curr.last_tenant = tenant

            if tenant != "empty":
                curr.tenant_last_access_time[tenant] = _get_epoch()

            return PrefixMatchResult(
                tenant=tenant,
                matched_char_count=matched_chars,
                input_char_count=len(key),
            )

    def prefix_match(self, text: str) -> Tuple[str, str]:
        key = self._to_key(text)
        result = self.prefix_match_with_counts(text)
        return key[: result.matched_char_count], result.tenant

    def prefix_match_tenant(self, text: str, tenant: str) -> str:
        key = self._to_key(text)
        with self._lock:
            remaining = key
            matched_chars = 0
            prev = self.root

            while remaining:
                first_char = remaining[0]
                child = prev.children.get(first_char)
                if child is None:
                    break
                if tenant not in child.tenant_last_access_time:
                    break

                shared_count = _shared_prefix_count(remaining, child.text)
                child_len = len(child.text)

                if shared_count == child_len:
                    matched_chars += shared_count
                    remaining = remaining[shared_count:]
                    prev = child
                else:
                    matched_chars += shared_count
                    prev = child
                    break

            if tenant in prev.tenant_last_access_time:
                prev.tenant_last_access_time[tenant] = _get_epoch()
                prev.last_tenant = tenant

            return key[:matched_chars]

    @staticmethod
    def _leaf_of(node: _Node) -> List[str]:
        candidates: Dict[str, bool] = {tenant: True for tenant in node.tenant_last_access_time}
        for child in node.children.values():
            for tenant in child.tenant_last_access_time:
                candidates[tenant] = False
        return [tenant for tenant, is_leaf in candidates.items() if is_leaf]

    def evict_tenant_by_size(self, max_size: int) -> None:
        with self._lock:
            stack = [self.root]
            pq: List[Tuple[int, str, _Node]] = []

            while stack:
                curr = stack.pop()
                stack.extend(curr.children.values())
                for tenant in self._leaf_of(curr):
                    ts = curr.tenant_last_access_time.get(tenant)
                    if ts is not None:
                        heappush(pq, (ts, tenant, curr))

            while pq:
                _, tenant, node = heappop(pq)
                used_size = self.tenant_char_count.get(tenant, 0)
                if used_size <= max_size:
                    continue

                if tenant not in node.tenant_last_access_time:
                    continue
                if any(tenant in child.tenant_last_access_time for child in node.children.values()):
                    continue

                node_len = len(node.text)
                self.tenant_char_count[tenant] = max(0, self.tenant_char_count.get(tenant, 0) - node_len)

                node.tenant_last_access_time.pop(tenant, None)
                if node.last_tenant == tenant:
                    node.last_tenant = next(iter(node.tenant_last_access_time), None)

                parent = node.parent
                if not node.children and not node.tenant_last_access_time and parent is not None:
                    if node.text:
                        parent.children.pop(node.text[0], None)

                if parent is not None and tenant in parent.tenant_last_access_time:
                    has_child_with_tenant = any(
                        tenant in child.tenant_last_access_time for child in parent.children.values()
                    )
                    if not has_child_with_tenant:
                        ts = parent.tenant_last_access_time.get(tenant)
                        if ts is not None:
                            heappush(pq, (ts, tenant, parent))

                if self.tenant_char_count.get(tenant, 0) == 0:
                    self.tenant_char_count.pop(tenant, None)

    def remove_tenant(self, tenant: str) -> None:
        with self._lock:
            stack = [self.root]
            queue: List[_Node] = []

            while stack:
                curr = stack.pop()
                stack.extend(curr.children.values())

                if tenant in curr.tenant_last_access_time:
                    has_child_with_tenant = any(
                        tenant in child.tenant_last_access_time for child in curr.children.values()
                    )
                    if not has_child_with_tenant:
                        queue.append(curr)

            while queue:
                curr = queue.pop(0)
                curr.tenant_last_access_time.pop(tenant, None)
                if curr.last_tenant == tenant:
                    curr.last_tenant = next(iter(curr.tenant_last_access_time), None)

                parent = curr.parent
                if not curr.children and not curr.tenant_last_access_time and parent is not None:
                    if curr.text:
                        parent.children.pop(curr.text[0], None)

                if parent is not None and tenant in parent.tenant_last_access_time:
                    has_child_with_tenant = any(
                        tenant in child.tenant_last_access_time for child in parent.children.values()
                    )
                    if not has_child_with_tenant:
                        queue.append(parent)

            self.tenant_char_count.pop(tenant, None)

    def get_tenant_char_count(self) -> Dict[str, int]:
        with self._lock:
            return dict(self.tenant_char_count)

    def get_used_size_per_tenant(self) -> Dict[str, int]:
        with self._lock:
            used_size_per_tenant: Dict[str, int] = {}
            stack = [self.root]
            while stack:
                curr = stack.pop()
                text_count = len(curr.text)
                for tenant in curr.tenant_last_access_time:
                    used_size_per_tenant[tenant] = used_size_per_tenant.get(tenant, 0) + text_count
                stack.extend(curr.children.values())
            return used_size_per_tenant
