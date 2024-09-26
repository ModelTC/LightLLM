# 文法表达形式限制
# 1. 必须是 LR(1) 文法
# 1. 起始表示符一定是 S‘
# 2. 不支持 "ε" 表达式

from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Union, Dict, List, Tuple, Set, FrozenSet

# 项中的点标识，主要用于格式化输出
@dataclass
class Dot:
    def __repr__(self) -> str:
        return "@"


# 终结符
@dataclass
class T:
    value: str

    def is_terminal(self):
        return True

    def is_finished(self):
        # 当 value 是空的时候，代表结束符号
        return self.value == ""

    def __hash__(self) -> int:
        return self.value.__hash__()

    def __repr__(self) -> str:
        return f"t({self.value})"


# 非终结符
@dataclass
class NT:
    name: str

    def is_terminal(self):
        return False

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __repr__(self) -> str:
        return f"nt({self.name})"


@dataclass
class Gen:
    nt: NT
    gen_tuple: Tuple[Union[NT, T], ...]
    gen_id: int

    def __hash__(self) -> int:
        return self.gen_id.__hash__()

    def __repr__(self) -> str:
        return f"Gen({self.gen_id}, {self.nt} = {[e for e in self.gen_tuple]})"

    def __eq__(self, __value: object) -> bool:
        return self.gen_id == __value.gen_id


@dataclass
class Item:
    gen: Gen
    loc: int
    item_id: int = None

    def __post_init__(self):
        self.item_id = self.gen.gen_id * 100000000 + self.loc  # 很难出现文法长度超过这个界限的，可以认为可以保证唯一性
        return

    def __hash__(self) -> int:
        return self.item_id.__hash__()

    def __eq__(self, __value: object) -> bool:
        return self.item_id == __value.item_id

    def get_next_input(self) -> Union[T, NT, None]:
        if self.is_finished_loc():
            return None
        else:
            return self.gen.gen_tuple[self.loc]

    def is_finished_loc(self):
        return self.loc == len(self.gen.gen_tuple)

    def get_next_t_or_nt_mark(self):
        return self.gen.gen_tuple[self.loc]

    def __repr__(self) -> str:
        dot_list = [e for e in self.gen.gen_tuple]
        dot_list.insert(self.loc, Dot())
        return f"Item({self.gen.nt} = {dot_list})"


@dataclass
class ItemLookAhead:
    item: Item
    lookahead_set: Union[Set[T], FrozenSet[T]]
    hash_id: int = None

    def mark_hash_id(self):
        # 生成 hash id 用于 dict 中的存储
        self.lookahead_set = frozenset(self.lookahead_set)
        self.hash_id = hash((self.item.item_id, self.lookahead_set))
        return

    def __hash__(self) -> int:
        if self.hash_id is None:
            raise Exception("not ready")
        return self.hash_id

    def can_accept_input(self, t: T):
        if self.item.is_finished_loc():
            return t in self.lookahead_set
        else:
            return self.item.gen.gen_tuple[self.item.loc] == t

    def can_finished(self):
        if self.item.is_finished_loc():
            if T("") in self.lookahead_set:
                return True
        return False

    def get_next_first(self, first_map):
        loc = self.item.loc + 1
        if loc == len(self.item.gen.gen_tuple):
            return self.lookahead_set
        else:
            next_mark = self.item.gen.gen_tuple[loc]
            if next_mark.is_terminal():
                return {
                    next_mark,
                }
            else:
                return first_map[next_mark]

    def get_next_gen_item_la(self, input_nt_or_t: Union[T, NT]):
        if self.item.is_finished_loc():
            return None
        nt_or_t = self.item.gen.gen_tuple[self.item.loc]
        if nt_or_t != input_nt_or_t:
            return None
        new_item = Item(gen=self.item.gen, loc=self.item.loc + 1)
        ans = ItemLookAhead(item=new_item, lookahead_set=set(self.lookahead_set))
        ans.mark_hash_id()
        return ans

    def __repr__(self) -> str:
        dot_list = [e for e in self.item.gen.gen_tuple]
        dot_list.insert(self.item.loc, Dot())
        return f"ItemLookAhead({self.item.gen.nt} = {dot_list} # la = {self.lookahead_set})"

    def to_simple_str(self) -> str:
        dot_list = [e for e in self.item.gen.gen_tuple]
        dot_list.insert(self.item.loc, Dot())
        return f"({self.item.gen.nt} = {dot_list} # la = {tuple(self.lookahead_set)})"


@dataclass
class ItemSet:
    item_dict: Dict[Item, ItemLookAhead]
    edge_to_next: Dict[Union[T, NT], "ItemSet"] = field(default_factory=dict)
    hash_id: int = None
    node_id: int = None
    into_t_or_nt: Union[T, NT] = None
    back_pair_list: List[Tuple[Item, Item]] = None

    # 如果 ItemSet 是一个可以规约的项目，则添加一个 T to ItemLookAhead 的存储信息，方便快速的找到某个lookahead 的T信息
    # 对应的生成式
    t_to_item_la: Dict[T, ItemLookAhead] = None

    def __post_init__(self):
        self._mark_hash_id()
        return

    def _mark_hash_id(self):
        assert self.hash_id is None
        self.hash_id = hash(frozenset(self.item_dict.values()))
        return

    def __hash__(self) -> int:
        if self.hash_id is None:
            raise Exception("not ready")
        return self.hash_id

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ItemSet):
            if self.item_dict == __value.item_dict:
                return True
        return False

    def get_next_input_set(self):
        ans: Set[Union[T, NT]] = set()
        for item in self.item_dict.keys():
            t_ans = item.get_next_input()
            if t_ans is not None:
                ans.add(t_ans)
        return ans

    def get_next_graphs(self, first_map, grammar_dict) -> List[object]:
        ans = []
        for input_nt_or_t in self.get_next_input_set():
            new_graph_node = self.get_next_graph(input_nt_or_t, first_map, grammar_dict)
            ans.append((input_nt_or_t, new_graph_node))
        return ans

    def get_next_graph(self, input_nt_or_t: Union[T, NT], first_map, grammar_dict):
        new_item_dict = {}
        for item_la in self.item_dict.values():
            new_gen_item_la = item_la.get_next_gen_item_la(input_nt_or_t=input_nt_or_t)
            if new_gen_item_la is not None:
                new_item_dict[new_gen_item_la.item] = new_gen_item_la
        gen_closure(new_item_dict, first_map, grammar_dict)
        return ItemSet(item_dict=new_item_dict)

    def can_finished(self):
        for item_la in self.item_dict.values():
            if item_la.can_finished():
                return True
        return False

    def init_t_to_item_la(self):
        self.t_to_item_la = {}
        for item_la in self.item_dict.values():
            if item_la.item.is_finished_loc():
                for t in item_la.lookahead_set:
                    if not t.is_finished():
                        self.t_to_item_la[t] = item_la
        return

    def init_back_pair(self):
        self.back_pair_list = []  # 这个结构用于后续 dpda 检测回退成环的情况。
        for item1 in self.item_dict.keys():
            for item2 in self.item_dict.keys():
                if item1 != item2:
                    if isinstance(item1.gen.gen_tuple[-1], NT) and item1.loc == len(item1.gen.gen_tuple) - 1:
                        if item2.loc == 0 and item2.gen.nt == item1.gen.gen_tuple[-1]:
                            self.back_pair_list.append((item1, item2))

    def __repr__(self) -> str:
        ans = f"graph node: id {self.node_id} start #####################\n"
        ans += "\n".join([str(e) for e in self.item_dict.values()])
        ans += "\n"
        ans += f"graph node: id {self.node_id} end   #####################"
        ans += "\n\n"
        return ans

    def to_simple_str(self) -> str:
        ans = f"id: {self.node_id}\n"
        for item_la in self.item_dict.values():
            ans += f"{item_la.to_simple_str()}\n"
        return ans


@dataclass
class Graph:
    graph_nodes: List[ItemSet]
    node_id_to_graph_node: Dict[int, ItemSet] = None

    def __post_init__(self):
        self.node_id_to_graph_node = {}
        for node in self.graph_nodes:
            self.node_id_to_graph_node[node.node_id] = node
        return

    def check_lr1(self):
        for node in self.graph_nodes:
            items = list(node.item_dict.values())
            for i in range(len(items)):
                if items[i].item.is_finished_loc():
                    for cur_t in items[i].lookahead_set:
                        for j in range(i + 1, len(items)):
                            if items[j].can_accept_input(cur_t):
                                print("check failed node:", node)
                                raise Exception("lr1 check fialed")

        return

    def visit_print(self):
        for node in self.graph_nodes:
            print(node)
            for edge_nt_or_t, node in node.edge_to_next.items():
                print("edge:", edge_nt_or_t, "to", node.node_id)
            print("#" * 10)

    def to_mermaid(self):
        ans = "```mermaid\n"
        ans += "flowchart LR\n"
        for graph_node in self.graph_nodes:
            graph_info = graph_node.to_simple_str()
            ans += f'{graph_node.node_id}["' + graph_info + '"]\n'

        ans += "\n"
        for graph_node in self.graph_nodes:
            for nt_or_t, next_graph_node in graph_node.edge_to_next.items():
                if isinstance(nt_or_t, T):
                    ans += f"{graph_node.node_id} --> \
                    {graph_node.node_id}_{nt_or_t.value}_{next_graph_node.node_id} ---> {next_graph_node.node_id}\n"
                else:
                    ans += f"{graph_node.node_id} --> \
                    {graph_node.node_id}_{nt_or_t.name}_{next_graph_node.node_id} ---> {next_graph_node.node_id}\n"
        ans += "```"
        return ans


def grammar_to_dict(grammar: List[Tuple[NT, List[Union[NT, T]]]]) -> Dict[NT, List[Gen]]:
    grammar_dict: Dict[NT, List[List[Union[NT, T]]]] = defaultdict(list)
    for index, (nt, gen_list) in enumerate(grammar):
        grammar_dict[nt].append(Gen(gen_id=index, nt=nt, gen_tuple=tuple(gen_list)))

    return grammar_dict


def compute_first(grammar: List[Tuple[NT, List[Union[NT, T]]]]) -> Dict[NT, Set[T]]:
    first_map: Dict[NT, Set[T]] = defaultdict(set)
    while True:
        has_update = False
        for nt, gen_list in grammar:
            # 因为文法约束了不支持 "ε" 表达式， 所以只需要首个生成对象进行处理就可以了。
            # first 必然出现在第一个位置
            for obj in gen_list[0:1]:
                obj: Union[T, NT] = obj
                if obj.is_terminal():
                    if obj not in first_map[nt]:
                        first_map[nt].add(obj)
                        has_update = True
                else:
                    if not first_map[obj].issubset(first_map[nt]):
                        first_map[nt].update(first_map[obj])
                        has_update = True
        if has_update is False:
            break
    return first_map


def gen_closure(item_dict: Dict[Item, ItemLookAhead], first_map, grammar_dict):
    handle_queue = deque()
    for t_item in item_dict.keys():
        handle_queue.append(t_item)

    while len(handle_queue) != 0:
        origin_item: ItemLookAhead = handle_queue.popleft()
        origin_item_la = item_dict[origin_item]
        if origin_item_la.item.is_finished_loc():
            continue

        next_mark = origin_item_la.item.get_next_t_or_nt_mark()
        new_t_set = origin_item_la.get_next_first(first_map)
        if not next_mark.is_terminal():
            gen_list = grammar_dict[next_mark]
            for gen in gen_list:
                new_item = Item(gen, 0)
                if new_item in item_dict:
                    # 只更新 lookahead 信息
                    new_item_la = item_dict[new_item]
                    if not new_t_set.issubset(new_item_la.lookahead_set):
                        new_item_la.lookahead_set.update(new_t_set)
                        handle_queue.append(new_item)
                else:
                    new_item_la = ItemLookAhead(item=new_item, lookahead_set=set())
                    new_item_la.lookahead_set.update(new_t_set)
                    item_dict[new_item] = new_item_la
                    handle_queue.append(new_item)

    for item_la in item_dict.values():
        if item_la.hash_id is None:
            item_la.mark_hash_id()
    return


def compute_graph(grammar: List[Tuple[NT, List[Union[NT, T]]]], start_symbol):
    grammar_dict = grammar_to_dict(grammar)
    start_nt = NT(start_symbol)
    first_map = compute_first(grammar)

    item_dict = {}
    gen_list = grammar_dict[start_nt]
    for gen in gen_list:
        item = Item(gen, 0)
        item_la = ItemLookAhead(item, {T(value="")})
        item_dict[item] = item_la

    gen_closure(item_dict=item_dict, first_map=first_map, grammar_dict=grammar_dict)
    first_graph_node = ItemSet(item_dict=item_dict)

    handle_queue = deque()
    handle_queue.append(first_graph_node)
    graph_dict = {first_graph_node: first_graph_node}
    first_graph_node.node_id = 0
    while len(handle_queue) != 0:
        cur_graph_node: ItemSet = handle_queue.popleft()
        new_graph_nodes: List[Tuple[Union[NT, T], ItemSet]] = cur_graph_node.get_next_graphs(first_map, grammar_dict)
        for nt_or_t, new_graph in new_graph_nodes:
            if new_graph not in graph_dict:
                graph_dict[new_graph] = new_graph
                new_graph.node_id = len(graph_dict) - 1
                cur_graph_node.edge_to_next[nt_or_t] = new_graph
                handle_queue.append(new_graph)
            else:
                new_graph = graph_dict[new_graph]  # 替换成内部对象
                if nt_or_t not in cur_graph_node.edge_to_next:
                    cur_graph_node.edge_to_next[nt_or_t] = new_graph
                    handle_queue.append(new_graph)

    graph_nodes = [node for node in graph_dict.values()]

    return Graph(graph_nodes=graph_nodes)


def dfs_visit(graph: Graph):
    for node in graph.graph_nodes:
        print(node)
        for edge_nt_or_t, node in node.edge_to_next.items():
            print("edge:", edge_nt_or_t, "to", node.node_id)


if __name__ == "__main__":
    grammar = [
        (NT("S'"), [NT("S")]),
        (NT("S"), [NT("A"), NT("B")]),
        (NT("A"), [T("a"), NT("A")]),
        (NT("A"), [T("a")]),
        (NT("B"), [T("b"), NT("B")]),
        (NT("B"), [T("b")]),
    ]
    ans = compute_first(grammar)
    print(ans)

    graph = compute_graph(grammar=grammar, start_symbol="S'")
    graph.visit_print()
    graph.check_lr1()
