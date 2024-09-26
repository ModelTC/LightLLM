# 利用构建的 lr(1) 自动机，构建对应的dpda 自动机
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Union, Dict, List, Tuple, Set, FrozenSet
from core import Item, T, NT, ItemLookAhead, ItemSet, Graph, Gen


@dataclass
class Edge:
    source_id: int
    transfer_input: Union[T, NT]
    dest_id: int
    is_loop_edge: bool = False


@dataclass
class LRGraph:
    origin_graph: Graph
    node_id_to_itemset: Dict[int, ItemSet] = None
    source_id_to_edge: Dict[int, Dict[Union[T, NT], Edge]] = None
    dest_id_to_edges: Dict[int, List[Edge]] = None
    s_id_e_id_to_edge: Dict[int, Dict[int, Edge]] = None
    # 保存能接收结束符的状态
    can_finished_node_id_set: Set[int] = None

    def __post_init__(self):
        self.node_id_to_itemset = {}
        self.source_id_to_edge = defaultdict(dict)
        self.dest_id_to_edges = defaultdict(list)
        self.can_finished_node_id_set = set()
        self.s_id_e_id_to_edge = defaultdict(dict)
        for node in self.origin_graph.graph_nodes:
            self.node_id_to_itemset[node.node_id] = node
            for input_t_or_nt, dest_node in node.edge_to_next.items():
                edge = Edge(node.node_id, input_t_or_nt, dest_node.node_id)
                self.source_id_to_edge[node.node_id][edge.transfer_input] = edge
                self.dest_id_to_edges[dest_node.node_id].append(edge)
                self.s_id_e_id_to_edge[edge.source_id][edge.dest_id] = edge
            if node.can_finished():
                self.can_finished_node_id_set.add(node.node_id)


# DPDA 的转换边信息
@dataclass
class DPDAEdge:
    lookah_input_t: Union[T, NT]
    input_t: Union[T, NT]
    pop: Tuple[int, ...]  # 与栈的方向相反
    push: Tuple[int, ...]  # 与栈的方向相同
    dest_node_id: int
    source_node_id: int

    def to_simple_str(self):
        ans = ""
        ans += f"t:{self.input_t.value}#pop:{self.pop}#push:{self.push}#s:{self.source_node_id}#e:{self.dest_node_id}"
        return ans


@dataclass
class DPDAEdgeMap:
    lookah_pop_to_edge: Dict[T, Dict[Tuple[int, ...], DPDAEdge]] = field(default_factory=lambda: defaultdict(dict))
    # pop_input_to_edge: Dict[Tuple[int, ...], Dict[T, DPDAEdge]] = field(default_factory=lambda: defaultdict(dict))
    to_dest_edges: Dict[int, List[DPDAEdge]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class DPDA:
    lr_graph: LRGraph
    one_step_node_id_to_dpda_edges: Dict[int, DPDAEdgeMap] = None
    none_jump_node_id_to_dpda_edges: Dict[int, DPDAEdgeMap] = None
    direct_jump_node_id_to_dpda_edges: Dict[int, DPDAEdgeMap] = None

    def add_one_step_dpadge(self, dpda_edge: DPDAEdge):
        self.one_step_node_id_to_dpda_edges[dpda_edge.source_node_id].lookah_pop_to_edge[dpda_edge.lookah_input_t][
            dpda_edge.pop
        ] = dpda_edge
        self.one_step_node_id_to_dpda_edges[dpda_edge.source_node_id].to_dest_edges[dpda_edge.dest_node_id].append(
            dpda_edge
        )

    def add_none_jump_dpadge(self, dpda_edge: DPDAEdge):
        self.none_jump_node_id_to_dpda_edges[dpda_edge.source_node_id].lookah_pop_to_edge[dpda_edge.lookah_input_t][
            dpda_edge.pop
        ] = dpda_edge
        self.none_jump_node_id_to_dpda_edges[dpda_edge.source_node_id].to_dest_edges[dpda_edge.dest_node_id].append(
            dpda_edge
        )
        return

    def add_direct_jump_dpadge(self, dpda_edge: DPDAEdge):
        assert dpda_edge.input_t == dpda_edge.lookah_input_t
        self.direct_jump_node_id_to_dpda_edges[dpda_edge.source_node_id].lookah_pop_to_edge[dpda_edge.lookah_input_t][
            dpda_edge.pop
        ] = dpda_edge
        self.direct_jump_node_id_to_dpda_edges[dpda_edge.source_node_id].to_dest_edges[dpda_edge.dest_node_id].append(
            dpda_edge
        )
        return

    def __repr__(self) -> str:
        ans = ""
        for source_id, edge_map in self.one_step_node_id_to_dpda_edges.items():
            ans += f"source id {source_id} ###################\n"
            for input_t, t_dict in edge_map.lookah_pop_to_edge.items():
                for pop, edge in t_dict.items():
                    ans += f"lookah {edge.lookah_input_t} input {input_t} pop {pop} \
                        push {edge.push}, dest id {edge.dest_node_id} \n"
            ans += "##########################################\n"
        return ans

    def __post_init__(self):
        self.one_step_node_id_to_dpda_edges = defaultdict(DPDAEdgeMap)
        self.none_jump_node_id_to_dpda_edges = defaultdict(DPDAEdgeMap)
        self.direct_jump_node_id_to_dpda_edges = defaultdict(DPDAEdgeMap)

        # 一些对后续处理有帮助的信息初始化
        for graph_node in self.lr_graph.origin_graph.graph_nodes:
            graph_node.init_t_to_item_la()
            graph_node.init_back_pair()
            # 为每个 graph node 标记上，进入他的转移输入是什么
            tmp_edges = self.lr_graph.dest_id_to_edges[graph_node.node_id]
            if len(tmp_edges) == 0:
                assert graph_node.node_id == 0
            else:
                graph_node.into_t_or_nt = tmp_edges[0].transfer_input

        # 找到 LR(1) 自动机有向图中可以形成环所有内部环。
        circles = []
        visit_stack = []
        visit_state = {}
        self.dfs_to_find_circle(0, visit_stack, visit_state, circles)

        circles = self.remove_same_circle(circles)
        # 处理找到的circle, 并做一些loop edge标记边, 和 添加一些 one step dpda_edge
        self.handle_circles(circles)

        # 添加一些普通边，对应的one step 跳转情况边
        for source_id, nt_or_t_to_edge in self.lr_graph.source_id_to_edge.items():
            for edge in nt_or_t_to_edge.values():
                # 找到不是回旋的情况，进行添加
                if len(self.one_step_node_id_to_dpda_edges[source_id].to_dest_edges[edge.dest_id]) == 0:
                    dpda_edge = DPDAEdge(
                        lookah_input_t=edge.transfer_input,
                        input_t=edge.transfer_input,
                        pop=(source_id,),
                        push=(source_id, edge.dest_id),
                        dest_node_id=edge.dest_id,
                        source_node_id=source_id,
                    )
                    self.add_one_step_dpadge(dpda_edge)

        # 递归处理生成direct jump 的跳转情况
        self.update_all_direct_jump()
        return

    def update_all_direct_jump(self):
        for graph_node in self.lr_graph.origin_graph.graph_nodes:
            if isinstance(graph_node.into_t_or_nt, T):
                for t in graph_node.t_to_item_la.keys():
                    self._get_direct_jump(graph_node.node_id, t)
        return

    def _get_direct_jump(self, node_id: int, input_t: T) -> List[DPDAEdge]:
        tmp_ans_dict = self.direct_jump_node_id_to_dpda_edges[node_id].lookah_pop_to_edge[input_t]
        if tmp_ans_dict:
            return tmp_ans_dict.values()
        # 递归求解
        visit_set_state = set()
        visit_set_state.add(node_id)
        cur_edge = DPDAEdge(
            lookah_input_t=None,
            input_t=None,
            pop=(node_id,),
            push=(node_id,),
            source_node_id=node_id,
            dest_node_id=node_id,
        )
        self._gen_direct_jump_rec(node_id, input_t, visit_set_state, cur_edge)
        return self.direct_jump_node_id_to_dpda_edges[node_id].lookah_pop_to_edge[input_t].values()

    def _gen_direct_jump_rec(self, cur_node_id: int, input_t: T, visit_set_state: Set[int], cur_edge: DPDAEdge):

        # check 一下 none jump 边是不是正常
        for none_jump_edge in self._get_none_jump(cur_node_id, input_t):
            assert none_jump_edge.input_t is None
            assert none_jump_edge.lookah_input_t == input_t

        # 处理 none jump 和已有跳转历史边的融合情况
        for none_jump_edge in self._get_none_jump(cur_node_id, input_t):
            if none_jump_edge.dest_node_id in visit_set_state:  # 形成环了, 不处理了
                continue

            merged_tmp_edge = self.merge_dpda_edge(cur_edge, none_jump_edge)
            if merged_tmp_edge is None:  # 不合法的拼接
                continue

            tmp_graph_node = self.lr_graph.node_id_to_itemset[merged_tmp_edge.dest_node_id]
            if input_t in tmp_graph_node.t_to_item_la:  # 说明还可以进行回退, 可以进行规约
                visit_set_state.add(none_jump_edge.dest_node_id)
                self._gen_direct_jump_rec(tmp_graph_node.node_id, input_t, visit_set_state, merged_tmp_edge)
                visit_set_state.remove(none_jump_edge.dest_node_id)
            else:
                # 可以移进，进行边生成
                for one_step_jump_edge in (
                    self.one_step_node_id_to_dpda_edges[merged_tmp_edge.dest_node_id]
                    .lookah_pop_to_edge[input_t]
                    .values()
                ):
                    ok_edge = self.merge_dpda_edge(merged_tmp_edge, one_step_jump_edge)
                    ok_edge.lookah_input_t = input_t
                    ok_edge.input_t = input_t
                    self.add_direct_jump_dpadge(ok_edge)
        return

    def _get_none_jump(self, node_id: int, input_t: T) -> List[DPDAEdge]:
        tmp_ans_dict = self.none_jump_node_id_to_dpda_edges[node_id].lookah_pop_to_edge[input_t]
        if tmp_ans_dict:
            return tmp_ans_dict.values()
        # 递归求解
        graph_node = self.lr_graph.node_id_to_itemset[node_id]
        assert input_t in graph_node.t_to_item_la
        pop_list = []
        visit_set_state = set()
        visit_set_state.add(node_id)
        self._gen_none_jump_rec(
            node_id, node_id, graph_node.t_to_item_la[input_t], input_t, -1, pop_list, visit_set_state
        )
        assert len(pop_list) == 0
        return self.none_jump_node_id_to_dpda_edges[node_id].lookah_pop_to_edge[input_t].values()

    def _gen_none_jump_rec(
        self,
        cur_node_id: int,
        iter_node_id: int,
        item_la: ItemLookAhead,
        input_t: Union[T, NT],
        back_index: int,
        pop_list: List[int],
        visit_set_state: Set[int],
    ):
        # back_index 从 -1 开始
        if -back_index > len(item_la.item.gen.gen_tuple):
            pop_list.append(iter_node_id)
            pop_tuple = tuple(pop_list)
            pop_list.pop()

            jump_to_node_id = self.lr_graph.source_id_to_edge[iter_node_id][item_la.item.gen.nt].dest_id
            if jump_to_node_id in visit_set_state:  # 回退成环
                return

            fake_edge = DPDAEdge(
                lookah_input_t=input_t,
                input_t=None,
                pop=pop_tuple,
                push=(iter_node_id,),
                source_node_id=cur_node_id,
                dest_node_id=iter_node_id,
            )

            for edge in self.one_step_node_id_to_dpda_edges[iter_node_id].to_dest_edges[jump_to_node_id]:
                assert isinstance(edge.input_t, NT)
                none_jump_edge = self.merge_dpda_edge(fake_edge, edge)
                none_jump_edge.lookah_input_t = input_t
                none_jump_edge.input_t = None
                print("none jump edge:", none_jump_edge)
                self.add_none_jump_dpadge(none_jump_edge)
            return

        pop_list.append(iter_node_id)
        for edge in self.lr_graph.dest_id_to_edges[iter_node_id]:
            assert edge.transfer_input == item_la.item.gen.gen_tuple[back_index]
            self._gen_none_jump_rec(
                cur_node_id, edge.source_id, item_la, input_t, back_index - 1, pop_list, visit_set_state
            )
        pop_list.pop()
        return

    def merge_dpda_edge(self, first_edge: DPDAEdge, second_edge: DPDAEdge):
        # assert first_edge.input_t is None, "None 代表空跳转"
        if len(second_edge.pop) >= len(first_edge.push):
            push_size = len(first_edge.push)
            for i in range(push_size):
                # print(first_edge, second_edge)
                if first_edge.push[i] != second_edge.pop[push_size - i - 1]:  # 可能会有多条回退路径，可能存在不匹配的回退路径，如果不匹配就返回None
                    return None
            left_count = len(second_edge.pop) - len(first_edge.push)
            if left_count != 0:
                return DPDAEdge(
                    lookah_input_t=None,
                    input_t=None,
                    pop=(first_edge.pop + second_edge.pop[-left_count:]),  # [-0:] 不是截取尾巴上的个数
                    push=second_edge.push,
                    dest_node_id=second_edge.dest_node_id,
                    source_node_id=first_edge.source_node_id,
                )
            else:
                return DPDAEdge(
                    lookah_input_t=None,
                    input_t=None,
                    pop=first_edge.pop,
                    push=second_edge.push,
                    dest_node_id=second_edge.dest_node_id,
                    source_node_id=first_edge.source_node_id,
                )
        else:
            print(first_edge.push, second_edge.pop)
            for i in range(len(second_edge.pop)):
                if first_edge.push[-(i + 1)] != second_edge.pop[i]:
                    return None
            left_count = len(first_edge.push) - len(second_edge.pop)
            return DPDAEdge(
                lookah_input_t=None,
                input_t=None,
                pop=first_edge.pop,
                push=first_edge.push[0:left_count] + second_edge.push,
                dest_node_id=second_edge.dest_node_id,
                source_node_id=first_edge.source_node_id,
            )

    def dfs_to_find_circle(
        self, cur_graph_id: int, visit_stack: List[int], visit_state_dict: Dict[int, int], ans_circles: List[List[int]]
    ):
        if cur_graph_id not in visit_state_dict:
            visit_stack.append(cur_graph_id)
            visit_state_dict[cur_graph_id] = len(visit_stack)  # 记录位置
            for edge in self.lr_graph.source_id_to_edge[cur_graph_id].values():
                self.dfs_to_find_circle(edge.dest_id, visit_stack, visit_state_dict, ans_circles)
            visit_stack.pop()
            del visit_state_dict[cur_graph_id]
            return
        else:
            origin_loc = visit_state_dict[cur_graph_id]
            tmp_ans = visit_stack[origin_loc - 1 :].copy()
            tmp_ans.append(cur_graph_id)
            ans_circles.append(tmp_ans)
            return

    def remove_same_circle(self, circles: List[List[int]]):
        tmp_set = set()
        for circle in circles:
            tmp_set.add(tuple(circle))
        return list(list(tuple_circle) for tuple_circle in tmp_set)

    def can_absorb_circle(self, circle: List[int]):
        # 添加 circle 环边
        # start_graph_node = self.lr_graph.node_id_to_itemset[circle[-2]]
        dest_graph_node = self.lr_graph.node_id_to_itemset[circle[-1]]
        input_t = dest_graph_node.into_t_or_nt  # graph node 的输入符
        for item_la in dest_graph_node.item_dict.values():
            if item_la.item.gen.gen_tuple[-1] == input_t:
                return True

        return False

    def handle_circles(self, circles: List[List[int]]):
        # 统计回旋的关键边是不是只有一种情况, 如果存在多种情况，则需要重新思考其形成过程
        state_counter = defaultdict(lambda: 0)
        for circle in circles:
            assert circle[0] == circle[-1]
            state_counter[(circle[-2], circle[-1])] += 1
            print(circle)

        for key, value in state_counter.items():
            assert value == 1, f"error {key}"

        # 将找到的路径，进行添加。
        for circle in circles:
            is_back_loop = self.judge_circle_is_back_loop(circle)
            print(f"cur handle circle: {circle} : is_back_loop: {is_back_loop}")
            if not is_back_loop:
                continue
            # 添加 circle 环边
            start_graph_node = self.lr_graph.node_id_to_itemset[circle[-2]]
            dest_graph_node = self.lr_graph.node_id_to_itemset[circle[-1]]
            input_t = dest_graph_node.into_t_or_nt  # graph node 的输入符
            # 将已经知道的DPDAEdge 进行添加
            dpda_edge = DPDAEdge(
                lookah_input_t=input_t,
                input_t=input_t,
                pop=tuple(list(reversed(circle[0:-1]))),
                push=(circle[-1],),
                dest_node_id=dest_graph_node.node_id,
                source_node_id=start_graph_node.node_id,
            )
            self.add_one_step_dpadge(dpda_edge)

            # 添加 其他边
            state_list = [circle[-1], circle[-2]]
            visit_state = {circle[-1], circle[-2]}
            ans_list = []
            self.find_back_from_circle_rec(circle, state_list, visit_state, -2, ans_list)

            for path in ans_list:
                # 将已经知道的DPDAEdge 进行添加
                dpda_edge = DPDAEdge(
                    lookah_input_t=input_t,
                    input_t=input_t,
                    pop=tuple(path[1:]),
                    push=tuple(list(reversed(path))),
                    dest_node_id=dest_graph_node.node_id,
                    source_node_id=start_graph_node.node_id,
                )
                self.add_one_step_dpadge(dpda_edge)
        return

    def judge_circle_is_back_loop(self, circle: List[int]):
        print(f"cur handle circle: {circle}")
        node_and_edge_list = []
        for i in range(len(circle) - 1):
            graph_node = self.lr_graph.node_id_to_itemset[circle[i]]
            node_and_edge_list.append(graph_node)
            dest_graph_node = self.lr_graph.node_id_to_itemset[circle[i + 1]]
            edge = self.lr_graph.s_id_e_id_to_edge[graph_node.node_id][dest_graph_node.node_id]
            node_and_edge_list.append(edge)

        for index in range(len(node_and_edge_list)):
            cur_obj = node_and_edge_list[index]
            if isinstance(cur_obj, ItemSet) and len(cur_obj.back_pair_list) != 0:
                for item_tuple in cur_obj.back_pair_list:
                    is_ok = self.judge_back_loop_rec(
                        cur_obj.node_id, item_tuple, node_and_edge_list, item_tuple, index, len(node_and_edge_list) // 2
                    )
                    if is_ok:
                        return True
        return False

    def judge_back_loop_rec(
        self,
        origin_node_id: int,
        origin_item_tuple: Tuple[Item, Item],
        node_and_edge_list: List[Union[ItemSet, Edge]],
        item_tuple: Tuple[Item, Item],
        cur_index: int,
        left_edge_count: int,
    ):
        if left_edge_count < 0:
            return False

        # cur_node: ItemSet = node_and_edge_list[cur_index]
        item1, item2 = item_tuple
        iter_index = cur_index

        # if len(item1.gen.gen_tuple[0:-1]) == 0:
        #     return False

        for nt_or_t in reversed(item1.gen.gen_tuple[0:-1]):
            iter_index = (iter_index - 1) % len(node_and_edge_list)
            left_edge_count -= 1
            if node_and_edge_list[iter_index].transfer_input != nt_or_t:
                return False
            iter_index = (iter_index - 1) % len(node_and_edge_list)

        # 到下一个节点
        next_index = (cur_index - 2 * (len(item1.gen.gen_tuple) - 1)) % len(node_and_edge_list)
        next_node: ItemSet = node_and_edge_list[next_index]

        for (n_item1, n_item2) in next_node.back_pair_list:
            if item1.gen != n_item2.gen:  # 连续关系匹配
                continue

            if (
                next_node.node_id == origin_node_id and left_edge_count == 0 and origin_item_tuple[1].gen == item1.gen
            ):  # 成环结束条件
                return True

            is_ok = self.judge_back_loop_rec(
                origin_node_id, origin_item_tuple, node_and_edge_list, (n_item1, n_item2), next_index, left_edge_count
            )
            if is_ok:
                return True

        return False

    def find_back_from_circle_rec(
        self, circle: List[int], state_list: List[int], visit_state: Set[int], index: int, ans_list: List[List[int]]
    ):
        if index == -len(circle):
            return

        cur_node_id = circle[index]
        assert cur_node_id == state_list[-1]
        edges = self.lr_graph.dest_id_to_edges[cur_node_id]
        for edge in edges:
            if edge.source_id in visit_state:  # 不能有回退环
                print("find back loop")
                continue

            if edge.source_id != circle[index - 1]:
                # 直接生成一种情况的答案
                state_list.append(edge.source_id)
                ans_list.append(state_list.copy())
                state_list.pop()
            else:
                state_list.append(edge.source_id)
                visit_state.add(edge.source_id)
                self.find_back_from_circle_rec(circle, state_list, visit_state, index - 1, ans_list)
                visit_state.remove(edge.source_id)
                state_list.pop()
        return

    def to_mermaid(self):
        ans = "```mermaid\n"
        ans += "flowchart LR\n"
        for graph_node in self.lr_graph.origin_graph.graph_nodes:
            graph_info = graph_node.to_simple_str()
            ans += f'{graph_node.node_id}["' + graph_info + '"]\n'

        ans += "\n"
        for edge_maps in self.one_step_node_id_to_dpda_edges.values():
            for input_t_or_nt, pop_to_dpda_edge_dict in edge_maps.lookah_pop_to_edge.items():
                if isinstance(input_t_or_nt, T):
                    for pop, edge in pop_to_dpda_edge_dict.items():
                        assert edge.input_t == edge.lookah_input_t
                        edge_str = edge.to_simple_str().replace("(", "")
                        edge_str = edge_str.replace(")", "")
                        edge_str = edge_str.replace(" ", "")
                        ans += f"{edge.source_node_id} --> {edge_str} ---> {edge.dest_node_id}\n"

        for edge_maps in self.direct_jump_node_id_to_dpda_edges.values():
            for input_t_or_nt, pop_to_dpda_edge_dict in edge_maps.lookah_pop_to_edge.items():
                if isinstance(input_t_or_nt, T):
                    for pop, edge in pop_to_dpda_edge_dict.items():
                        assert edge.input_t == edge.lookah_input_t
                        edge_str = edge.to_simple_str().replace("(", "")
                        edge_str = edge_str.replace(")", "")
                        edge_str = edge_str.replace(" ", "")
                        ans += f"{edge.source_node_id} --> {edge_str} ---> {edge.dest_node_id}\n"

        ans += "```"
        return ans

    def remove_no_input_node_to_edges(self):
        """
        删除一些graph node，其没有入口边，只有出口边
        """
        visited_nodes = set()
        self.dfs_to_find_reached(0, visited_nodes)
        for node_id in range(len(self.lr_graph.origin_graph.graph_nodes)):
            if node_id not in visited_nodes:
                self.one_step_node_id_to_dpda_edges[node_id] = DPDAEdgeMap()  # 搞一个空的替换
        return

    def dfs_to_find_reached(self, start_id: int, visited_nodes: Set[int]):
        if start_id not in visited_nodes:
            visited_nodes.add(start_id)
            for input_t, pop_to_edges in self.one_step_node_id_to_dpda_edges[start_id].lookah_pop_to_edge.items():
                for edge in pop_to_edges.values():
                    self.dfs_to_find_reached(edge.dest_node_id, visited_nodes)
            for input_t, pop_to_edges in self.direct_jump_node_id_to_dpda_edges[start_id].lookah_pop_to_edge.items():
                for edge in pop_to_edges.values():
                    self.dfs_to_find_reached(edge.dest_node_id, visited_nodes)
        else:
            return

    def accept(self, input_str: str):
        stack = [0]
        current_node_id = 0
        for t in input_str:
            t = T(t)
            input_pop_edge1 = self.one_step_node_id_to_dpda_edges[current_node_id].lookah_pop_to_edge
            input_pop_edge2 = self.direct_jump_node_id_to_dpda_edges[current_node_id].lookah_pop_to_edge

            if t not in input_pop_edge1 and t not in input_pop_edge2:
                raise Exception("not accept")

            pop_edge = []
            if t in input_pop_edge1:
                pop_edge.extend([(pop, edge) for pop, edge in input_pop_edge1[t].items()])
            elif t in input_pop_edge2:
                pop_edge.extend([(pop, edge) for pop, edge in input_pop_edge2[t].items()])
            else:
                assert False, "can not to here"

            find = False
            find_count = 0
            for pop, edge in pop_edge:
                if self._stack_match(stack, pop):
                    del stack[-len(pop) :]
                    stack.extend(edge.push)
                    current_node_id = edge.dest_node_id
                    find = True
                    find_count += 1
                    break
            assert find_count == 1, f"find_count {find_count}"
            if not find:
                # print(stack)
                raise Exception("not accept")

        if current_node_id not in self.lr_graph.can_finished_node_id_set:
            raise Exception("not accept")

        return

    def _stack_match(self, stack: List[int], pop: List[int]):
        if len(stack) < len(pop):
            return False

        for i, p in enumerate(pop, 1):
            if p != stack[-i]:
                return False

        return True


if __name__ == "__main__":
    from core import compute_first, compute_graph

    grammar = [
        (NT("S'"), [NT("S")]),
        (NT("S"), [NT("A"), NT("B")]),
        (NT("A"), [T("a"), NT("A")]),
        (NT("A"), [T("a")]),
        (NT("B"), [T("b"), NT("B")]),
        (NT("B"), [T("b")]),
    ]
    # grammar = [
    #     (NT("S'"), [NT("W")]),
    #     (NT("S'"), [NT("F")]),
    #     (NT("W"), [T("l"), NT("F")]),
    #     (NT("F"), [NT("W")]),
    #     (NT("F"), [T("f")])
    # ]

    grammar = [
        (NT("S'"), [NT("S")]),
        (
            NT("S"),
            [
                NT("A"),
            ],
        ),
        (NT("A"), [T("a"), NT("C")]),
        (NT("C"), [T("c"), NT("D")]),
        (NT("D"), [T("d"), NT("A"), T("d")]),
        (NT("D"), [T("d")]),
    ]

    grammar = [
        (NT("S'"), [NT("S")]),
        (
            NT("S"),
            [
                NT("A"),
            ],
        ),
        (NT("A"), [T("a"), NT("C")]),
        (NT("C"), [T("c"), NT("D")]),
        (NT("D"), [T("d"), NT("E"), NT("A"), T("d")]),
        (NT("E"), [T("e")]),
        (NT("D"), [T("d")]),
    ]

    grammar = [
        (NT("S'"), [NT("S")]),
        (NT("S"), [NT("A")]),
        (NT("A"), [T("a"), T("a"), NT("A")]),
        (NT("A"), [T("a")]),
        (NT("A"), [T("c"), T("a"), NT("A")]),
    ]

    grammar = [
        (NT("S'"), [NT("S")]),
        (NT("S"), [NT("A")]),
        (NT("A"), [T("a"), NT("C"), NT("A")]),
        (NT("A"), [T("a")]),
        (NT("C"), [T("c")]),
    ]

    ans = compute_first(grammar)
    print(ans)

    graph = compute_graph(grammar=grammar, start_symbol="S'")
    graph.visit_print()
    # graph.check_lr1()
    graph_str = graph.to_mermaid()
    with open("mermaid.md", mode="+w") as file:
        file.write(graph_str)
    graph.check_lr1()
    lr_graph = LRGraph(graph)
    dpda = DPDA(lr_graph=lr_graph)
    print(dpda)

    dpda.remove_no_input_node_to_edges()

    with open("mermaid1.md", mode="+w") as file:
        file.write(dpda.to_mermaid())

    for in_str in ["a", "aa", "aaa", "aaab", "aaaaabbbb"]:
        try:
            dpda.accept(in_str)
            print(f"{in_str} accepted")
        except:
            print(f"{in_str} not accepted")
