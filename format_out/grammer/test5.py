from core import compute_first, compute_graph
from dpda import NT, LRGraph, Graph, DPDA
from dpda import T


def create_grammer():
    # add = TT("a")
    # sub = TT("-")
    mul = T("c")
    # div = TT("/")
    lparen = T("l")
    rparen = T("r")
    num = [T("1"), T("2"), T("3")]

    E = NT("E")
    M = NT("M")
    F = NT("F")
    # NUM = NT("NUM")

    # 1cl1cl1c1rr
    grammar = [
        (NT("S'"), [E]),
        # (E, [E, add, T]),
        # (E, [E, sub, T]),
        (E, [M]),
        (M, [M, mul, F]),
        # (T, [T, div, F]),
        (M, [F]),
        (F, [lparen, E, rparen]),
        (F, [num[0]]),
        # (F, [NUM]),
        # (NUM, [num[0]]),
        # (NUM, [num[1]]),
        # (NUM, [num[2]]),
    ]
    return grammar


grammar = create_grammer()

ans = compute_first(grammar)
print(ans)

graph = compute_graph(grammar=grammar, start_symbol="S'")

graph_str = graph.to_mermaid()
with open("mermaid.md", mode="+w") as file:
    file.write(graph_str)
    file.flush()

graph.visit_print()
graph.check_lr1()

lr_graph = LRGraph(graph)
dpda = DPDA(lr_graph=lr_graph)
print(dpda)

# dpda.remove_no_input_node_to_edges()

with open("mermaid1.md", mode="+w") as file:
    file.write(dpda.to_mermaid())

# accept test
for in_str in [
    "1c1",
    "1cl1c1r",
    "1cl1cl1c1rr",
]:
    try:
        dpda.accept(in_str)
        print(f"{in_str} accepted")
        assert True
    except:
        print(f"{in_str} not accepted")
        assert False

print("####################")
# not accept test
for in_str in ["aa", "aaabb"]:
    try:
        dpda.accept(in_str)
        print(f"{in_str} accepted")
        assert False
    except:
        print(f"{in_str} not accepted")
        assert True


print(dpda.direct_jump_node_id_to_dpda_edges)
