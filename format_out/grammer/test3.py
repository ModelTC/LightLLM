from core import compute_first, compute_graph
from dpda import NT, LRGraph, Graph, DPDA
from dpda import T


def create_grammer():
    grammar = [
        (NT("S'"), [NT("S")]),
        (NT("S"), [NT("A"), NT("B")]),
        (NT("A"), [T("a"), NT("A")]),
        (NT("A"), [T("a")]),
        (NT("B"), [T("b"), NT("B")]),
        (NT("B"), [T("b")]),
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
for in_str in ["ab", "aab", "aaaab", "ab", "abb", "aaabb"]:
    try:
        dpda.accept(in_str)
        print(f"{in_str} accepted")
        assert True
    except:
        print(f"{in_str} not accepted")
        assert False

print("####################")
# not accept test
for in_str in ["ba", "b"]:
    try:
        dpda.accept(in_str)
        print(f"{in_str} accepted")
        assert False
    except:
        print(f"{in_str} not accepted")
        assert True


print(dpda.direct_jump_node_id_to_dpda_edges)
