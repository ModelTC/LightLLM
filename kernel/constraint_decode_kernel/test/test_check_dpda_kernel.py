import time
import torch

from lightllm_constraint_decode_kernel import check_dpda

from lightllm.server.router.model_infer.mode_backend.continues_batch.format_out.grammar.core import (
    compute_first,
    compute_graph,
)
from lightllm.server.router.model_infer.mode_backend.continues_batch.format_out.grammar.dpda import (
    NT,
    LRGraph,
    Graph,
    DPDA,
)
from lightllm.server.router.model_infer.mode_backend.continues_batch.format_out.grammar.dpda import T
from lightllm.server.router.model_infer.mode_backend.continues_batch.format_out.grammar.example_grammar import (
    json_grammar,
)

from transformers import AutoTokenizer


def test_kernel():

    grammar = [
        (NT("S'"), [NT("S")]),
        (NT("S"), [NT("A")]),
        (NT("A"), [T("a"), NT("C"), NT("A")]),
        (NT("A"), [T("a")]),
        (NT("C"), [T("c")]),
    ]

    grammar = json_grammar
    # ans = compute_first(grammar)
    graph = compute_graph(grammar=grammar, start_symbol="JSON")
    graph.check_lr1()
    lr_graph = LRGraph(graph)
    dpda = DPDA(lr_graph=lr_graph)

    shift_table, edge_num_table, push_table, pop_table, dest_table, symbol_to_id = dpda.dump_to_tensor()
    print(shift_table.shape)
    print(shift_table, edge_num_table, push_table, pop_table, dest_table, symbol_to_id, sep="\n")

    tokenizer = AutoTokenizer.from_pretrained("/data/chenjunyi/models/qwen2-7b-chat")

    #
    # print(check_str)
    check_str = list(tokenizer.get_vocab().keys())
    # print(check_str)

    other_token_id = symbol_to_id["a"]

    char_sequences = []
    sequence_len = []
    for s in check_str:
        char_sequences.append([symbol_to_id[c] if c in symbol_to_id else other_token_id for c in s])
        sequence_len.append(len(s))
    sequence_len = torch.tensor(sequence_len, dtype=torch.int32, device="cuda")
    input_sequences = torch.empty((len(char_sequences), torch.max(sequence_len)), dtype=torch.int32, device="cuda")
    for i, s in enumerate(char_sequences):
        input_sequences[i, : len(s)] = torch.tensor(s, dtype=torch.int32, device="cuda")
    print(input_sequences)
    current_state = torch.tensor([0] * len(check_str), dtype=torch.int32, device="cuda")
    current_stack = torch.tensor([0], dtype=torch.int32, device="cuda")

    torch.cuda.synchronize()
    start = time.time()
    check_dpda(
        input_sequences,
        sequence_len,
        shift_table,
        edge_num_table,
        push_table,
        pop_table,
        dest_table,
        current_stack,
        current_state,
        50,
    )
    torch.cuda.synchronize()
    print(f"CUDA Elapsed Time: {time.time() - start}s")

    torch.cuda.synchronize()
    start_time = time.time()
    for i, str in enumerate(check_str):
        dpda.try_shift(str, [0], 0)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Python Elapsed time: {end_time - start_time:.4f}s")

    for i in range(current_state.shape[0]):
        if current_state[i] != -1:
            print(f"{check_str[i]} is accepted.")


if __name__ == "__main__":
    test_kernel()
