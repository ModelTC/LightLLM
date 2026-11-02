import pytest
import torch

from lightllm.common.ngram_embedding import NGramHasher

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
EOS = 248044


def make_hasher(ngram_size=3, heads_per_ngram=8, overflow=False):
    heads = (ngram_size - 1) * heads_per_ngram
    multipliers = [39719048695937, 29866592791859, 5701618528319, 123456789123, 9876543212345][:ngram_size]
    if overflow:
        multipliers = [(1 << 62) + i * 141 + 1 for i in range(ngram_size)]
    sizes = torch.arange(heads, device="cuda", dtype=torch.int64) * 2 + 20000003
    offsets = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int64), sizes.cumsum(0)[:-1]])
    return NGramHasher(torch.tensor(multipliers, device="cuda", dtype=torch.int64), sizes, offsets, EOS)


def pack(sequences, dtype=torch.int64):
    lengths = torch.tensor([0] + [len(seq) for seq in sequences], device="cuda", dtype=torch.int32)
    tokens = torch.tensor([token for seq in sequences for token in seq], device="cuda", dtype=dtype)
    return tokens, lengths.cumsum(0).to(torch.int32)


def reference(hasher, sequences, history):
    rows, next_histories = [], []
    context = hasher.context_len
    multipliers = hasher.multipliers.cpu()
    sizes = hasher.head_vocab_sizes.tolist()
    offsets = hasher.head_offsets.tolist()
    per_order = hasher.heads // context
    for request, sequence in enumerate(sequences):
        full = history[request].tolist() + sequence
        for index in range(context, len(full)):
            shifted = [full[index]]
            for shift in range(1, context + 1):
                preceding = full[index - shift : index]
                shifted.append(EOS if EOS in preceding else full[index - shift])
            row = []
            for order in range(2, context + 2):
                mixed = torch.tensor(shifted[0], dtype=torch.int64) * multipliers[0]
                for pos in range(1, order):
                    mixed ^= torch.tensor(shifted[pos], dtype=torch.int64) * multipliers[pos]
                begin = (order - 2) * per_order
                for head in range(begin, begin + per_order):
                    row.append((int(mixed) % sizes[head]) + offsets[head])
            rows.append(row)
        next_histories.append(full[-context:])
    indices = torch.tensor(rows, device="cuda", dtype=torch.int64).reshape(-1, hasher.heads)
    next_history = torch.tensor(next_histories, device="cuda", dtype=torch.int64).reshape(-1, context)
    return indices, next_history


@pytest.mark.parametrize("ngram_size,heads_per_ngram", [(2, 1), (3, 8), (4, 3), (5, 2)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("overflow", [False, True])
def test_hash_matches_eos_segment_reference(ngram_size, heads_per_ngram, dtype, overflow):
    hasher = make_hasher(ngram_size, heads_per_ngram, overflow)
    sequences = [[], [7], [EOS], [11, EOS, 13, 14, EOS, EOS, 15, 16, 17], [], list(range(130))]
    history = hasher.empty_history(len(sequences))
    history[0, -1] = 777
    history[3, -1] = 999
    if hasher.context_len > 1:
        history[3, -2] = EOS
    before = history.clone()
    tokens, starts = pack(sequences, dtype)
    actual = hasher.hash(tokens, starts, history)
    expected = reference(hasher, sequences, history)
    for result, ref in zip(actual, expected):
        torch.testing.assert_close(result, ref, atol=0, rtol=0)
    torch.testing.assert_close(history, before, atol=0, rtol=0)
    assert torch.all(actual[0] >= hasher.head_offsets)
    assert torch.all(actual[0] < hasher.head_offsets + hasher.head_vocab_sizes)


def test_chunked_prefill_and_decode_match_full_sequence():
    hasher = make_hasher()
    sequences = [[1, 2, EOS, 3, 4, 5, EOS, 6], [11, EOS, 12, 13, EOS, 14], [EOS, 31]]
    original = hasher.empty_history(len(sequences))
    expected, full_history = hasher.hash(*pack(sequences), original)
    history = original
    per_request = [[] for _ in sequences]
    consumed = [0] * len(sequences)
    for counts in [(1, 0, 1), (2, 3, 1), (1, 1, 0), (4, 2, 0)]:
        chunks = [sequence[start : start + count] for sequence, start, count in zip(sequences, consumed, counts)]
        tokens, starts = pack(chunks)
        result, history = hasher.hash(tokens, starts, history)
        for request, count in enumerate(counts):
            per_request[request].append(result[int(starts[request]) : int(starts[request + 1])])
            consumed[request] += count
    combined = torch.cat([torch.cat(chunks) for chunks in per_request])
    torch.testing.assert_close(combined, expected, atol=0, rtol=0)
    torch.testing.assert_close(history, full_history, atol=0, rtol=0)


def test_speculative_rejection_and_cache_snapshot():
    hasher = make_hasher()
    sequences = [[1, 2, 3, EOS], [11, EOS, 12], [21, 22], []]
    tokens, starts = pack(sequences)
    initial = hasher.empty_history(len(sequences))
    initial[:, -1] = 99
    snapshot = initial.clone()

    hasher.hash(tokens, starts, initial)
    accepted = torch.tensor([1, 2, 0, 0], device="cuda", dtype=torch.int32)
    committed = hasher.history_after(tokens, starts, initial, accepted)
    expected = reference(hasher, [seq[:n] for seq, n in zip(sequences, [1, 2, 0, 0])], snapshot)[1]
    torch.testing.assert_close(committed, expected, atol=0, rtol=0)
    torch.testing.assert_close(initial, snapshot, atol=0, rtol=0)

    followup = [[7], [8], [9], [10]]
    result = hasher.hash(*pack(followup), committed)
    ref = reference(hasher, followup, expected)
    for actual, expected in zip(result, ref):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    order = torch.tensor([2, 0, 2], device="cuda")
    histories = snapshot.index_select(0, order)
    sequences = [[31], [32, EOS, 33], [34]]
    for actual, expected in zip(hasher.hash(*pack(sequences), histories), reference(hasher, sequences, histories)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_checkpoint_and_cuda_graph_replay():
    source = make_hasher()
    prefix = "model.language_model.layers.1.ple.ple_embedding"
    buffers = {
        f"{prefix}.layer_multipliers": source.multipliers,
        f"{prefix}.ngram_heads_vocab_sizes": source.head_vocab_sizes,
        f"{prefix}.ngram_heads_offsets": source.head_offsets,
    }
    hasher = NGramHasher.from_checkpoint(buffers, prefix, EOS)
    assert hasher.multipliers is source.multipliers
    history = hasher.empty_history(3)
    tokens, starts = pack([[1, 2, 3], [11, 12], [21, 22]])
    accepted = torch.tensor([3, 1, 2], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            hasher.hash(tokens, starts, history)
            hasher.history_after(tokens, starts, history, accepted)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices, next_history = hasher.hash(tokens, starts, history)
        committed = hasher.history_after(tokens, starts, history, accepted)
    for sequences, counts in [
        ([[EOS, 2], [], [11, 12, 13, 14, 15]], [1, 0, 4]),
        ([[], [31, EOS, 32], [41, 42, 43, 44]], [0, 2, 0]),
    ]:
        updated_tokens, updated_starts = pack(sequences)
        tokens.copy_(updated_tokens)
        starts.copy_(updated_starts)
        accepted.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        history[:, -1] = 151
        graph.replay()
        ref_indices, ref_history = reference(hasher, sequences, history)
        torch.testing.assert_close(indices, ref_indices, atol=0, rtol=0)
        torch.testing.assert_close(next_history, ref_history, atol=0, rtol=0)
        ref_committed = reference(hasher, [seq[:n] for seq, n in zip(sequences, counts)], history)[1]
        torch.testing.assert_close(committed, ref_committed, atol=0, rtol=0)


def test_empty_batches_and_bad_layouts():
    hasher = make_hasher()
    for sequences in [[], [[], [], []]]:
        history = hasher.empty_history(len(sequences))
        indices, next_history = hasher.hash(*pack(sequences), history)
        assert indices.shape == (0, hasher.heads)
        torch.testing.assert_close(next_history, history, atol=0, rtol=0)
    tokens, starts = pack([[1, 2, 3]])
    history = hasher.empty_history(1)
    with pytest.raises(ValueError, match="packed"):
        hasher.hash(tokens.view(1, -1), starts, history)
    with pytest.raises(ValueError, match="Starts"):
        hasher.hash(tokens, starts.long(), history)
    with pytest.raises(ValueError, match="History"):
        hasher.hash(tokens, starts, history.int())
    with pytest.raises(ValueError, match="contiguous"):
        hasher.hash(tokens, starts, torch.empty(1, 4, device="cuda", dtype=torch.int64)[:, ::2])
    with pytest.raises(ValueError, match="Accepted"):
        hasher.history_after(tokens, starts, history, torch.ones(1, device="cuda", dtype=torch.int64))


@pytest.mark.parametrize("invalid", ["multipliers", "empty_heads", "dtype", "zero_size", "overlap"])
def test_invalid_checkpoint_buffers(invalid):
    from dataclasses import replace

    hasher = make_hasher()
    with pytest.raises(ValueError):
        if invalid == "multipliers":
            replace(hasher, multipliers=hasher.multipliers[:1])
        elif invalid == "empty_heads":
            replace(hasher, head_vocab_sizes=hasher.head_vocab_sizes[:0])
        elif invalid == "dtype":
            replace(hasher, multipliers=hasher.multipliers.int())
        elif invalid == "zero_size":
            sizes = hasher.head_vocab_sizes.clone()
            sizes[0] = 0
            replace(hasher, head_vocab_sizes=sizes)
        else:
            offsets = hasher.head_offsets.clone()
            offsets[1] = 0
            replace(hasher, head_offsets=offsets)
