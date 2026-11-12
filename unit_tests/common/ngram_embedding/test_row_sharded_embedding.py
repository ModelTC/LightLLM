from types import SimpleNamespace
import importlib

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.row_sharded_embedding_weight import RowShardedEmbeddingWeight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


@pytest.fixture(autouse=True)
def runtime_configuration(monkeypatch):
    base = importlib.import_module("lightllm.common.basemodel.layer_weights.meta_weights.base_weight")
    platform = importlib.import_module("lightllm.common.basemodel.layer_weights.meta_weights.platform_op")
    monkeypatch.setattr(base, "get_current_device_id", lambda: torch.cuda.current_device())
    monkeypatch.setattr(
        platform,
        "get_env_start_args",
        lambda: SimpleNamespace(
            hardware_platform="cuda",
            enable_torch_fallback=False,
            enable_triton_fallback=False,
        ),
    )


def make_weight(rank=0, world=1):
    return RowShardedEmbeddingWeight(5, 24, f"{PREFIX}.weight", torch.bfloat16, 4, rank, world)


def shards(table):
    return [{f"{PREFIX}.shard_{i}.weight": part} for i, part in enumerate(table.chunk(4))]


@pytest.mark.parametrize("world", [1, 2, 3, 8])
def test_streamed_shards_and_vocab_parallel_lookup(world):
    table = torch.arange(120, dtype=torch.bfloat16).view(24, 5)
    checkpoint = shards(table)
    ids = torch.tensor([0, 5, 6, 7, 8, 11, 12, 15, 17, 18, 23], device="cuda", dtype=torch.int64)
    outputs = []
    for rank in range(world):
        weight = make_weight(rank, world)
        assert not weight.verify_load()
        required, loaded = set(), set()
        for i in range(4):
            if max(i * 6, weight.tp_vocab_start_id) < min((i + 1) * 6, weight.tp_vocab_end_id):
                required.add(i)
        for index in [2, 0, 3, 1]:
            weight.load_hf_weights(checkpoint[index])
            loaded.add(index)
            assert weight.verify_load() == (loaded >= required)
        torch.testing.assert_close(
            weight.weight.cpu(), table[weight.tp_vocab_start_id : weight.tp_vocab_end_id], atol=0, rtol=0
        )
        outputs.append(weight(ids))

    torch.testing.assert_close(
        torch.stack(outputs).sum(0), torch.nn.functional.embedding(ids, table.cuda()), atol=0, rtol=0
    )


def test_missing_shards_bad_shapes_and_other_prefixes():
    weight = make_weight()
    table = torch.arange(120, dtype=torch.bfloat16).view(24, 5)
    checkpoint = shards(table)
    for index in [0, 1, 3]:
        weight.load_hf_weights(checkpoint[index])
    assert not weight.verify_load()
    weight.load_hf_weights({"other.shard_2.weight": table[:6]})
    assert not weight.verify_load()
    with pytest.raises(ValueError, match="shard 2"):
        weight.load_hf_weights({f"{PREFIX}.shard_2.weight": table[:5]})
    assert not weight.verify_load()
    weight.load_hf_weights(checkpoint[2])
    assert weight.verify_load()

    weight.load_hf_weights({"other.weight": table})
    assert weight.verify_load()


def test_full_checkpoint_compatibility_and_reject_mixed_formats():
    table = torch.arange(120, dtype=torch.bfloat16).view(24, 5)
    weight = make_weight(1, 3)
    weight.load_hf_weights({f"{PREFIX}.weight": table})
    assert weight.verify_load()
    weight.load_hf_weights({"other.weight": table})
    assert weight.verify_load()
    torch.testing.assert_close(weight.weight.cpu(), table[8:16], atol=0, rtol=0)
    with pytest.raises(ValueError, match="mix"):
        weight.load_hf_weights(shards(table)[0])
    partial = make_weight()
    partial.load_hf_weights(shards(table)[1])
    with pytest.raises(ValueError, match="mix"):
        partial.load_hf_weights({f"{PREFIX}.weight": table})
    with pytest.raises(ValueError, match="dimensions"):
        make_weight().load_hf_weights({f"{PREFIX}.weight": table[:5]})


def test_row_shard_configuration_and_graph_replay():
    with pytest.raises(ValueError, match="divisible"):
        RowShardedEmbeddingWeight(5, 23, f"{PREFIX}.weight", torch.bfloat16, 4, 0, 1)
    table = torch.arange(120, dtype=torch.bfloat16).view(24, 5)
    weight = make_weight()
    for part in shards(table):
        weight.load_hf_weights(part)
    ids = torch.tensor([0, 6, 12, 23], device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            weight(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = weight(ids)
    for values in [[23, 12, 6, 0], [1, 7, 13, 19]]:
        ids.copy_(torch.tensor(values, device="cuda"))
        graph.replay()
        torch.testing.assert_close(output.cpu(), table[values], atol=0, rtol=0)


def test_packed_hash_lookup_and_ple_composition():
    from lightllm.common.ngram_embedding import NGramHasher
    from lightllm.common.ngram_embedding.ple import PLELayer

    torch.manual_seed(137)
    sizes = torch.tensor([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53], device="cuda", dtype=torch.int64)
    offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
    hasher = NGramHasher(
        torch.tensor([23703573157769, 20109073645365, 8052911324071], device="cuda"), sizes, offsets, 248044
    )
    tokens = torch.tensor([1, 2, 248044, 3, 4, 5, 6, 7], device="cuda", dtype=torch.int64)
    starts = torch.tensor([0, 3, 4, 4, 8], device="cuda", dtype=torch.int32)
    indices, _ = hasher.hash(tokens, starts, hasher.empty_history(4))

    table = torch.randn(512, 4, dtype=torch.bfloat16)
    partition_outputs = []
    for rank in range(3):
        weight = RowShardedEmbeddingWeight(4, 512, f"{PREFIX}.weight", torch.bfloat16, 128, rank, 3)
        for index, shard in reversed(list(enumerate(table.chunk(128)))):
            weight.load_hf_weights({f"{PREFIX}.shard_{index}.weight": shard})
        assert weight.verify_load()
        partition_outputs.append(weight(indices.flatten()))
    embeddings = torch.stack(partition_outputs).sum(0).view(tokens.numel(), 64)
    expected_embeddings = torch.nn.functional.embedding(indices, table.cuda()).flatten(-2)
    torch.testing.assert_close(embeddings, expected_embeddings, atol=0, rtol=0)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    layer = PLELayer(
        rand(64, 64) / 8, rand(32, 64) / 8, rand(64) * 0.1, rand(64) * 0.1, rand(64) * 0.1, rand(64, 4) * 0.1, 2, 3
    )
    history = layer.empty_history(4)
    residual = rand(tokens.numel(), 64)
    output, conv_inputs = layer.forward(residual, embeddings, starts, history)
    expected_output, expected_inputs = layer.forward(residual, expected_embeddings, starts, history)
    torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
    torch.testing.assert_close(conv_inputs, expected_inputs, atol=0, rtol=0)
