"""Check packed embedding lookup and the shared HF AWQ lm_head loader."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def single_gpu_rank(runtime_start_args):
    if not torch.cuda.is_available():
        return
    from lightllm.utils import dist_utils

    dist_utils.set_current_device_id(0)
    dist_utils.set_current_rank_in_dp(0)
    dist_utils.set_dp_world_size(1)


@pytest.mark.parametrize("packed_zeros", [False, True])
def test_packed_embedding_matches_dense_rows_and_masks_image_ids(packed_zeros):
    from lightllm.models.sensenova_loop.layer_weights.awq_weight import LoopINT4EmbeddingWeight

    torch.manual_seed(42)
    vocab, dim, group = 16, 256, 128
    quantized = torch.randint(0, 16, (vocab, dim), dtype=torch.int64)
    shifts = torch.arange(8) * 4
    packed = (quantized.reshape(vocab, dim // 8, 8) << shifts).sum(-1).to(torch.int32)
    scales = (torch.rand(vocab, dim // group) + 0.01).half()
    zeros = torch.randint(0, 16, scales.shape, dtype=torch.uint8)
    source_zeros = zeros
    if packed_zeros:
        padded = torch.nn.functional.pad(zeros.long(), (0, 8 - zeros.shape[1]))
        source_zeros = ((padded << shifts).sum(-1)).to(torch.int32).unsqueeze(-1)
    embedding = LoopINT4EmbeddingWeight(dim, vocab, group, torch.bfloat16)
    embedding.load_hf_weights(
        {
            f"model.embed_tokens.{k}": v
            for k, v in {
                "qweight": packed,
                "scales": scales,
                "qzeros": source_zeros,
            }.items()
        }
    )
    assert embedding.verify_load()
    ids = torch.tensor([0, 7, 15, vocab, -1], device="cuda")
    actual = embedding(ids).cpu()
    reference = (
        (quantized.float() - zeros.repeat_interleave(group, -1).float()) * scales.repeat_interleave(group, -1).float()
    ).bfloat16()
    torch.testing.assert_close(actual[:3], reference[[0, 7, 15]], rtol=0, atol=0)
    assert torch.count_nonzero(actual[3:]) == 0
    assert not hasattr(embedding, "weight")

    from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb

    image_cache = torch.randn((1, 1, dim), device="cuda", dtype=torch.bfloat16)
    image_out = embedding(ids)
    multimodal_emb(
        image_out,
        ids,
        embedding.qweight,
        image_cache,
        torch.tensor([1], device="cuda"),
        torch.tensor([vocab], device="cuda"),
        torch.tensor([0], device="cuda"),
        0,
        vocab,
        1,
        skip_text_embeddings=True,
    )
    torch.testing.assert_close(image_out[:3].cpu(), actual[:3], rtol=0, atol=0)
    torch.testing.assert_close(image_out[3], image_cache[0, 0], rtol=0, atol=0)


@pytest.mark.parametrize("suffix,dtype", [("qweight", torch.int32), ("scales", torch.float16), ("qzeros", torch.uint8)])
def test_embedding_loader_rejects_broadcast_rows(suffix, dtype):
    from lightllm.models.sensenova_loop.layer_weights.awq_weight import LoopINT4EmbeddingWeight

    embedding = LoopINT4EmbeddingWeight(256, 16, 128, torch.bfloat16)
    columns = 32 if suffix == "qweight" else 2
    with pytest.raises(ValueError, match="tensor shape"):
        embedding.load_hf_weights({f"model.embed_tokens.{suffix}": torch.ones(1, columns, dtype=dtype)})


@pytest.mark.parametrize("rank", [0, 1])
def test_packed_embedding_shards_vocab_rows_across_tp(rank):
    from lightllm.utils import dist_utils
    from lightllm.models.sensenova_loop.layer_weights.awq_weight import LoopINT4EmbeddingWeight

    dist_utils.set_current_rank_in_dp(rank)
    dist_utils.set_dp_world_size(2)
    vocab, dim, group = 16, 1024, 128
    torch.manual_seed(100 + rank)
    quantized = torch.randint(0, 16, (vocab, dim), dtype=torch.int64)
    shifts = torch.arange(8, device="cuda") * 4
    packed = (quantized.cuda().reshape(vocab, dim // 8, 8) << shifts).sum(-1).to(torch.int32).cpu()
    scales = (torch.rand(vocab, dim // group) + 0.01).bfloat16()
    zeros = torch.randint(0, 16, scales.shape, dtype=torch.int64)
    zero_shifts = torch.arange(8, dtype=torch.int64) * 4
    packed_zeros = (zeros.reshape(vocab, -1, 8) << zero_shifts).sum(-1).to(torch.int32)

    embedding = LoopINT4EmbeddingWeight(dim, vocab, group, torch.bfloat16)
    embedding.load_hf_weights(
        {
            "model.embed_tokens.qweight": packed,
            "model.embed_tokens.scales": scales,
            "model.embed_tokens.qzeros": packed_zeros,
        }
    )
    assert embedding.verify_load()
    assert embedding.tp_vocab_start_id == rank * (vocab // 2)
    assert embedding.tp_vocab_end_id == (rank + 1) * (vocab // 2)

    ids = torch.arange(vocab + 2, device="cuda")
    actual = embedding(ids).cpu()
    reference = (
        (quantized.float() - zeros.repeat_interleave(group, -1).float()) * scales.repeat_interleave(group, -1).float()
    ).bfloat16()
    expected = torch.zeros_like(actual)
    expected[embedding.tp_vocab_start_id : embedding.tp_vocab_end_id] = reference[
        embedding.tp_vocab_start_id : embedding.tp_vocab_end_id
    ]
    torch.testing.assert_close(actual[:vocab], expected[:vocab], rtol=0, atol=0)
    assert torch.count_nonzero(actual[vocab:]) == 0


@pytest.mark.parametrize("rank,world_size", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("embedding_group", [None, 64])
def test_hf_awq_loader_normalizes_prefix_and_projects_local_logits(rank, world_size, embedding_group):
    from lightllm.models.sensenova_loop.layer_weights.pre_and_post_layer_weight import (
        SenseNovaLoopPreAndPostLayerWeight,
    )
    from lightllm.utils import dist_utils

    dist_utils.set_current_rank_in_dp(rank)
    dist_utils.set_dp_world_size(world_size)
    torch.manual_seed(301)
    vocab, dim, group = 256, 1024, 128
    config = {
        "n_layer": 1,
        "hidden_size": dim,
        "vocab_size": vocab,
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": group,
            "zero_point": True,
            "version": "gemm",
            "lm_head": True,
            "embedding": {} if embedding_group is None else {"group_size": embedding_group},
        },
    }
    layer = SenseNovaLoopPreAndPostLayerWeight(torch.bfloat16, config)
    embedding_group = group if embedding_group is None else embedding_group
    embedding_q = torch.randint(0, 16, (vocab, dim), dtype=torch.int64)
    embedding_zeros = torch.randint(0, 16, (vocab, dim // embedding_group), dtype=torch.int64)
    embedding_scales = (torch.rand(vocab, dim // embedding_group) * 0.02 + 0.001).half()
    shifts = torch.arange(8) * 4
    qweight = torch.randint(-(2 ** 31), 2 ** 31 - 1, (dim, vocab // 8), dtype=torch.int32)
    qzeros = torch.randint(-(2 ** 31), 2 ** 31 - 1, (dim // group, vocab // 8), dtype=torch.int32)
    scales = (torch.rand(dim // group, vocab) * 0.02 + 0.001).bfloat16()
    layer.load_hf_weights(
        {
            "model.language_model.embed_tokens.qweight": (embedding_q.reshape(vocab, -1, 8) << shifts)
            .sum(-1)
            .to(torch.int32),
            "model.language_model.embed_tokens.qzeros": (embedding_zeros.reshape(vocab, -1, 8) << shifts)
            .sum(-1)
            .to(torch.int32),
            "model.language_model.embed_tokens.scales": embedding_scales,
            "model.language_model.norm.weight": torch.ones(dim, dtype=torch.bfloat16),
            "lm_head.qweight": qweight,
            "lm_head.qzeros": qzeros,
            "lm_head.scales": scales,
        }
    )
    layer.verify_load()

    ids = torch.tensor([0, vocab // 2 - 1, vocab // 2, vocab - 1], device="cuda")
    dense_embedding = (
        (embedding_q.float() - embedding_zeros.repeat_interleave(embedding_group, -1).float())
        * embedding_scales.repeat_interleave(embedding_group, -1).float()
    ).bfloat16()
    expected = dense_embedding[ids.cpu()].clone()
    start, end = rank * (vocab // world_size), (rank + 1) * (vocab // world_size)
    expected[(ids.cpu() < start) | (ids.cpu() >= end)] = 0
    torch.testing.assert_close(layer.wte_weight_(ids).cpu(), expected, rtol=0, atol=0)

    # Decode canonical AWQ column packing independently of Marlin's repacking.
    inverse = [0, 4, 1, 5, 2, 6, 3, 7]
    unpack = lambda packed: ((packed.long().unsqueeze(-1) >> shifts) & 15)[..., inverse].flatten(1)
    dense = unpack(qweight).float() - unpack(qzeros).repeat_interleave(group, 0).float()
    dense = (dense * scales.repeat_interleave(group, 0).float()).bfloat16().cuda()
    hidden = torch.randn(dim, 3, device="cuda", dtype=torch.bfloat16)
    expected_logits = (hidden.t() @ dense[:, start:end]).t()
    head = layer.lm_head_weight_
    assert (head.tp_vocab_start_id, head.tp_vocab_end_id) == (start, end)
    torch.testing.assert_close(head(hidden), expected_logits, rtol=0.03, atol=0.03)
    out = torch.empty_like(expected_logits)
    assert head(hidden, out=out) is out
    torch.testing.assert_close(out, expected_logits, rtol=0.03, atol=0.03)
