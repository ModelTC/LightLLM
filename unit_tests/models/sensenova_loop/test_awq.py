"""Check the export layout, independent input scaling, and graph replay on CUDA."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def single_gpu_rank():
    if not torch.cuda.is_available():
        return
    from lightllm.utils import dist_utils

    dist_utils.set_current_device_id(0)
    dist_utils.set_current_rank_in_dp(0)
    dist_utils.set_dp_world_size(1)


def test_packed_embedding_matches_dense_rows_and_masks_image_ids():
    from lightllm.models.sensenova_loop.layer_weights.awq_weight import LoopINT4EmbeddingWeight

    torch.manual_seed(42)
    vocab, dim, group = 16, 256, 128
    quantized = torch.randint(0, 16, (vocab, dim), dtype=torch.int64)
    shifts = torch.arange(8) * 4
    packed = (quantized.reshape(vocab, dim // 8, 8) << shifts).sum(-1).to(torch.int32)
    scales = (torch.rand(vocab, dim // group) + 0.01).bfloat16()
    zeros = torch.randint(0, 16, scales.shape, dtype=torch.uint8)
    embedding = LoopINT4EmbeddingWeight(dim, vocab, group, torch.bfloat16)
    embedding.load_hf_weights(
        {
            f"model.embed_tokens.{k}": v
            for k, v in {
                "qweight": packed,
                "scales": scales,
                "qzeros": zeros,
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


def test_independent_awq_input_scales_and_cuda_graph():
    from lightllm.common.quantization import Quantcfg
    from lightllm.models.sensenova_loop.layer_weights.awq_weight import InputScaledAWQBundle

    torch.manual_seed(7)
    config = {
        "n_layer": 1,
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "gemm",
        },
    }
    quant = Quantcfg(config)
    bundle = InputScaledAWQBundle(
        256,
        [128, 128],
        ["a.weight", "b.weight"],
        torch.bfloat16,
        lambda: quant.get_quant_method(0, "test"),
    )
    source = {}
    reference_weights, input_scales = [], []
    for name in ("a", "b"):
        qw = torch.randint(-(2 ** 31), 2 ** 31 - 1, (256, 16), dtype=torch.int32, device="cuda")
        qz = torch.randint(-(2 ** 31), 2 ** 31 - 1, (2, 16), dtype=torch.int32, device="cuda")
        scales = (torch.rand(2, 128, device="cuda") * 0.02 + 0.001).bfloat16()
        input_scale = torch.rand(256, device="cuda") + 0.2
        source.update(
            {f"{name}.qweight": qw, f"{name}.qzeros": qz, f"{name}.scales": scales, f"{name}.input_scale": input_scale}
        )
        # Canonical AWQ column packing: [0,2,4,6,1,3,5,7]. Decode independently
        # of Marlin; the legacy awq_dequantize extension supports FP16 only.
        shifts = torch.arange(8, device="cuda") * 4
        inverse = [0, 4, 1, 5, 2, 6, 3, 7]
        unpack = lambda packed: ((packed.unsqueeze(-1) >> shifts) & 15)[..., inverse].flatten(1)
        dense = unpack(qw).float() - unpack(qz).repeat_interleave(128, 0).float()
        dense = dense * scales.repeat_interleave(128, 0).float()
        reference_weights.append(dense.bfloat16())
        input_scales.append(input_scale)
    bundle.load_hf_weights(source)
    assert bundle.verify_load()
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)

    def reference():
        return torch.cat(
            [(x.float() / scale).bfloat16() @ weight for scale, weight in zip(input_scales, reference_weights)], dim=-1
        )

    actual = bundle.mm(x, use_custom_tensor_mananger=False)
    torch.testing.assert_close(actual, reference(), rtol=0.03, atol=0.025)
    # Warming up before capture also compiles the input-scale kernel.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bundle.mm(x, use_custom_tensor_mananger=False)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(captured, bundle.mm(x, use_custom_tensor_mananger=False), rtol=0, atol=0)


def test_input_scale_must_be_loaded_and_positive():
    from lightllm.common.quantization import Quantcfg
    from lightllm.models.sensenova_loop.layer_weights.awq_weight import InputScaledAWQWeight

    config = {
        "n_layer": 1,
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "gemm",
        },
    }
    weight = InputScaledAWQWeight(256, 128, "test.weight", torch.bfloat16, Quantcfg(config).get_quant_method(0, "test"))
    assert not weight.verify_load()
    with pytest.raises(ValueError, match="Invalid input scale"):
        weight.load_hf_weights({"test.input_scale": torch.zeros(256)})
