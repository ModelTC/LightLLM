import torch

from lightllm.models.qwen4_exp.layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from lightllm.models.qwen4_exp.batch import PackedBatchMetadata
from lightllm.models.qwen4_exp.lexical_cache import LexicalRequestCache
from lightllm.common.quantization import Quantcfg
from .helpers import checkpoint
from .reference import reference_ple


def make_cache(config):
    source = checkpoint(config, 1)
    weights = Qwen4ExpTransformerLayerWeight(1, torch.bfloat16, config, Quantcfg(config))
    weights.load_hf_weights(dict(source))
    weights.verify_load()
    return LexicalRequestCache(weights, 8), source


def metadata(requests, lengths, queries):
    tensors = [torch.tensor(values, device="cuda", dtype=torch.int32) for values in (requests, lengths, queries)]
    return PackedBatchMetadata.prepare(*tensors, sum(queries), 7, 4)


def reference_segment(residual, tokens, source, config, ngram, convolution):
    updates, states = [], [(ngram.clone(), convolution.clone())]
    for row in range(tokens.numel()):
        update, ngram, convolution = reference_ple(
            residual[row : row + 1], tokens[row : row + 1], source, config, ngram, convolution
        )
        updates.append(update)
        states.append((ngram.clone(), convolution.clone()))
    return torch.cat(updates), states
