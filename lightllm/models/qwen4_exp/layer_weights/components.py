from functools import partial
from typing import Callable, Mapping

import torch

from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeight
from lightllm.common.basemodel.layer_weights.meta_weights.parameter_weight import ParameterWeight
from lightllm.common.gated_residual import GatedResidual
from lightllm.common.ngram_embedding import NGramHasher
from lightllm.common.ngram_embedding.ple import PLELayer
from lightllm.common.qsa.indexer import QSAIndexer
from lightllm.models.qwen4_exp.checkpoint import ngram_eos_token_id


class ComponentWeight(BaseWeight):
    def __init__(self, specs: Mapping[str, tuple], builder: Callable):
        super().__init__()
        self._specs = specs
        self._builder = builder
        self._component = None
        self._create_weight()

    def _create_weight(self):
        self.parameters = {name: ParameterWeight(name, dtype, shape) for name, (shape, dtype) in self._specs.items()}

    def load_hf_weights(self, weights):
        for name, parameter in self.parameters.items():
            if name not in weights:
                continue
            source = weights[name]
            shape, dtype = self._specs[name]
            if source.shape != shape:
                raise ValueError(f"Component checkpoint shape mismatch for {name}: {tuple(source.shape)} != {shape}")
            if dtype == torch.int64 and source.dtype != torch.int64:
                raise ValueError(f"Integer component buffer {name} must use int64")
            if dtype != torch.int64 and source.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"Component tensor {name} requires unquantized floating point weights")
            parameter.load_hf_weights(weights)

    def verify_load(self):
        ready = all(parameter.verify_load() for parameter in self.parameters.values())
        if ready and self._component is None:
            self._component = self._builder({name: parameter.weight for name, parameter in self.parameters.items()})
        return ready

    @property
    def component(self):
        if self._component is None and not self.verify_load():
            raise RuntimeError("Component weights are not fully loaded")
        return self._component


def gated_residual_weight(prefix: str, config: dict, dtype: torch.dtype, use_combine: bool = True):
    width = config["hc_count"] * config["hidden_size"]
    rank = config["hc_lowrank"]
    specs = {
        f"{prefix}.hc_norm.weight": ((width,), dtype),
        f"{prefix}.input_mix_weight_down.weight": ((rank, width), dtype),
        f"{prefix}.input_mix_weight_up.weight": ((width, rank), dtype),
    }
    if use_combine:
        specs[f"{prefix}.block_inject_weight.weight"] = ((config["hc_count"], width), dtype)
    return ComponentWeight(
        specs,
        partial(
            GatedResidual.from_checkpoint,
            prefix=prefix,
            branches=config["hc_count"],
            eps=config["rms_norm_eps"],
            use_combine=use_combine,
        ),
    )


def qsa_indexer_weight(prefix: str, config: dict, dtype: torch.dtype):
    dim, heads = config["indexer_head_dim"], config["indexer_n_heads"]
    if config["indexer_kv_heads"] != 1:
        raise ValueError("Qwen4Exp QSA requires one index key head")
    specs = {
        f"{prefix}.index_qk_proj.weight": (((heads + 1) * dim, config["hidden_size"]), dtype),
        f"{prefix}.q_layernorm.weight": ((dim,), dtype),
        f"{prefix}.k_layernorm.weight": ((dim,), dtype),
    }
    return ComponentWeight(
        specs,
        partial(
            QSAIndexer.from_checkpoint,
            prefix=prefix,
            query_heads=heads,
            token_budget=config["indexer_budget"],
            use_cuda_topk=True,
            compress_ratio=config["indexer_compress_ratio"],
            eps=config["rms_norm_eps"],
        ),
    )


def ple_weight(prefix: str, config: dict, dtype: torch.dtype):
    width = config["hc_count"] * config["hidden_size"]
    embed = config.get("ple_embed_dim", config["hidden_size"])
    specs = {
        f"{prefix}.key_proj.weight": ((width, embed), dtype),
        f"{prefix}.value_proj.weight": ((config["hidden_size"], embed), dtype),
        f"{prefix}.conv1d.weight": ((width, 1, config["ple_conv_kernel_size"]), dtype),
    }
    for norm in ("norm_key", "norm_query", "norm_conv"):
        specs[f"{prefix}.{norm}.weight"] = ((width,), dtype)
    return ComponentWeight(
        specs,
        partial(
            PLELayer.from_checkpoint,
            prefix=prefix,
            branches=config["hc_count"],
            dilation=config["ngram_size"],
            eps=config["rms_norm_eps"],
        ),
    )


def ngram_hasher_weight(prefix: str, config: dict):
    heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
    specs = {
        f"{prefix}.layer_multipliers": ((config["ngram_size"],), torch.int64),
        f"{prefix}.ngram_heads_vocab_sizes": ((heads,), torch.int64),
        f"{prefix}.ngram_heads_offsets": ((heads,), torch.int64),
    }
    return ComponentWeight(
        specs, partial(NGramHasher.from_checkpoint, prefix=prefix, eos_token_id=ngram_eos_token_id(config))
    )
