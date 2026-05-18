import gguf
import numpy as np
import torch
from gguf import dequantize
from gguf.gguf_reader import GGUFReader, ReaderTensor
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from typing import Any, Callable, Dict, Optional

from lightllm.utils.dist_utils import get_current_device_id

DEQUANT_KEYS = ["token_embd.weight", "output.weight"]


def build_gguf_to_hf_mapping(config: Dict[str, Any]) -> Dict[str, str]:
    num_layers = config.get("num_hidden_layers")
    assert num_layers is not None, "num_hidden_layers is not found in config"
    model_type = config.get("model_type")
    assert model_type is not None, "model_type is not found in config"
    arch = None
    for gguf_arch, hf_arch in gguf.MODEL_ARCH_NAMES.items():
        if hf_arch == model_type:
            arch = gguf_arch
            break
    assert arch is not None, "model_type is not found in gguf.MODEL_ARCH_NAMES"
    tensor_name_map = gguf.get_tensor_name_map(arch, num_layers)

    config_cls = CONFIG_MAPPING[model_type]
    assert config_cls is not None, f"config_cls is not found in CONFIG_MAPPING for model_type={model_type}"
    hf_config = config_cls(**config)
    with torch.device("meta"):
        dummy_model = AutoModelForCausalLM.from_config(hf_config)
    gguf_to_hf_name_mapping = {}
    for hf_name in dummy_model.state_dict():
        name, extension = hf_name.rsplit(".", 1)
        gguf_name = tensor_name_map.get_name(name)
        gguf_to_hf_name_mapping[f"{gguf_name}.{extension}"] = hf_name

    return gguf_to_hf_name_mapping


def _reader_tensor_to_torch_cpu(rt: ReaderTensor, dequant: bool = False) -> torch.Tensor:
    assert rt.shape.ndim <= 2, "GGUF tensor must be 2D or less"
    if dequant:
        d_tensor = dequantize(rt.data, rt.tensor_type)
    else:
        d_tensor = rt.data
    arr = np.array(d_tensor, copy=True)

    return torch.from_numpy(arr)


def _gguf_reader_to_weight_dict(reader: GGUFReader) -> Dict[str, torch.Tensor]:
    gguf_weights = {}
    for t in reader.tensors:
        if t.name in DEQUANT_KEYS:
            gguf_weights[t.name] = _reader_tensor_to_torch_cpu(t, dequant=True)
        else:
            gguf_weights[t.name] = _reader_tensor_to_torch_cpu(t, dequant=False)
    return gguf_weights


def rename_weights(
    weights: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    gguf_to_hf: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    if gguf_to_hf is None:
        gguf_to_hf = build_gguf_to_hf_mapping(config)
    return {gguf_to_hf[k]: v for k, v in weights.items() if k in gguf_to_hf}


def load_gguf_weights(
    data_type: str,
    weight_dir: str,
    config: Dict[str, Any],
    pre_post_layer: Any = None,
    transformer_layer_list: Any = None,
    weight_dict: Optional[Dict[str, torch.Tensor]] = None,
    reader: Optional[GGUFReader] = None,
    gguf_to_hf: Optional[Dict[str, str]] = None,
    release_reader: Optional[Callable[[], None]] = None,
) -> None:
    if isinstance(data_type, str):
        data_type = torch.float16 if data_type == "fp16" else torch.float32
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == data_type, "type is not right"
    if transformer_layer_list is not None:
        assert transformer_layer_list[
            0].data_type_ == data_type, "type is not right"
    if weight_dict:
        torch.cuda.set_device(get_current_device_id())
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weight_dict)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weight_dict)
        del weight_dict
        if release_reader is not None:
            release_reader()
        return

    need_init_reader = reader is None
    if need_init_reader:
        reader = GGUFReader(weight_dir)
    try:
        weights = _gguf_reader_to_weight_dict(reader)
        weights = rename_weights(weights, config=config, gguf_to_hf=gguf_to_hf)
    finally:
        if need_init_reader:
            del reader
        elif release_reader is not None:
            release_reader()

    torch.cuda.set_device(get_current_device_id())
    if pre_post_layer is not None:
        pre_post_layer.load_hf_weights(weights)
    if transformer_layer_list is not None:
        for layer in transformer_layer_list:
            layer.load_hf_weights(weights)
    del weights
