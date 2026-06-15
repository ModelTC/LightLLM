import gguf
import numpy as np
import os
import torch
from functools import lru_cache
from gguf import GGMLQuantizationType, dequantize, quant_shape_to_byte_shape
from gguf.gguf_reader import GGUFReader, ReaderTensor
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.log_utils import init_logger
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

try:
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
except ImportError:
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = {}

logger = init_logger(__name__)

DEQUANT_KEYS = frozenset(["token_embd.weight", "output.weight"])

# Native float tensors in GGUF are stored as plain arrays, not byte-packed blocks.
UNQUANTIZED_GGML_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16}


class LightLLMGGUFReader:

    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self._reader: Optional[GGUFReader] = None
        self._config: Optional[dict] = None
        self._gguf_to_hf: Optional[Dict[str, str]] = None
        self._quant_meta_map: Optional[Dict[str, Any]] = None

    @property
    def reader(self) -> GGUFReader:
        if self._reader is None:
            self._reader = GGUFReader(self.gguf_path)
        return self._reader

    def read_field(self, field_name: str) -> Any:
        """ Read a field from the GGUF reader. """
        field = self.reader.fields.get(field_name)
        if field is None:
            return None
        return field.contents()

    def load_config(self) -> dict:
        """ Load model config from the GGUF reader. """
        if self._config is not None:
            return self._config

        try:
            from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
        except ImportError as e:
            raise ImportError(
                "Loading config from GGUF requires transformers with GGUF support and the gguf package."
            ) from e
        config: dict = load_gguf_checkpoint(self.gguf_path, return_tensors=False)["config"]
        config["architectures"] = self._resolve_hf_architectures(config)

        self._config = config
        return config

    def get_gguf_to_hf_mapping(self, config: Optional[dict] = None) -> Dict[str, str]:
        """ Build a mapping from gguf tensor name to hf tensor name. """
        if self._gguf_to_hf is not None:
            return self._gguf_to_hf

        if config is None:
            config = self.load_config()

        self._gguf_to_hf = build_gguf_to_hf_mapping(config)
        return self._gguf_to_hf

    def build_quant_meta_map(self, config: Optional[dict] = None) -> Dict[str, Any]:
        """ Build a quant metadata map for GGUF quantized weights. """
        if self._quant_meta_map is not None:
            return self._quant_meta_map

        from lightllm.common.quantization.quantize_method import GGUFWeightMeta

        if config is None:
            config = self.load_config()
        gguf_to_hf = self.get_gguf_to_hf_mapping(config)
        gguf_quant_meta_map = {}
        for t in self.reader.tensors:
            if t.name in DEQUANT_KEYS:
                continue
            hf_name = gguf_to_hf.get(t.name)
            if hf_name is None:
                continue
            logical_shape = _gguf_logical_shape(t)
            if len(logical_shape) != 2:
                continue
            np_data = np.asarray(t.data)
            gguf_quant_meta_map[hf_name] = GGUFWeightMeta(
                shape=logical_shape,
                dtype=_numpy_dtype_to_torch(np_data.dtype),
                quant_type=t.tensor_type,
            )

        self._quant_meta_map = gguf_quant_meta_map
        return gguf_quant_meta_map

    def load_weights(
        self,
        data_type: str,
        config: dict,
        pre_post_layer: Any = None,
        transformer_layer_list: Any = None,
    ) -> None:
        """ Load GGUF weights into model layers, then release the reader. """
        if isinstance(data_type, str):
            data_type = torch.float16 if data_type == "fp16" else torch.float32

        if pre_post_layer is not None:
            assert pre_post_layer.data_type_ == data_type, "type is not right"

        if transformer_layer_list is not None:
            assert transformer_layer_list[0].data_type_ == data_type, "type is not right"

        try:
            gguf_to_hf = self.get_gguf_to_hf_mapping(config)
            gguf_tensor_names = {t.name for t in self.reader.tensors}
            validate_gguf_weight_mapping(
                gguf_to_hf=gguf_to_hf,
                gguf_tensor_names=gguf_tensor_names,
            )
            torch.cuda.set_device(get_current_device_id())
            gguf_weights = _gguf_reader_to_weight_dict(self.reader)
            hf_weights = rename_weights(gguf_weights, gguf_to_hf)
            del gguf_weights
            if pre_post_layer is not None:
                pre_post_layer.load_hf_weights(hf_weights)
            if transformer_layer_list is not None:
                for layer in transformer_layer_list:
                    layer.load_hf_weights(hf_weights)
            del hf_weights
        finally:
            self.close()

    def close(self) -> None:
        self._reader = None

    def __enter__(self) -> "LightLLMGGUFReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _resolve_hf_architectures(self, config: Dict[str, Any]) -> List[str]:
        """ Resolve HuggingFace architectures from model_type. """
        model_type = config.get("model_type")
        assert model_type is not None, "model_type is not found in config"
        assert model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, (
            f"model_type {model_type!r} is not found in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES"
        )

        return [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]]


@lru_cache(maxsize=10)
def get_gguf_reader(gguf_path: str) -> LightLLMGGUFReader:
    gguf_path = os.path.abspath(gguf_path)
    return LightLLMGGUFReader(gguf_path)


def _numpy_dtype_to_torch(np_dtype: np.dtype) -> torch.dtype:
    return torch.from_numpy(np.zeros((), dtype=np_dtype)).dtype


def _gguf_logical_shape(rt: ReaderTensor) -> Tuple[int, ...]:
    return tuple(reversed([int(x) for x in rt.shape.tolist()]))


def _resolve_model_type(model_type: str) -> str:
    MODEL_TYPE_MAPPING = {
        "qwen2_vl": "qwen2vl",
        "qwen2_5_vl": "qwen2vl",
    }

    return MODEL_TYPE_MAPPING.get(model_type, model_type)


def _normalize_hf_weight_name(hf_name: str) -> str:
    multimodal_lm_prefix = "model.language_model."
    if hf_name.startswith(multimodal_lm_prefix):
        return "model." + hf_name[len(multimodal_lm_prefix) :]

    return hf_name


def _is_multimodal(config: Dict[str, Any]) -> bool:
    if config.get("vision_config") is not None:
        return True
    model_type = config.get("model_type")
    return model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


def _dummy_hf_model_from_config(config: Dict[str, Any], hf_config: Any):
    with torch.device("meta"):
        if _is_multimodal(config):
            return AutoModelForImageTextToText.from_config(hf_config)
        return AutoModelForCausalLM.from_config(hf_config)


def build_gguf_to_hf_mapping(config: Dict[str, Any]) -> Dict[str, str]:
    """ 
    Build a mapping from gguf tensor name to hf tensor name, 
    e.g., 'token_embd.weight' -> 'model.embed_tokens.weight'. 
    """
    num_layers = config.get("num_hidden_layers")
    assert num_layers is not None, "num_hidden_layers is not found in config"
    model_type = config.get("model_type")
    assert model_type is not None, "model_type is not found in config"
    arch = None
    resolve_model_type = _resolve_model_type(model_type)
    for gguf_arch, hf_arch in gguf.MODEL_ARCH_NAMES.items():
        if hf_arch == resolve_model_type:
            arch = gguf_arch
            break
    assert arch is not None, f"model_type {model_type!r} is not found in gguf.MODEL_ARCH_NAMES"
    tensor_name_map = gguf.get_tensor_name_map(arch, num_layers)

    config_cls = CONFIG_MAPPING[model_type]
    assert config_cls is not None, f"config_cls is not found in CONFIG_MAPPING for model_type={model_type}"
    hf_config = config_cls(**config)
    dummy_model = _dummy_hf_model_from_config(config, hf_config)

    gguf_to_hf_name_mapping: Dict[str, str] = {}
    invalid_hf_names: List[str] = []
    for hf_name in dummy_model.state_dict():
        lightllm_hf_name = _normalize_hf_weight_name(hf_name)
        name, extension = lightllm_hf_name.rsplit(".", 1)
        gguf_name = tensor_name_map.get_name(name)
        if gguf_name is None:
            invalid_hf_names.append(hf_name)
            continue
        gguf_key = f"{gguf_name}.{extension}"
        if gguf_key in gguf_to_hf_name_mapping:
            raise ValueError(
                f"duplicate GGUF tensor key {gguf_key!r} while mapping HF weights; "
                f"existing={gguf_to_hf_name_mapping[gguf_key]!r}, new={lightllm_hf_name!r}"
            )
        gguf_to_hf_name_mapping[gguf_key] = lightllm_hf_name

    if invalid_hf_names:
        logger.warning(
            "skipped %d HF weight(s) with no GGUF tensor name mapping (first 5: %s)",
            len(invalid_hf_names),
            invalid_hf_names[:5],
        )
    return gguf_to_hf_name_mapping


def validate_gguf_weight_mapping(
    gguf_to_hf: Dict[str, str],
    gguf_tensor_names: Set[str],
) -> None:
    """ Validate GGUF↔HF mapping and log tensors that will not be loaded. """
    # Check if all gguf tensors are mapped to hf tensors
    mapped_gguf_names = set(gguf_to_hf.keys())
    unmapped_gguf = sorted(gguf_tensor_names - mapped_gguf_names)
    if unmapped_gguf:
        logger.warning(
            "GGUF file contains %d tensor(s) without HF mapping; they will be skipped (first 10: %s)",
            len(unmapped_gguf),
            unmapped_gguf[:10],
        )

    # Check if all hf tensors are mapped to gguf tensors
    missing_in_gguf = sorted(mapped_gguf_names - gguf_tensor_names)
    if missing_in_gguf:
        logger.warning(
            "HF mapping expects %d GGUF tensor(s) that are absent in the file (first 10: %s)",
            len(missing_in_gguf),
            missing_in_gguf[:10],
        )


def rename_weights(
    weights: Dict[str, torch.Tensor],
    gguf_to_hf: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """ Rename GGUF weights to HF names. """
    renamed: Dict[str, torch.Tensor] = {}
    for gguf_name, tensor in weights.items():
        hf_name = gguf_to_hf.get(gguf_name)
        if hf_name is None:
            continue
        renamed[hf_name] = tensor

    return renamed


def _normalize_visual_module_key(module_key: str) -> str:
    """ LightLLM visual modules omit the HF 'visual.' prefix used by GGUF MMPROJ. """
    if module_key.startswith("visual."):
        return module_key
    return f"visual.{module_key}"


def build_mmproj_to_hf_mapping(
    module_keys: Iterable[str],
    depth: int,
) -> Tuple[Dict[str, str], List[str]]:
    """ Build gguf tensor name -> LightLLM visual module state dict name mapping. """
    tensor_name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, depth)
    gguf_to_hf: Dict[str, str] = {}
    skipped: List[str] = []
    for hf_name in module_keys:
        prefix, extension = hf_name.rsplit(".", 1)
        lookup_prefix = _normalize_visual_module_key(prefix)
        gguf_prefix = tensor_name_map.get_name(lookup_prefix)
        if gguf_prefix is None:
            skipped.append(hf_name)
            continue
        gguf_key = f"{gguf_prefix}.{extension}"
        if gguf_key in gguf_to_hf:
            raise ValueError(
                f"duplicate GGUF tensor key {gguf_key!r} while mapping mmproj weights; "
                f"existing={gguf_to_hf[gguf_key]!r}, new={hf_name!r}"
            )
        gguf_to_hf[gguf_key] = hf_name

    if skipped:
        logger.warning(
            "skipped %d visual module weight(s) with no GGUF MMPROJ mapping (first 5: %s)",
            len(skipped),
            skipped[:5],
        )
    return gguf_to_hf, skipped


def _merge_mmproj_patch_embed(gguf_weights: Dict[str, torch.Tensor]) -> None:
    """ Merge split Conv2d patch-embed slices back into a Conv3d weight. """
    patch_key = "v.patch_embd.weight"
    w0 = gguf_weights.pop(patch_key, None)
    w1 = gguf_weights.pop(f"{patch_key}.1", None)
    if w0 is None:
        return
    if w1 is not None:
        # GGUF stores temporal_patch_size=2 as two [out, in, H, W] slices.
        gguf_weights[patch_key] = torch.stack([w0, w1], dim=2)
    else:
        gguf_weights[patch_key] = w0


def _merge_mmproj_attn_qkv(gguf_weights: Dict[str, torch.Tensor], depth: int) -> None:
    """ Merge split q/k/v tensors in mmproj into fused qkv weights. """
    for i in range(depth):
        prefix = f"v.blk.{i}"
        qw = gguf_weights.pop(f"{prefix}.attn_q.weight", None)
        kw = gguf_weights.pop(f"{prefix}.attn_k.weight", None)
        vw = gguf_weights.pop(f"{prefix}.attn_v.weight", None)
        if qw is not None:
            gguf_weights[f"{prefix}.attn_qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
        qb = gguf_weights.pop(f"{prefix}.attn_q.bias", None)
        kb = gguf_weights.pop(f"{prefix}.attn_k.bias", None)
        vb = gguf_weights.pop(f"{prefix}.attn_v.bias", None)
        if qb is not None:
            gguf_weights[f"{prefix}.attn_qkv.bias"] = torch.cat([qb, kb, vb], dim=0)


def _supplement_mmproj_qkv_mapping(
    gguf_to_hf: Dict[str, str],
    gguf_weights: Dict[str, torch.Tensor],
    depth: int,
) -> None:
    """ Add gguf->module mappings for qkv tensors created after q/k/v merge. """
    for i in range(depth):
        prefix = f"v.blk.{i}"
        for extension in ("weight", "bias"):
            gguf_key = f"{prefix}.attn_qkv.{extension}"
            if gguf_key in gguf_weights:
                gguf_to_hf[gguf_key] = f"blocks.{i}.attn.qkv.{extension}"


def load_mmproj_gguf_weights(
    mmproj_path: str,
    module_keys: Iterable[str],
    depth: int,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """ Load and dequantize mmproj GGUF weights into LightLLM visual module key names. """
    gguf_to_hf, _ = build_mmproj_to_hf_mapping(module_keys, depth)
    reader = get_gguf_reader(mmproj_path)
    try:
        gguf_weights = _gguf_reader_to_weight_dict(reader.reader, dequant_all=True)
        _merge_mmproj_patch_embed(gguf_weights)
        _merge_mmproj_attn_qkv(gguf_weights, depth)
        _supplement_mmproj_qkv_mapping(gguf_to_hf, gguf_weights, depth)
        weight_dict = rename_weights(gguf_weights, gguf_to_hf)
        return {name: tensor.to(dtype) for name, tensor in weight_dict.items()}
    finally:
        reader.close()


def _reader_tensor_to_torch_cpu(
    rt: ReaderTensor,
    dequant: bool = False,
    logical_shape: Tuple[int, ...] = None,
) -> torch.Tensor:
    """ Read a GGUF tensor to a CPU torch tensor. """
    assert rt.shape.ndim <= 2, "GGUF tensor must be 2D or less"
    if dequant:
        d_tensor = dequantize(rt.data, rt.tensor_type)
        if logical_shape is not None:
            d_tensor = d_tensor.reshape(logical_shape)
        
        return torch.from_numpy(np.array(d_tensor, copy=True))

    arr = np.array(rt.data, copy=True)
    if logical_shape is None:
        logical_shape = _gguf_logical_shape(rt)

    # Norm/bias vectors and native float weights are plain arrays, not block-quantized bytes.
    if rt.tensor_type in UNQUANTIZED_GGML_TYPES or len(logical_shape) == 1:
        if arr.size != np.prod(logical_shape):
            raise ValueError(
                f"GGUF tensor {rt.name} has size {arr.size} but expected {np.prod(logical_shape)}"
            )
        return torch.from_numpy(arr.reshape(logical_shape))

    byte_shape = quant_shape_to_byte_shape(logical_shape, rt.tensor_type)
    if arr.size != np.prod(byte_shape):
        raise ValueError(f"GGUF tensor {rt.name} has size {arr.size} but expected {np.prod(byte_shape)}")

    arr = arr.reshape(byte_shape)

    return torch.from_numpy(arr)
    

def _gguf_reader_to_weight_dict(
    reader: GGUFReader,
    *,
    dequant_all: bool = False,
) -> Dict[str, torch.Tensor]:
    """ Read GGUF reader to torch tensor dictionary. """
    gguf_weights = {}
    for t in reader.tensors:
        if dequant_all or t.name in DEQUANT_KEYS:
            gguf_weights[t.name] = _reader_tensor_to_torch_cpu(
                t, dequant=True, logical_shape=_gguf_logical_shape(t)
            )
        else:
            gguf_weights[t.name] = _reader_tensor_to_torch_cpu(t, dequant=False)
    return gguf_weights


def dequant_gguf_weight(
    byte_weight: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    logical_shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """ 
    Dequantize a GGUF weight to a CPU torch tensor during loading for GGUF.
    """
    expected_byte_shape = quant_shape_to_byte_shape(logical_shape, quant_type)
    if tuple(byte_weight.shape) != expected_byte_shape:
        raise ValueError(
            f"byte shard shape {byte_weight.shape} != expected {expected_byte_shape}"
        )
    fp32 = dequantize(byte_weight.contiguous().numpy(), quant_type)
    arr = np.asarray(fp32, dtype=np.float32).reshape(logical_shape)

    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def load_weight_shard(
    *,
    raw_weight: torch.Tensor,
    param_name: str,
    weight_pack: Any,
    slicer: Any,
    quant_method: Any,
) -> None:
    """ 
    Load a GGUF linear weight shard with TP slicing. 
    For predquant weights, dequantize the full tensor first, then slice.
    """
    if weight_pack.gguf_load_predquant:
        meta = quant_method.gguf_quant_meta_map[param_name]
        full_fp = dequant_gguf_weight(
            byte_weight=raw_weight,
            quant_type=meta.quant_type,
            logical_shape=meta.shape,
            device=weight_pack.weight.device,
            dtype=weight_pack.weight.dtype,
        )
        shard = slicer._slice_weight(full_fp)
        weight_pack.weight.copy_(shard)
        weight_pack.load_ok[0] = True
    else:
        shard = slicer._slice_weight(raw_weight)
        quant_method.load_weight(shard, weight_pack)
