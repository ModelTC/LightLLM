import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Union

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelPaths:
    """ Resolved model-related paths for config / tokenizer / multimodal loading """
    model_dir: str
    tokenizer_dir: Optional[str] = None
    config_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    _gguf_path: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    @property
    def is_gguf(self) -> bool:
        return self._gguf_path is not None

    @property
    def gguf_path(self) -> Optional[str]:
        return self._gguf_path

    @property
    def processor_dir(self) -> str:
        return self.tokenizer_dir or self.model_dir

    @property
    def tokenizer_load_path(self) -> tuple[str, bool]:
        if self._gguf_path is not None and not self.tokenizer_dir:
            return self._gguf_path, True
        return self.processor_dir, False

    def resolve_visual_dirs(self) -> tuple[Optional[str], Optional[str]]:
        if self.is_gguf:
            return self.mmproj_path, self.tokenizer_dir
        return self.model_dir, self.model_dir

    def load_config(self) -> dict:
        if self.config_path is not None:
            return _load_config_from_path(self.config_path)

        gguf_path = self.gguf_path

        if gguf_path is not None and self.tokenizer_dir is not None:
            hf_config_path = os.path.join(self.tokenizer_dir, "config.json")
            assert os.path.isfile(hf_config_path), f"config.json {hf_config_path} is not found"
            return _load_config_from_path(hf_config_path)

        if gguf_path is None and self.model_dir:
            config_json_path = os.path.join(self.model_dir, "config.json")
            if os.path.isfile(config_json_path):
                return _load_config_from_path(config_json_path)

        if gguf_path is not None:
            return _load_config_from_gguf(gguf_path)

        raise FileNotFoundError(
            f"no model config found (config_path={self.config_path!r}, model_dir={self.model_dir!r}). "
            "Provide --config_path, place config.json under model_dir, or use a .gguf model path."
        )

    def align_quant_type(self, quant_type: str) -> str:
        if self.is_gguf:
            return "gguf"

        if quant_type == "gguf":
            raise ValueError("--quant_type gguf is not supported for non-GGUF models")

        return quant_type

    def __post_init__(self):
        object.__setattr__(self, "_gguf_path", _find_gguf_path_cached(self.model_dir))


def _fill_paths_from_env(paths: ModelPaths) -> ModelPaths:
    try:
        start_args = get_env_start_args()
    except KeyError:
        return paths

    config_path = paths.config_path if paths.config_path is not None else getattr(start_args, "config_path", None)
    tokenizer_dir = (
        paths.tokenizer_dir if paths.tokenizer_dir is not None else getattr(start_args, "tokenizer_dir", None)
    )
    mmproj_path = paths.mmproj_path if paths.mmproj_path is not None else getattr(start_args, "mmproj_path", None)

    if config_path == paths.config_path and tokenizer_dir == paths.tokenizer_dir and mmproj_path == paths.mmproj_path:
        return paths

    return ModelPaths(
        model_dir=paths.model_dir,
        config_path=config_path,
        tokenizer_dir=tokenizer_dir,
        mmproj_path=mmproj_path,
    )


def create_model_paths(
    model_dir_or_paths: Union[str, ModelPaths, None] = None,
    *,
    config_path: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    mmproj_path: Optional[str] = None,
) -> ModelPaths:
    if model_dir_or_paths is None:
        start_args = get_env_start_args()
        return create_model_paths(
            start_args.model_dir,
            config_path=getattr(start_args, "config_path", None),
            tokenizer_dir=getattr(start_args, "tokenizer_dir", None),
            mmproj_path=getattr(start_args, "mmproj_path", None),
        )

    if isinstance(model_dir_or_paths, ModelPaths):
        paths = model_dir_or_paths
    else:
        paths = ModelPaths(
            model_dir=model_dir_or_paths,
            config_path=config_path,
            tokenizer_dir=tokenizer_dir,
            mmproj_path=mmproj_path,
        )

    return _fill_paths_from_env(paths)


@lru_cache(maxsize=1)
def get_model_paths() -> ModelPaths:
    return create_model_paths()


def _load_config_from_path(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    with open(config_path, "r") as file:
        return json.load(file)


def _load_config_from_gguf(gguf_path: str) -> dict:
    from lightllm.common.basemodel.layer_weights.gguf_load_utils import get_gguf_reader

    return get_gguf_reader(gguf_path).load_config()


@lru_cache(maxsize=128)
def _find_gguf_path_cached(model_dir: Optional[str]) -> Optional[str]:
    if not model_dir:
        return None

    if model_dir.endswith(".gguf") and os.path.isfile(model_dir):
        return model_dir

    if os.path.isdir(model_dir):
        gguf_files = sorted(
            os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.endswith(".gguf")
        )
        if not gguf_files:
            return None
        if len(gguf_files) > 1:
            raise ValueError(
                f"multiple GGUF files found in {model_dir} is not supported, please specify the target gguf file."
            )
        return gguf_files[0]

    return None


def apply_gguf_quant_type(paths: Union[str, ModelPaths], quant_type: str) -> str:
    """Align quant_type for GGUF models and log when overriding."""
    paths = create_model_paths(paths)
    aligned = paths.align_quant_type(quant_type)
    if aligned != quant_type:
        logger.warning(
            f"model_dir contains GGUF weights; overriding --quant_type {quant_type!r} -> {aligned!r}"
        )
    return aligned


def _normalize_gguf_model_config(config: dict) -> dict:
    if config.get("head_dim") is None:
        hidden_size = config.get("hidden_size") or config.get("n_embd") or config.get("n_embed")
        num_heads = config.get("num_attention_heads") or config.get("n_head")
        if hidden_size and num_heads:
            config["head_dim"] = hidden_size // num_heads
    return config


@lru_cache(maxsize=None)
def get_model_config(paths: Union[str, ModelPaths]) -> dict:
    paths = create_model_paths(paths)
    config = paths.load_config()
    if paths.is_gguf:
        config = _normalize_gguf_model_config(config)
    return config


def check_gguf_multimodal_paths(paths: ModelPaths, enable_multimodal: bool = False) -> None:
    if not enable_multimodal or not paths.is_gguf:
        return

    if not paths.tokenizer_dir:
        raise ValueError("tokenizer_dir is required when enable_multimodal is True for GGUF models")
    if not os.path.isdir(paths.tokenizer_dir):
        raise FileNotFoundError(f"tokenizer_dir {paths.tokenizer_dir} is not found")

    effective_config_path = paths.config_path
    if effective_config_path is not None:
        if not os.path.isfile(effective_config_path):
            raise FileNotFoundError(f"config.json {effective_config_path} is not found")
    else:
        effective_config_path = os.path.join(paths.tokenizer_dir, "config.json")
        if not os.path.isfile(effective_config_path):
            raise FileNotFoundError(
                f"config.json is not provided and not found in tokenizer_dir: "
                f"{paths.tokenizer_dir} when enable_multimodal is True for GGUF models"
            )

    processor_path = os.path.join(paths.tokenizer_dir, "preprocessor_config.json")
    if not os.path.isfile(processor_path):
        raise FileNotFoundError(
            f"preprocessor_config.json not found in tokenizer_dir: "
            f"{paths.tokenizer_dir} when enable_multimodal is True for GGUF models"
        )

    if not paths.mmproj_path:
        raise ValueError("mmproj_path is required when enable_multimodal is True for GGUF models")
    if not os.path.isfile(paths.mmproj_path):
        raise FileNotFoundError(f"mmproj_path {paths.mmproj_path} is not found")


@lru_cache(maxsize=1)
def get_start_args_model_config() -> dict:
    return get_model_config(get_model_paths())


def _derive_max_req_total_len_from_model_config(paths: ModelPaths) -> Optional[int]:
    """
    Derive `max_req_total_len` from model config.json.

    Keep the derivation aligned with LightLLM's RoPE initialization logic:
    - If `max_sequence_length` exists: use it directly.
    - Otherwise: use `max_position_embeddings * rope_scaling.factor` (factor defaults to 1.0).
    """

    try:
        cfg = get_model_config(paths)
    except Exception as e:
        logger.warning(f"failed to load config.json for max_req_total_len derive: {e}")
        return None

    candidates = [cfg]

    llm_cfg = cfg.get("llm_config")
    if isinstance(llm_cfg, dict):
        candidates.append(llm_cfg)

    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        candidates.append(text_cfg)

    thinker_cfg = cfg.get("thinker_config")
    if isinstance(thinker_cfg, dict):
        thinker_text_cfg = thinker_cfg.get("text_config")
        if isinstance(thinker_text_cfg, dict):
            candidates.append(thinker_text_cfg)

    def _find_key(key: str):
        for c in candidates:
            if isinstance(c, dict) and key in c and c[key] is not None:
                return c.get(key)
        return None

    def _find_rope_scaling() -> dict:
        rope_scaling = _find_key("rope_scaling")
        if rope_scaling is None:
            return {}
        if isinstance(rope_scaling, dict):
            return rope_scaling
        return {}

    max_sequence_length = _find_key("max_sequence_length")
    if max_sequence_length is not None:
        try:
            val = int(max_sequence_length)
            if val > 0:
                return val
        except Exception:
            return None

    max_position_embeddings = _find_key("max_position_embeddings")
    if max_position_embeddings is None:
        return None

    rope_scaling = _find_rope_scaling()
    rope_type = None
    for k in ("rope_type", "type", "__type"):
        v = rope_scaling.get(k)
        if isinstance(v, str) and v.strip():
            rope_type = v.strip().lower()
            break

    # Align with `lightllm/models/llama/model.py` RoPE initialization:
    # - `yarn/dynamic/su/llama3`: do NOT multiply by `rope_scaling.factor` for max length.
    # - `default/mrope` (and unknown): multiply by factor when present.
    no_factor_types = {"yarn", "dynamic", "su", "llama3"}
    multiply_factor = True
    if rope_type is not None and rope_type in no_factor_types:
        multiply_factor = False

    try:
        factor_raw = rope_scaling.get("factor", 1.0)
        factor = 1.0 if factor_raw is None else float(factor_raw)
    except Exception:
        factor = 1.0

    try:
        max_pos = float(max_position_embeddings)
        val = int(max_pos * factor) if multiply_factor else int(max_pos)
        if val > 0:
            logger.info(
                "auto set max_req_total_len=%s (rope_type=%s,max_position_embeddings=%s,factor=%s, multiply_factor=%s)",
                val,
                rope_type,
                max_position_embeddings,
                factor,
                multiply_factor,
            )
            return val
    except Exception:
        return None

    return None


def auto_set_max_req_total_len(args, paths: ModelPaths) -> None:
    """
    Ensure `args.max_req_total_len` is an int.

    If the user provides a value, keep it.
    If it's None, auto-derive from config.json; fallback to 16384.
    """

    default_fallback = 16384
    if args.max_req_total_len is not None:
        return

    if not paths.model_dir:
        logger.warning("model_dir is empty; fallback max_req_total_len=16384")
        args.max_req_total_len = default_fallback
        return

    try:
        derived = _derive_max_req_total_len_from_model_config(paths)
    except Exception as e:
        logger.warning(f"failed to derive max_req_total_len from model config: {e}")
        derived = None

    if derived is None:
        logger.warning(f"cannot derive max_req_total_len from model config; fallback to {default_fallback}")
        args.max_req_total_len = default_fallback
        return

    args.max_req_total_len = int(derived)
    logger.info(f"auto derived max_req_total_len={args.max_req_total_len} from model config")


def _get_config_llm_keyvalue(paths: ModelPaths, key_name: list[str]):
    config_json = get_model_config(paths)
    for key in key_name:
        try:
            value = config_json[key]
        except:
            # for some multimodal model
            try:
                value = config_json["llm_config"][key]
            except:
                value = config_json.get("text_config", {}).get(key)
        if config_json.get("thinker_config") is not None:
            value = config_json.get("thinker_config", {}).get("text_config").get(key)
        if value is not None:
            return value

    logger.error(f"cannot get {key_name} from config.json, return None")

    return None


def get_hidden_size(model_dir_or_paths: Union[str, ModelPaths]) -> Optional[int]:
    paths = create_model_paths(model_dir_or_paths)
    hidden_size = _get_config_llm_keyvalue(paths, key_name=["hidden_size", "n_embd", "n_embed"])
    if isinstance(hidden_size, int):
        return hidden_size
    return None


@lru_cache(maxsize=None)
def get_num_key_value_heads(model_dir_or_paths: Union[str, ModelPaths]) -> int:
    paths = create_model_paths(model_dir_or_paths)
    num_key_value_heads = _get_config_llm_keyvalue(paths, key_name=["num_key_value_heads"])
    if isinstance(num_key_value_heads, int):
        return num_key_value_heads
    return None


@lru_cache(maxsize=None)
def get_num_attention_heads(model_dir_or_paths: Union[str, ModelPaths]) -> int:
    paths = create_model_paths(model_dir_or_paths)
    num_attention_heads = _get_config_llm_keyvalue(paths, key_name=["num_attention_heads"])
    if isinstance(num_attention_heads, int):
        return num_attention_heads
    return None


@lru_cache(maxsize=None)
def get_head_dim(model_dir_or_paths: Union[str, ModelPaths]) -> int:
    paths = create_model_paths(model_dir_or_paths)
    head_dim = _get_config_llm_keyvalue(paths, key_name=["head_dim"])
    if isinstance(head_dim, int):
        return head_dim

    head_dim = get_hidden_size(paths) // get_num_attention_heads(paths)

    return head_dim


@lru_cache(maxsize=None)
def get_layer_num(model_dir_or_paths: Union[str, ModelPaths]) -> int:
    paths = create_model_paths(model_dir_or_paths)
    num_hidden_layers = _get_config_llm_keyvalue(paths, key_name=["num_hidden_layers"])
    if isinstance(num_hidden_layers, int):
        return num_hidden_layers
    return None


def get_eos_token_ids(model_dir_or_paths: Union[str, ModelPaths]) -> Optional[List[int]]:
    paths = create_model_paths(model_dir_or_paths)
    try:
        config_json = get_model_config(paths)
        assert config_json["architectures"][0] == "Qwen3OmniMoeForConditionalGeneration"
        return [151645]
    except:
        pass

    # Qwen3.5 checkpoints can have an eos_token_id in config that differs from
    # tokenizer.eos_token_id. In practice tokenizer.eos_token_id is the reliable
    # stop id (<|im_end|>)
    try:
        config_json = get_model_config(paths)
        model_type = config_json.get("model_type") or config_json.get("text_config", {}).get("model_type")
        if model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(paths.processor_dir, trust_remote_code=False)
            if tokenizer.eos_token_id is not None:
                return [int(tokenizer.eos_token_id)]
    except Exception:
        pass

    eos_token_id = _get_config_llm_keyvalue(paths, key_name=["eos_token_id"])
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, list):
        return eos_token_id

    assert False, "error eos_token_id format in config.json"
    return


def get_model_architectures(model_dir_or_paths: Union[str, ModelPaths]):
    paths = create_model_paths(model_dir_or_paths)
    try:
        config_json = get_model_config(paths)
        arch = config_json["architectures"][0]
        return arch
    except:
        logger.error("can not get architectures from config.json, return unknown_architecture")
        return "unknown_architecture"


def get_vocab_size(paths: Optional[Union[str, ModelPaths]] = None) -> int:
    try:
        if paths is None:
            paths = get_model_paths()
        elif not isinstance(paths, ModelPaths):
            paths = create_model_paths(paths)
        config_json = get_model_config(paths)
        # qwen3-omini special
        if "thinker_config" in config_json:
            config_json = config_json["thinker_config"]
        if "llm_config" in config_json:
            vocab_size = int(config_json["llm_config"]["vocab_size"])
            return vocab_size
        elif "text_config" in config_json:
            vocab_size = int(config_json["text_config"]["vocab_size"])
            return vocab_size
        vocab_size = config_json["vocab_size"]
        if not isinstance(vocab_size, int):
            vocab_size = int(vocab_size)
        return vocab_size
    except:
        logger.error("can not get vocab_size from config.json, return 0")
        return 0


def get_dtype(model_dir_or_paths: Union[str, ModelPaths]):
    paths = create_model_paths(model_dir_or_paths)
    torch_dtype = _get_config_llm_keyvalue(paths, key_name=["torch_dtype", "dtype", "model_dtype"])
    if torch_dtype is None:
        logger.warning("torch_dtype not in config.json, use float16 as default")
        return "float16"
    else:
        return torch_dtype


@lru_cache(maxsize=None)
def get_fixed_kv_len():
    model_cfg = get_start_args_model_config()
    if "prompt_cache_token_ids" in model_cfg:
        return len(model_cfg["prompt_cache_token_ids"])
    else:
        return 0


@lru_cache(maxsize=None)
def has_vision_module(model_dir_or_paths: Union[str, ModelPaths]) -> bool:
    paths = create_model_paths(model_dir_or_paths)
    try:
        model_cfg = get_model_config(paths)
        model_type = model_cfg["model_type"]
        if model_type == "qwen":
            # QWenVisionTransformer
            model_cfg["visual"]
            return True
        elif model_type == "qwen2_vl":
            # Qwen2VisionTransformerPretrainedModel
            model_cfg["vision_config"]
            return True
        elif model_type == "qwen2_5_vl":
            # Qwen2_5_VisionTransformerPretrainedModel
            model_cfg["vision_config"]
            return True
        elif model_type in ["qwen3_vl", "qwen3_vl_moe"]:
            # Qwen3VisionTransformerPretrainedModel
            model_cfg["vision_config"]
            return True
        elif model_cfg["architectures"][0] == "TarsierForConditionalGeneration":
            # TarsierVisionTransformerPretrainedModel
            return True
        elif model_type == "llava":
            # LlavaVisionModel
            return True
        elif model_type == "internvl_chat":
            return True
        elif model_type == "gemma3":
            return True
        elif (
            model_cfg.get("thinker_config", {}).get("vision_config", {}).get("model_type")
            == "qwen3_omni_moe_vision_encoder"
        ):
            # Qwen3OmniMoeVisionTransformerPretrainedModel
            return True
        elif model_type in ["qwen3_5", "qwen3_5_moe"]:
            return True
        else:
            raise Exception("unknown vision model type")
    except:
        logger.info(f"model path: {paths.model_dir} does not has vision module")
        return False


@lru_cache(maxsize=None)
def has_audio_module(model_dir_or_paths: Union[str, ModelPaths]) -> bool:
    paths = create_model_paths(model_dir_or_paths)
    try:
        model_cfg = get_model_config(paths)
        if model_cfg.get("thinker_config") is not None:
            model_cfg = model_cfg["thinker_config"]
        audio_config = model_cfg["audio_config"]
        model_type = audio_config["model_type"]
        if model_type == "clap_audio_model" or model_type == "whisper":
            # WhisperAudioModel
            return True
        elif model_type == "qwen3_omni_moe_audio_encoder":
            # Qwen3OmniMoeAudioEncoder
            return True
        else:
            raise Exception("unknown audio model type")
    except:
        logger.info(f"model path: {paths.model_dir} does not has audio module")
        return False


@lru_cache(maxsize=None)
def is_linear_att_mixed_model(model_dir_or_paths: Union[str, ModelPaths]) -> bool:
    paths = create_model_paths(model_dir_or_paths)
    try:
        model_cfg = get_model_config(paths)
        model_type = model_cfg["model_type"]
        if model_type in ["qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"]:
            return True
        else:
            return False
    except:
        logger.info(f"model path: {paths.model_dir} does not has linear hybrid attention")
        return False


def get_model_type(paths: Union[str, ModelPaths]) -> Optional[str]:
    """Get model type from model config."""
    try:
        config_json = get_model_config(paths)
        model_type = config_json.get("model_type") or config_json.get("text_config", {}).get("model_type")
        return model_type
    except Exception as e:
        logger.error(f"Failed to get model_type (paths={paths!r}): {e}")
        return None


def get_tool_call_parser_for_model(paths: Union[str, ModelPaths]) -> Optional[str]:
    """Auto-detect tool_call_parser based on model type"""
    model_type = get_model_type(paths)
    if model_type is None:
        return None

    # Qwen3.5 series
    if model_type in ["qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"]:
        return "qwen3_coder"

    # Qwen3 series
    if model_type in ["qwen3", "qwen3_moe", "qwen3_vl", "qwen3_vl_moe", "qwen3_vl_text", "qwen3_vl_moe_text"]:
        return "qwen25"

    # DeepSeek V3
    if model_type == "deepseek_v3":
        return "deepseekv3"

    # DeepSeek V3.1
    if model_type == "deepseek_v31":
        return "deepseekv31"

    # DeepSeek V32
    if model_type == "deepseek_v32":
        return "deepseekv32"

    return None


def get_reasoning_parser_for_model(paths: Union[str, ModelPaths]) -> Optional[str]:
    model_type = get_model_type(paths)
    if model_type is None:
        return None

    # Qwen3.5 and Qwen3 series
    if model_type in [
        "qwen3",
        "qwen3_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_vl_text",
        "qwen3_vl_moe_text",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_text",
        "qwen3_5_moe_text",
    ]:
        return "qwen3"

    # DeepSeek V3
    if model_type in ["deepseek_v3", "deepseek_v31", "deepseek_v32"]:
        return "deepseek-v3"

    # DeepSeek R1
    if model_type == "deepseek_r1":
        return "deepseek-r1"

    return None
