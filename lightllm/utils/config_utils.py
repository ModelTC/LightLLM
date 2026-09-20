import json
import os
from typing import Optional, List
from functools import lru_cache
from .envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.model_config import (
    get_text_config,
    load_model_config,
    load_model_config_dict,
    read_model_config,
    get_config_trust_remote_code,
)

logger = init_logger(__name__)


def _load_config(model_path, *, trust_remote_code=None):
    if trust_remote_code is None:
        trust_remote_code = get_config_trust_remote_code()
    return load_model_config(model_path, trust_remote_code=trust_remote_code)


def get_config_json(model_path: str, *, trust_remote_code=None):
    """Compatibility entry point returning the normalized, complete root config."""
    return load_model_config_dict(model_path, trust_remote_code=trust_remote_code)


def _derive_max_req_total_len_from_model_config(model_dir: str, *, trust_remote_code=None) -> Optional[int]:
    """
    Derive `max_req_total_len` from model config.json.

    Keep the derivation aligned with LightLLM's RoPE initialization logic:
    - If `max_sequence_length` exists: use it directly.
    - Otherwise: use `max_position_embeddings * rope_scaling.factor` (factor defaults to 1.0).
    """

    try:
        cfg = _load_config(model_dir, trust_remote_code=trust_remote_code)
    except Exception as e:
        logger.warning(f"failed to load config.json for max_req_total_len derive: {e}")
        return None

    candidates = [get_text_config(cfg), cfg]

    def _find_key(key: str):
        for c in candidates:
            value = getattr(c, key, None)
            if value is not None:
                return value
        return None

    def _find_rope_scaling() -> dict:
        rope_scaling = _find_key("rope_parameters") or _find_key("rope_scaling")
        if isinstance(rope_scaling, dict) and "full_attention" in rope_scaling:
            rope_scaling = rope_scaling["full_attention"]
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
    no_factor_types = {"yarn", "dynamic", "su", "longrope", "llama3"}
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


def auto_set_max_req_total_len(args) -> None:
    """
    Ensure `args.max_req_total_len` is an int.

    If the user provides a value, keep it.
    If it's None, auto-derive from config.json; fallback to 16384.
    """

    default_fallback = 16384
    if args.max_req_total_len is not None:
        return

    model_dir = args.model_dir
    if not model_dir:
        logger.warning("model_dir is empty; fallback max_req_total_len=16384")
        args.max_req_total_len = default_fallback
        return

    try:
        derived = _derive_max_req_total_len_from_model_config(
            model_dir, trust_remote_code=getattr(args, "trust_remote_code", False)
        )
    except Exception as e:
        logger.warning(f"failed to derive max_req_total_len from model config: {e}")
        derived = None

    if derived is None:
        logger.warning(f"cannot derive max_req_total_len from model config; fallback to {default_fallback}")
        args.max_req_total_len = default_fallback
        return

    args.max_req_total_len = int(derived)
    logger.info(f"auto derived max_req_total_len={args.max_req_total_len} from model config")


def auto_set_fused_shared_experts(args) -> None:
    """
    Route fused shared experts to supported model families and write the final
    decision to `args.enable_fused_shared_experts`.
    """

    if args.enable_fused_shared_experts:
        logger.info("skip auto setting fused shared experts: already enabled")
        return

    if args.enable_ep_moe:
        logger.info("do not enable fused shared experts: EP MoE uses a separate implementation")
        return

    model_dir = args.model_dir
    if not model_dir:
        logger.info("do not enable fused shared experts: model_dir is empty")
        return

    model_type = get_model_type(model_dir, trust_remote_code=getattr(args, "trust_remote_code", False))
    supported_model_types = {
        "deepseek_v3",
        "deepseek_v31",
        "deepseek_v32",
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
    if model_type not in supported_model_types:
        logger.info(f"do not enable fused shared experts: unsupported model_type={model_type}")
        return

    args.enable_fused_shared_experts = True
    logger.info(f"auto enable fused shared experts for model_type={model_type}")


def _get_config_llm_keyvalue(model_path: str, key_name: list[str], *, trust_remote_code=None):
    config_json = _load_config(model_path, trust_remote_code=trust_remote_code)
    for config in (get_text_config(config_json), config_json):
        for key in key_name:
            value = getattr(config, key, None)
            if value is not None:
                return value

    logger.error(f"cannot get {key_name} from config.json, return None")

    return None


def get_hidden_size(model_path: str, *, trust_remote_code=None) -> Optional[int]:
    hidden_size = _get_config_llm_keyvalue(
        model_path=model_path, key_name=["hidden_size", "n_embd", "n_embed"], trust_remote_code=trust_remote_code
    )
    if isinstance(hidden_size, int):
        return hidden_size
    return None


@lru_cache(maxsize=None)
def get_num_key_value_heads(model_path: str) -> int:
    num_key_value_heads = _get_config_llm_keyvalue(model_path=model_path, key_name=["num_key_value_heads"])
    if isinstance(num_key_value_heads, int):
        return num_key_value_heads
    return None


@lru_cache(maxsize=None)
def get_num_attention_heads(model_path: str) -> int:
    num_attention_heads = _get_config_llm_keyvalue(model_path=model_path, key_name=["num_attention_heads"])
    if isinstance(num_attention_heads, int):
        return num_attention_heads
    return None


@lru_cache(maxsize=None)
def get_head_dim(model_path: str) -> int:
    head_dim = _get_config_llm_keyvalue(model_path=model_path, key_name=["head_dim"])
    if isinstance(head_dim, int):
        return head_dim

    # calcu head_dim
    head_dim = get_hidden_size(model_path=model_path) // get_num_attention_heads(model_path=model_path)

    return head_dim


@lru_cache(maxsize=None)
def get_layer_num(model_path: str) -> int:
    num_hidden_layers = _get_config_llm_keyvalue(model_path=model_path, key_name=["num_hidden_layers"])
    if isinstance(num_hidden_layers, int):
        return num_hidden_layers
    return None


def get_eos_token_ids(model_path: str, *, trust_remote_code=None) -> Optional[List[int]]:
    # gemma4 special eos_token_id
    try:
        model_type = get_model_type(model_path, trust_remote_code=trust_remote_code)
        assert model_type == "gemma4"

        generation_config_path = os.path.join(model_path, "generation_config.json")
        with open(generation_config_path, "r") as file:
            eos_token_id = json.load(file).get("eos_token_id")

        assert eos_token_id is not None
        if isinstance(eos_token_id, int):
            return [eos_token_id]
        elif isinstance(eos_token_id, list):
            return list(eos_token_id)
    except:
        pass

    try:
        # qwen3-omini special eos_token_id
        config_json = get_config_json(model_path, trust_remote_code=trust_remote_code)
        assert config_json["architectures"][0] == "Qwen3OmniMoeForConditionalGeneration"
        return [151645]
    except:
        pass

    # Qwen3.5 checkpoints can have an eos_token_id in config that differs from
    # tokenizer.eos_token_id. In practice tokenizer.eos_token_id is the reliable
    # stop id (<|im_end|>, <|endoftext|>) for detokenization/stop behavior.
    try:
        config_json = get_config_json(model_path, trust_remote_code=trust_remote_code)
        model_type = config_json.get("model_type") or config_json.get("text_config", {}).get("model_type")
        if model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}:
            from transformers import AutoTokenizer

            eos_token_ids = []

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
                if trust_remote_code is not None
                else get_config_trust_remote_code(),
            )
            if tokenizer.eos_token_id is not None:
                eos_token_ids.append(int(tokenizer.eos_token_id))

            generation_config_path = os.path.join(model_path, "generation_config.json")
            if os.path.exists(generation_config_path):
                with open(generation_config_path, "r") as file:
                    generation_eos_token_id = json.load(file).get("eos_token_id")
                if isinstance(generation_eos_token_id, int):
                    eos_token_ids.append(generation_eos_token_id)
                elif isinstance(generation_eos_token_id, list):
                    eos_token_ids.extend(generation_eos_token_id)

            config_eos_token_id = _get_config_llm_keyvalue(
                model_path=model_path, key_name=["eos_token_id"], trust_remote_code=trust_remote_code
            )
            if isinstance(config_eos_token_id, int):
                eos_token_ids.append(config_eos_token_id)
            elif isinstance(config_eos_token_id, list):
                eos_token_ids.extend(config_eos_token_id)

            if eos_token_ids:
                return list(set(eos_token_ids))
    except Exception:
        pass

    eos_token_id = _get_config_llm_keyvalue(
        model_path=model_path, key_name=["eos_token_id"], trust_remote_code=trust_remote_code
    )
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, list):
        return eos_token_id

    assert False, "error eos_token_id format in config.json"
    return


def get_token_id(token: str) -> int:
    from lightllm.server.build_prompt import tokenizer

    return int(tokenizer.convert_tokens_to_ids(token))


def get_model_architectures(model_path: str):
    try:
        config = _load_config(model_path)
        arch = config.architectures[0]
        return arch
    except:
        logger.error("can not get architectures from config.json, return unknown_architecture")
        return "unknown_architecture"


def get_vocab_size(model_path: str):
    try:
        return int(_get_config_llm_keyvalue(model_path, ["vocab_size"]))
    except:
        logger.error("can not get vocab_size from config.json, return 0")
        return 0


def get_dtype(model_path: str, *, trust_remote_code=None):
    torch_dtype = _get_config_llm_keyvalue(
        model_path=model_path, key_name=["dtype", "torch_dtype", "model_dtype"], trust_remote_code=trust_remote_code
    )
    if torch_dtype is None:
        logger.warning("torch_dtype not in config.json, use float16 as default")
        return "float16"
    else:
        return str(torch_dtype).removeprefix("torch.")


@lru_cache(maxsize=None)
def get_fixed_kv_len():
    start_args = get_env_start_args()
    model_cfg = _load_config(start_args.model_dir)
    if getattr(model_cfg, "prompt_cache_token_ids", None) is not None:
        fixed_kv_len = len(model_cfg.prompt_cache_token_ids)
        # 固定 KV 最终会插入 radix cache，只加载完整的模型 KV 页面；不足一页
        # 的尾部直接截断，因此 router 也只扣除实际常驻的页面容量。
        return fixed_kv_len // start_args.page_size * start_args.page_size
    else:
        return 0


@lru_cache(maxsize=None)
def has_vision_module(model_path: str, *, trust_remote_code=None) -> bool:
    try:
        config = _load_config(model_path, trust_remote_code=trust_remote_code)
        if config.model_type == "qwen":
            return getattr(config, "visual", None) is not None
        if config.model_type == "llava":
            return True  # Original flat LLaVA stores mm_vision_tower instead of a component.
        config = getattr(config, "thinker_config", None) or config
        return getattr(config, "vision_config", None) is not None
    except Exception:
        logger.info(f"model path: {model_path} does not has vision module")
        return False


@lru_cache(maxsize=None)
def has_audio_module(model_path: str, *, trust_remote_code=None) -> bool:
    try:
        config = _load_config(model_path, trust_remote_code=trust_remote_code)
        config = getattr(config, "thinker_config", None) or config
        audio = getattr(config, "audio_config", None)
        model_type = audio.get("model_type") if isinstance(audio, dict) else getattr(audio, "model_type", None)
        return model_type in {"clap_audio_model", "whisper", "qwen3_omni_moe_audio_encoder"}
    except Exception:
        logger.info(f"model path: {model_path} does not has audio module")
        return False


@lru_cache(maxsize=None)
def is_linear_att_mixed_model(model_path: str) -> bool:
    return get_model_type(model_path) in {"qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}


def is_hybrid_att_model(model_path: str) -> bool:
    """Models whose non-full attention state follows hybrid checkpoint pages."""
    return is_linear_att_mixed_model(model_path)


def get_model_type(model_path: str, *, trust_remote_code=None) -> Optional[str]:
    """Get model type from config.json"""
    try:
        return (
            read_model_config(model_path).get("model_type")
            or _load_config(model_path, trust_remote_code=trust_remote_code).model_type
        )
    except Exception as e:
        logger.error(f"Failed to get model_type from {model_path}: {e}")
        return None


@lru_cache(maxsize=None)
def get_model_type_v1() -> Optional[str]:
    start_args = get_env_start_args()
    return get_model_type(start_args.model_dir)


def get_tool_call_parser_for_model(model_path: str, *, trust_remote_code=None) -> Optional[str]:
    """Auto-detect tool_call_parser based on model type"""
    model_type = get_model_type(model_path, trust_remote_code=trust_remote_code)
    if model_type is None:
        return None

    # Qwen3.5 series
    if model_type in ["qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"]:
        return "qwen3_coder"

    # Qwen3 series
    if model_type in [
        "qwen3",
        "qwen3_moe",
        "qwen3_omni_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_vl_text",
        "qwen3_vl_moe_text",
    ]:
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


def get_reasoning_parser_for_model(model_path: str, *, trust_remote_code=None) -> Optional[str]:
    """Auto-detect reasoning_parser based on model type"""
    model_type = get_model_type(model_path, trust_remote_code=trust_remote_code)
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
        "qwen3_omni_moe",
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

    # Gemma-4 (all variants share the same Harmony-like <|channel>...<channel|> format)
    if model_type == "gemma4":
        return "gemma4"

    return None


def auto_set_response_parsers(args) -> None:
    """Infer response parsers from model config unless explicitly configured."""
    if args.tool_call_parser is None:
        args.tool_call_parser = get_tool_call_parser_for_model(
            args.model_dir, trust_remote_code=getattr(args, "trust_remote_code", False)
        )
        if args.tool_call_parser:
            logger.info(f"Auto set tool_call_parser to {args.tool_call_parser} based on model type")

    if args.reasoning_parser is None:
        args.reasoning_parser = get_reasoning_parser_for_model(
            args.model_dir, trust_remote_code=getattr(args, "trust_remote_code", False)
        )
        if args.reasoning_parser:
            logger.info(f"Auto set reasoning_parser to {args.reasoning_parser} based on model type")


@lru_cache(maxsize=None)
def ffn_use_tanh_approximate_gelu() -> bool:
    try:
        start_args = get_env_start_args()
        model_type = get_model_type(start_args.model_dir)
        if model_type in ["gemma4"]:
            logger.info("Gemma4 uses tanh-approximate-gelu for FFN")
            return True
    except:
        pass

    return False
