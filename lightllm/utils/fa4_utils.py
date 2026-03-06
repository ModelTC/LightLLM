import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    from flash_attn.cute import flash_attn_varlen_func
    from flash_attn.cute.interface import _flash_attn_fwd

    HAS_FA4 = True
except Exception:
    flash_attn_varlen_func = None
    _flash_attn_fwd = None
    HAS_FA4 = False
    logger.warning("flash-attn-4 is not installed")


def is_fa4_supported_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major in (9, 10, 11, 12)


def ensure_fa4_available() -> None:
    if not HAS_FA4:
        raise ImportError(
            "flash-attn-4 is unavailable. Install it first, e.g. `pip install flash-attn-4`, "
            "or install from the local flash-attention repo."
        )


def ensure_fa4_supported_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("FA4 backend requires CUDA, but CUDA is not available.")
    major, minor = torch.cuda.get_device_capability()
    if major not in (9, 10, 11, 12):
        raise RuntimeError(
            f"FA4 backend requires Hopper/Blackwell-class GPUs (SM90/SM100/SM110/SM120). "
            f"Current device capability is {major}.{minor}."
        )


def sm90_fa4_paged_kv_tile_n(head_dim: int, head_dim_v: int, window_size: tuple[int, int] = (-1, -1)) -> int | None:
    major, _minor = torch.cuda.get_device_capability()
    if major != 9:
        return None

    is_local = window_size != (-1, -1)
    if head_dim <= 64:
        return 128
    if head_dim <= 96:
        return 128 if is_local else 144
    if head_dim <= 128:
        return 128
    if head_dim <= 192:
        return 96 if is_local else (128 if head_dim_v <= 128 else 112)
    return 64 if is_local else 80


def infer_fa4_page_size(model_dir: str) -> int | None:
    from transformers.configuration_utils import PretrainedConfig
    from lightllm.utils.device_utils import is_sm100_gpu

    if is_sm100_gpu():
        return 128

    model_cfg, _ = PretrainedConfig.get_config_dict(model_dir)
    llm_config = model_cfg.get("text_config", model_cfg)

    head_dim = llm_config.get("head_dim")
    if head_dim is None:
        head_dim = llm_config["hidden_size"] // llm_config["num_attention_heads"]
    head_dim_v = llm_config.get("v_head_dim", head_dim)

    window_size = (-1, -1)
    sliding_window = llm_config.get("sliding_window", None)
    if sliding_window is not None and not llm_config.get("full_attention_interval", None):
        window_size = (sliding_window - 1, sliding_window - 1)

    return sm90_fa4_paged_kv_tile_n(head_dim=head_dim, head_dim_v=head_dim_v, window_size=window_size)


def unwrap_fa4_output(output):
    return output[0] if isinstance(output, tuple) else output
