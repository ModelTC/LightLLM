"""Make HuggingFace transformers recognize the ``deepseek_v32`` model_type.

DeepSeek-V3.2 ships ``config.json`` with ``model_type="deepseek_v32"``, which
transformers (>=5.x) does not know. ``AutoTokenizer``/``AutoConfig`` then fall
back to the base ``PreTrainedConfig`` and crash during RoPE standardization
(``'PreTrainedConfig' object has no attribute 'max_position_embeddings'``).

V3.2 is architecturally a V3 variant, so we alias its config to
``DeepseekV3Config``. lightllm uses its own model implementation and reads
``config.json`` directly; this registration only fixes loading the HF tokenizer
through ``AutoTokenizer`` (see ``lightllm/server/tokenizer.py``).
"""
from transformers import AutoConfig

try:
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    class DeepseekV32Config(DeepseekV3Config):
        model_type = "deepseek_v32"

    AutoConfig.register("deepseek_v32", DeepseekV32Config, exist_ok=True)
except Exception:
    # Older transformers without deepseek_v3, or a build that already
    # supports deepseek_v32 natively. Nothing to do in either case.
    pass
