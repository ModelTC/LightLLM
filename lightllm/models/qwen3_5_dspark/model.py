from lightllm.models.qwen3_dspark.model import Qwen3DSparkModel
from lightllm.models.qwen3_dspark.adapter import Qwen3DSparkModelAdapter


class Qwen3_5DSparkModel(Qwen3DSparkModel):
    """Adapter for the current Qwen3 DSpark checkpoint with a Qwen3.5 target."""

    model_adapter_class = Qwen3DSparkModelAdapter
