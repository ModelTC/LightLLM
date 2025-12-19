import torch
import dataclasses
from functools import lru_cache
from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
from lightllm.utils.log_utils import init_logger
from lightllm.utils.config_utils import get_hidden_size

logger = init_logger(__name__)


@dataclasses.dataclass
class EmbedCacheMeta:
    token_num: int
    layer_num: int
    hidden_size: int
    data_type: torch.dtype

    def calcu_size(self):
        return self.token_num * self.calcu_one_token_size()

    def calcu_one_token_size(self):
        return self.layer_num * self.hidden_size * self.data_type.itemsize


@lru_cache(maxsize=None)
def calcu_embed_cache_meta() -> "EmbedCacheMeta":
    args = get_env_start_args()
    assert args.enable_multimodal
    from lightllm.utils.llm_utils import get_llm_model_class
    from lightllm.models import Qwen3VLTpPartModel, Qwen3VLMOETpPartModel

    model_class = get_llm_model_class()
    model_dir = args.model_dir

    if model_class in [Qwen3VLTpPartModel, Qwen3VLMOETpPartModel]:
        embed_cache_meta_data = EmbedCacheMeta(
            token_num=None,
            layer_num=4,
            hidden_size=get_hidden_size(model_dir),
            data_type=get_llm_data_type(),
        )
    else:
        embed_cache_meta_data = EmbedCacheMeta(
            token_num=None,
            layer_num=1,
            hidden_size=get_hidden_size(model_dir),
            data_type=get_llm_data_type(),
        )

    token_num = int(
        (args.embed_cache_storage_size * 1024 * 1024 * 1024) / (embed_cache_meta_data.calcu_one_token_size())
    )
    embed_cache_meta_data.token_num = token_num

    logger.info(f"embed cache token num: {embed_cache_meta_data.token_num}")

    return embed_cache_meta_data
