import copy

from transformers import Qwen3Config
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

from .configuration_neo_vit import NEOVisionConfig


logger = logging.get_logger(__name__)


class NEOLLMConfig(Qwen3Config):
    def __init__(self, rope_theta_hw=10000.0, max_position_embeddings_hw=10000, **kwargs):
        super().__init__(**kwargs)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw


class NEOChatConfig(PretrainedConfig):
    model_type = 'neo_chat'
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        downsample_ratio=0.5,
        template=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['NEOVisionModel']}
            logger.info('vision_config is None. Initializing the NEOVisionConfig with default values.')

        if llm_config is None:
            llm_config = {'architectures': ['Qwen3ForCausalLM']}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')
        assert 'architectures' in llm_config, "Should specify architecture in llm_config"

        if isinstance(vision_config, dict):
            self.vision_config = NEOVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if isinstance(llm_config, dict):
            self.llm_config = NEOLLMConfig(**llm_config)
        else:
            self.llm_config = llm_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template

        return output
