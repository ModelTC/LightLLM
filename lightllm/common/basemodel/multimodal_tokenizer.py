"""
BaseMultiModalTokenizer is an abstract base class that defines the interface for a multimodal tokenizer.
This class serves as a blueprint for implementing tokenizers that handle multimodal inputs, such as text,
images, and audio. It provides methods for encoding prompts, initializing extra parameters for image and
audio items, and calculating token lengths for these modalities.

Attributes:
    tokenizer: The underlying tokenizer object that this class wraps. It provides the core text tokenization
               functionality.

Methods:
    encode(prompt, multimodal_params: MultimodalParams = None, **kwargs):
        Abstract method to encode a given prompt with optional multimodal parameters.
    
    init_imageitem_extral_params(img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams):
        Abstract method to initialize extra parameters for an image item in the context of multimodal processing.
    
    init_audioitem_extral_params(audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams):
        Abstract method to initialize extra parameters for an audio item in the context of multimodal processing.
    
    get_image_token_length(img: ImageItem):
        Abstract method to calculate the token length for a given image item.
    
    get_audio_token_length(audio: AudioItem):
        Abstract method to calculate the token length for a given audio item.
"""

from abc import ABC, abstractmethod
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.multimodal_params import AudioItem, ImageItem, MultimodalParams


class BaseMultiModalTokenizer(ABC):
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        obj_dict = object.__getattribute__(self, "__dict__")
        if name in obj_dict:
            return obj_dict[name]
        return getattr(self.tokenizer, name)

    @abstractmethod
    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        pass

    @abstractmethod
    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        pass

    @abstractmethod
    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        pass

    @abstractmethod
    def get_image_token_length(self, img: ImageItem):
        pass

    @abstractmethod
    def get_audio_token_length(self, audio: AudioItem):
        pass
