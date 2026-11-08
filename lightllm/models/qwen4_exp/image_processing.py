from lightllm.models.qwen2_vl.vision_process import Qwen2VLImageProcessor


class Qwen4ExpImageProcessor(Qwen2VLImageProcessor):
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        config, overrides = cls.get_image_processor_dict(model_path, **kwargs)
        config.update(overrides)
        return cls(**config)
