# Configuration compatibility

Native Transformers 5.8 Config classes are preferred. These modules provide specific missing families and legacy layouts; they do not import LightLLM model implementations or wrap arbitrary unknown JSON in a generic Config.

| Definition | Configuration reference | Local handling |
| --- | --- | --- |
| `InternLMConfig` | [InternLM-7B](https://huggingface.co/internlm/internlm-7b/blob/96e127d08d851a88cac736a9b091dd953ae1b873/configuration_internlm.py) | Adapted configuration code; examples removed; copy the mutable rotary default. Original Apache notice retained. |
| `InternLM2Config` | [InternLM2-7B](https://huggingface.co/internlm/internlm2-7b/blob/8fbef1b948aa2f021030c942308f123e9ffe58bf/configuration_internlm2.py) | Adapted defaults and validation; examples removed. Original Apache notice retained. |
| `MiniCPMConfig` | [MiniCPM-2B](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/blob/4ec16344ac13e6ef5010aeecaa533369ac8eb53c/configuration_minicpm.py) | Adapted defaults and validation; examples and `flash_attn` probing removed. Original Apache notice retained. |
| `InternVisionConfig` | [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B/blob/853e3a797a661694b1b8ece0cb72dc2b23e3dac9/configuration_intern_vit.py) | Adapted constructor; standalone nested-file loading removed. Original MIT notice retained; see `INTERNVL_LICENSE`. |
| `InternVLChatConfig` | [InternVL3 component contract](https://huggingface.co/OpenGVLab/InternVL3-8B/blob/853e3a797a661694b1b8ece0cb72dc2b23e3dac9/configuration_internvl_chat.py) | Local component definition resolves registered backbones, including LightLLM's InternLM variants. |
| `QWenConfig` | [Qwen-7B field/default reference](https://huggingface.co/Qwen/Qwen-7B/blob/ef3c5c9c57b252f3149c1408daf4d649ec8b6c85/configuration_qwen.py) | Local field/default table delegates storage and serialization to HF. Does not substitute Qwen2 defaults. |
| `TarsierConfig` | [Official Tarsier configuration contract](https://github.com/bytedance/tarsier/blob/63de87a760c2aec6b31470a08046969a13787126/models/modeling_tarsier.py#L24) | Local component definition preserves missing components and Tarsier projector/token defaults without importing its modeling module. |
| `DeepseekV32Config` | Previous LightLLM `models/deepseek3_2/__init__.py` | Existing `DeepseekV3Config` alias moved out of the model package. |
| `LegacyLlavaConfig` | Existing LightLLM flat LLaVA backbone path | Llama configuration with the original `llava` identity. |
| `Qwen3_5DraftConfig` | Existing LightLLM DFlash/DSpark flat checkpoint path | HF Qwen3.5 text configuration with the older root `qwen3_5` identity. |

Custom `auto_map.AutoConfig` code takes precedence when the caller explicitly enables `trust_remote_code`. Failures propagate; unknown families have no generic successful fallback. Additional nonstandard components may remain dictionaries when that is the owning Config's own protocol (for example a Qwen audio extension).
