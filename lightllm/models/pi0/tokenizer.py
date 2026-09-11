from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.utils.config_utils import get_pi_config, get_vocab_size


class Pi0Tokenizer:
    """PaliGemma tokenizer adapter with an explicit tokenized-input escape hatch."""

    def __init__(
        self,
        max_length: int,
        tokenizer_name_or_path: str,
        *,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ):
        self.max_length = max_length
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = None
        self.sentencepiece = None
        tokenizer_path = Path(tokenizer_name_or_path)
        if tokenizer_path.is_file():
            import sentencepiece

            self.sentencepiece = sentencepiece.SentencePieceProcessor(
                model_file=str(tokenizer_path)
            )
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                local_files_only=True,
                use_fast=True,
                cache_dir=cache_dir,
            )

    def format_prompt(self, prompt: str) -> str:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        return f"{cleaned}\n"

    def _encode_texts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad already formatted prompts with the loaded tokenizer backend."""

        if self.sentencepiece is not None:
            token_rows = []
            mask_rows = []
            for text in texts:
                tokens = self._sentencepiece_encode(text)[: self.max_length]
                mask = [True] * len(tokens)
                padding = self.max_length - len(tokens)
                token_rows.append(tokens + [0] * padding)
                mask_rows.append(mask + [False] * padding)
            return torch.tensor(token_rows, dtype=torch.long), torch.tensor(
                mask_rows, dtype=torch.bool
            )
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"].bool()

    def _sentencepiece_encode(self, text: str) -> list[int]:
        cleaned = text[:-1] if text.endswith("\n") else text
        return self.sentencepiece.encode(
            cleaned, add_bos=True
        ) + self.sentencepiece.encode("\n")

    def tokenize(
        self, prompts: str | Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        return self._encode_texts([self.format_prompt(prompt) for prompt in prompts])


def resolve_tokenizer_path(model_dir: str, configured_path: str | None = None) -> str:
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path))
    model_path = Path(model_dir)
    candidates.extend(
        (
            model_path / "paligemma_tokenizer.model",
            model_path / "tokenizer.model",
            model_path / "tokenizer",
        )
    )
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return str(candidate)
    raise ValueError(
        "Pi0 tokenizer assets were not found locally; pass "
        "--vla_tokenizer_path pointing to paligemma_tokenizer.model or a "
        "local tokenizer directory"
    )


class Pi0TokenizerAdapter(BaseMultiModalTokenizer):
    """Expose OpenPI tokenization through LightLLM's multimodal tokenizer API."""

    # LeRobot normally records these under input_features.  Keep the OpenPI
    # camera order here as a fallback because this adapter owns named-image
    # validation and request ordering; the vision encoder only sees pixels.
    DEFAULT_IMAGE_KEYS = (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )

    all_special_ids = [0, 1, 2]
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    chat_template = None
    tokenizer_class = Pi0Tokenizer

    def __init__(self, model_dir: str, tokenizer_path: str | None = None):
        # Use the same startup overrides as VLAServer.
        from lightllm.utils.envs_utils import get_env_start_args

        try:
            args = get_env_start_args()
        except KeyError:
            args = None
        self.config = (
            getattr(args, "generation_runtime_config", None)
            if args is not None and getattr(args, "model_dir", None) == model_dir
            else None
        )
        if self.config is None:
            self.config = get_pi_config(model_dir)
        self.vocab_size = get_vocab_size(model_dir)
        self.image_keys = self.config["image_keys"] or self.DEFAULT_IMAGE_KEYS
        self.image_length = (self.config["image_resolution"][0] // 14) * (
            self.config["image_resolution"][1] // 14
        )
        self.policy_tokenizer = self.tokenizer_class(
            self.config["tokenizer_max_length"],
            resolve_tokenizer_path(model_dir, tokenizer_path),
        )
        super().__init__(self.policy_tokenizer)

    def init_imageitem_extral_params(self, img, multi_params, sampling_params):
        input_key = getattr(img, "input_key", None)
        if input_key is not None and input_key not in self.image_keys:
            raise ValueError(
                f"unknown pi0 image input_key={input_key!r}; expected one of "
                f"{list(self.image_keys)!r}"
            )
        return

    def _ordered_images(self, images):
        """Resolve named cameras in checkpoint order; retain legacy list order."""

        images = list(images)
        named = [getattr(image, "input_key", None) for image in images]
        if not any(name is not None for name in named):
            return images
        if any(name is None for name in named):
            raise ValueError("pi0 images must be either all named or all positional")
        if len(named) != len(set(named)):
            raise ValueError("pi0 image input_key values must be unique")
        unknown = set(named).difference(self.image_keys)
        if unknown:
            raise ValueError(
                "unknown pi0 image input keys: " + ", ".join(sorted(unknown))
            )
        by_name = dict(zip(named, images, strict=True))
        return [by_name[key] for key in self.image_keys if key in by_name]

    def init_audioitem_extral_params(self, audio, multi_params, sampling_params):
        raise NotImplementedError("pi0 does not have an audio encoder")

    def get_image_token_length(self, img) -> int:
        return self.image_length

    def get_audio_token_length(self, audio) -> int:
        raise NotImplementedError("pi0 does not have an audio encoder")

    def _tokenize_prompt(self, prompt: str, multimodal_params):
        return self.policy_tokenizer.tokenize([prompt])

    def encode(
        self,
        prompt: str,
        multimodal_params=None,
        add_special_tokens: bool = True,
        **_kwargs,
    ) -> list[int]:
        token_ids, token_mask = self._tokenize_prompt(prompt, multimodal_params)
        text_ids = token_ids[0, token_mask[0]].tolist()

        image_ids = []
        if multimodal_params is not None:
            for image in self._ordered_images(multimodal_params.images):
                if image.token_id is None or image.token_num is None:
                    raise ValueError(
                        "pi0 image cache resources must be allocated before tokenization"
                    )
                if image.token_num != self.image_length:
                    raise ValueError(
                        f"invalid pi0 image token count: {image.token_num} != {self.image_length}"
                    )
                image_ids.extend(
                    range(image.token_id, image.token_id + image.token_num)
                )
        return image_ids + text_ids

    def decode(self, token_ids, **_kwargs) -> str:
        if self.policy_tokenizer.sentencepiece is not None:
            return self.policy_tokenizer.sentencepiece.decode(list(token_ids))
        return self.policy_tokenizer.tokenizer.decode(token_ids, **_kwargs)

    def get_vocab(self) -> dict[str, int]:
        # Action requests never enter detokenization.  The small special-token
        # view is sufficient for the shared detokenizer's initialization.
        return {"<pad>": 0, "</s>": 1, "<bos>": 2}

    def convert_tokens_to_string(self, tokens) -> str:
        return "".join(tokens)
