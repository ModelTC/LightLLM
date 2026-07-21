from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import VLAModelType
from .config import Pi0VLAConfig


def discretize_state_256(state: np.ndarray | torch.Tensor | Sequence[float]):
    """Discretize normalized pi0.5 state with OpenPI's exact 256-bin rule."""
    if isinstance(state, torch.Tensor):
        boundaries = torch.linspace(-1.0, 1.0, 257, dtype=torch.float64, device=state.device)[:-1]
        return torch.searchsorted(boundaries, state.to(torch.float64), right=True) - 1
    values = np.asarray(state)
    boundaries = np.linspace(-1.0, 1.0, 257)[:-1]
    return np.digitize(values, bins=boundaries) - 1


def format_pi05_prompt(prompt: str, state: np.ndarray | torch.Tensor | Sequence[float]) -> str:
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    bins = discretize_state_256(state)
    if isinstance(bins, torch.Tensor):
        bins = bins.detach().cpu().reshape(-1).tolist()
    else:
        bins = bins.reshape(-1).tolist()
    state_text = " ".join(str(int(value)) for value in bins)
    return f"Task: {cleaned}, State: {state_text};\nAction: "


class Pi0Tokenizer:
    """PaliGemma tokenizer adapter with an explicit tokenized-input escape hatch."""

    def __init__(
        self,
        model_type: str | VLAModelType,
        max_length: int,
        tokenizer_name_or_path: str,
        *,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ):
        self.model_type = VLAModelType(model_type)
        self.max_length = max_length
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = None
        self.sentencepiece = None
        tokenizer_path = Path(tokenizer_name_or_path)
        if tokenizer_path.is_file():
            import sentencepiece

            self.sentencepiece = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                local_files_only=True,
                use_fast=True,
                cache_dir=cache_dir,
            )

    def format_prompt(self, prompt: str, state=None) -> str:
        if self.model_type is VLAModelType.PI05:
            if state is None:
                raise ValueError("pi0.5 prompt tokenization requires normalized state")
            return format_pi05_prompt(prompt, state)
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        return f"{cleaned}\n"

    def tokenize(self, prompts: str | Sequence[str], states=None) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if self.model_type is VLAModelType.PI05:
            if states is None or len(states) != len(prompts):
                raise ValueError("pi0.5 states must match the prompt batch")
            texts = [self.format_prompt(prompt, state) for prompt, state in zip(prompts, states, strict=True)]
        else:
            texts = [self.format_prompt(prompt) for prompt in prompts]
        if self.sentencepiece is not None:
            token_rows = []
            mask_rows = []
            for text in texts:
                if self.model_type is VLAModelType.PI05:
                    tokens = self.sentencepiece.encode(text, add_bos=True)
                else:
                    cleaned = text[:-1] if text.endswith("\n") else text
                    tokens = self.sentencepiece.encode(cleaned, add_bos=True) + self.sentencepiece.encode("\n")
                tokens = tokens[: self.max_length]
                mask = [True] * len(tokens)
                padding = self.max_length - len(tokens)
                token_rows.append(tokens + [0] * padding)
                mask_rows.append(mask + [False] * padding)
            return torch.tensor(token_rows, dtype=torch.long), torch.tensor(mask_rows, dtype=torch.bool)
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"].bool()


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


class Pi0TokenizerAdapter:
    """Expose OpenPI tokenization through LightLLM's multimodal tokenizer API."""

    all_special_ids = [0, 1, 2]
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    chat_template = None

    def __init__(self, model_dir: str, tokenizer_path: str | None = None):
        self.config = Pi0VLAConfig.from_model_dir(model_dir)
        self.vocab_size = self.config.vocab_size
        self.image_length = (self.config.image_resolution[0] // 14) * (self.config.image_resolution[1] // 14)
        self.policy_tokenizer = Pi0Tokenizer(
            self.config.model_type,
            self.config.tokenizer_max_length,
            resolve_tokenizer_path(model_dir, tokenizer_path),
        )

    def init_imageitem_extral_params(self, img, multi_params, sampling_params):
        return

    def init_audioitem_extral_params(self, audio, multi_params, sampling_params):
        raise NotImplementedError("pi0 does not have an audio encoder")

    def get_image_token_length(self, img) -> int:
        return self.image_length

    def get_audio_token_length(self, audio) -> int:
        raise NotImplementedError("pi0 does not have an audio encoder")

    def encode(
        self,
        prompt: str,
        multimodal_params=None,
        add_special_tokens: bool = True,
        **_kwargs,
    ) -> list[int]:
        state = None if multimodal_params is None else getattr(multimodal_params, "state", None)
        # Compatibility for callers that still carry state only inside the
        # action payload.  New text-only Pi0.5 callers use MultimodalParams.state.
        if state is None and multimodal_params is not None and getattr(multimodal_params, "action", None) is not None:
            action = multimodal_params.action
            state = action.state if hasattr(action, "state") else action["state"]

        states = None if not self.config.is_pi05 else [state]
        token_ids, token_mask = self.policy_tokenizer.tokenize([prompt], states=states)
        text_ids = token_ids[0, token_mask[0]].tolist()

        image_ids = []
        if multimodal_params is not None:
            for image in multimodal_params.images:
                if image.token_id is None or image.token_num is None:
                    raise ValueError("pi0 image cache resources must be allocated before tokenization")
                if image.token_num != self.image_length:
                    raise ValueError(f"invalid pi0 image token count: {image.token_num} != {self.image_length}")
                image_ids.extend(range(image.token_id, image.token_id + image.token_num))
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
