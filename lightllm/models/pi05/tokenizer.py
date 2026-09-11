from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from lightllm.models.pi0.preprocessing import Pi0PrePostProcessor
from lightllm.models.pi0.tokenizer import Pi0Tokenizer, Pi0TokenizerAdapter


def discretize_state_256(state: np.ndarray | torch.Tensor | Sequence[float]):
    """Discretize normalized Pi0.5 state with OpenPI's exact 256-bin rule."""

    if isinstance(state, torch.Tensor):
        boundaries = torch.linspace(
            -1.0, 1.0, 257, dtype=torch.float64, device=state.device
        )[:-1]
        return torch.searchsorted(boundaries, state.to(torch.float64), right=True) - 1
    values = np.asarray(state)
    boundaries = np.linspace(-1.0, 1.0, 257)[:-1]
    return np.digitize(values, bins=boundaries) - 1


def format_pi05_prompt(
    prompt: str, state: np.ndarray | torch.Tensor | Sequence[float]
) -> str:
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    bins = discretize_state_256(state)
    if isinstance(bins, torch.Tensor):
        bins = bins.detach().cpu().reshape(-1).tolist()
    else:
        bins = bins.reshape(-1).tolist()
    state_text = " ".join(str(int(value)) for value in bins)
    return f"Task: {cleaned}, State: {state_text};\nAction: "


class Pi05Tokenizer(Pi0Tokenizer):
    """PaliGemma tokenizer for Pi0.5's state-in-prefix prompt."""

    def format_prompt(self, prompt: str, state) -> str:
        return format_pi05_prompt(prompt, state)

    def _sentencepiece_encode(self, text: str) -> list[int]:
        return self.sentencepiece.encode(text, add_bos=True)

    def tokenize(
        self,
        prompts: str | Sequence[str],
        states,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if states is None or len(states) != len(prompts):
            raise ValueError("pi0.5 states must match the prompt batch")
        texts = [
            self.format_prompt(prompt, state)
            for prompt, state in zip(prompts, states, strict=True)
        ]
        return self._encode_texts(texts)


class Pi05TokenizerAdapter(Pi0TokenizerAdapter):
    """Normalize and discretize Pi0.5 state into the VLM prefix."""

    tokenizer_class = Pi05Tokenizer

    def __init__(self, model_dir: str, tokenizer_path: str | None = None):
        super().__init__(model_dir, tokenizer_path)
        self.preprocessor = Pi0PrePostProcessor(self.config)

    def _tokenize_prompt(self, prompt: str, multimodal_params):
        state = (
            None
            if multimodal_params is None
            else getattr(multimodal_params, "state", None)
        )
        # Older action callers stored state only inside the action payload.
        if (
            state is None
            and multimodal_params is not None
            and getattr(multimodal_params, "action", None) is not None
        ):
            action = multimodal_params.action
            state = action.state if hasattr(action, "state") else action["state"]
        if state is None:
            raise ValueError("pi0.5 prompt tokenization requires state")
        action = (
            None
            if multimodal_params is None
            else getattr(multimodal_params, "action", None)
        )
        if isinstance(action, dict):
            domain = action.get("domain")
        else:
            domain = None if action is None else getattr(action, "domain", None)
        normalized_state = self.preprocessor.normalize_state(state, domain=domain)
        return self.policy_tokenizer.tokenize([prompt], states=[normalized_state])
