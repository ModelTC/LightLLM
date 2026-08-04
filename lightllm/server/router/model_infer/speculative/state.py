from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.speculative.runtime import SpecRuntime


class SpecHiddenStore:
    """Stores target-model features that are passed into draft-model forwards.

    LightLLM's service path does not materialize HuggingFace-style
    `hidden_states`.  Instead, BaseModel calls SpecForwardContext while the
    target model is running.

    Captured tensors are keyed by microbatch index:
    - vanilla MTP consumes the final target hidden state:
      [token_num, hidden_size]
    - Eagle3 / DSpark-style draft models consume selected target layers
      concatenated on the hidden dimension after TP/SP all-gather:
      [token_num, hidden_size * len(target_layer_ids)]

    Draft-model forwards are identified by `mtp_draft_input_hiddens is not
    None`; their final hidden can also be captured because chained MTP drafts
    pass one draft's hidden into the next draft model.
    """

    def __init__(self, runtime: "SpecRuntime") -> None:
        self.runtime = runtime
        self._captured_hiddens = {}

    def select_hidden(self, *, infer_state, hidden: torch.Tensor, final_hidden: torch.Tensor) -> torch.Tensor:
        if self.runtime.is_draft_forward(infer_state):
            return final_hidden
        if self.runtime.needs_intermediate_target_hidden:
            return hidden
        return final_hidden

    def capture_hidden(self, *, infer_state, hidden: torch.Tensor, final_hidden: torch.Tensor) -> torch.Tensor:
        selected_hidden = self.select_hidden(
            infer_state=infer_state,
            hidden=hidden,
            final_hidden=final_hidden,
        )
        self._captured_hiddens[infer_state.microbatch_index] = selected_hidden
        return selected_hidden

    def get_hidden(self, microbatch_index: int = 0) -> torch.Tensor:
        hidden = self._captured_hiddens.get(microbatch_index)
        assert hidden is not None
        return hidden

    def unpad_hidden(self, *, token_num: int, microbatch_index: int = 0) -> None:
        hidden = self._captured_hiddens.get(microbatch_index)
        if hidden is not None and hidden.shape[0] > token_num:
            self._captured_hiddens[microbatch_index] = hidden[0:token_num]
        return

    def export_graph_capture(self):
        captured_hiddens = {
            microbatch_index: tensor_to_no_ref_tensor(hidden)
            for microbatch_index, hidden in self._captured_hiddens.items()
        }
        self._captured_hiddens = dict(captured_hiddens)
        return captured_hiddens

    def restore_graph_capture(self, captured_hiddens) -> None:
        if captured_hiddens is not None:
            self._captured_hiddens = dict(captured_hiddens)
        return


class SpecForwardContext:
    """Per-forward hidden capture context used by BaseModel.

    The target model and the draft model exchange only tensors, not model
    outputs.  During target forward, BaseModel calls `add_hidden` after each
    transformer layer.  The runtime decides which layer ids matter for the
    active draft algorithm.  At the end of forward, BaseModel calls `capture`
    with:
    - `hidden`: selected intermediate target feature, shape
      [token_num, hidden_size * selected_layer_num] after all-gather
    - `final_hidden`: final target feature, shape [token_num, hidden_size]

    Vanilla MTP uses `final_hidden`; Eagle3/DSpark-style proposers use
    `hidden`.
    """

    def __init__(self, *, runtime: "SpecRuntime", model, infer_state) -> None:
        self.runtime = runtime
        self.model = model
        self.infer_state = infer_state
        self.layer_ids = runtime.get_capture_layer_ids(model, infer_state)
        self.layer_hiddens: List[torch.Tensor] = []

    def add_hidden(self, *, layer_index: int, layer_num: int, hidden: torch.Tensor) -> None:
        if layer_index not in self.layer_ids:
            return

        if layer_index == layer_num - 1:
            self.layer_hiddens.append(hidden)
        else:
            self.layer_hiddens.append(hidden.clone())
        return

    def build_layer_hidden(self) -> Optional[torch.Tensor]:
        if not self.layer_hiddens:
            return None
        if len(self.layer_hiddens) == 1:
            return self.layer_hiddens[0]
        return torch.cat(self.layer_hiddens, dim=-1)

    def capture(self, *, hidden: torch.Tensor, final_hidden: torch.Tensor) -> torch.Tensor:
        return self.runtime.capture_hidden(
            infer_state=self.infer_state,
            hidden=hidden,
            final_hidden=final_hidden,
        )

    def capture_final_hidden(self, final_hidden: torch.Tensor) -> None:
        assert not self.layer_ids, (
            f"{self.runtime.spec_config.mode} needs intermediate hidden layers and does not support "
            "microbatch overlap forward now"
        )
        self.capture(hidden=final_hidden, final_hidden=final_hidden)
        return
