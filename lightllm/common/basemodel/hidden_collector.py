from __future__ import annotations

from typing import Iterable, List, Optional

import torch


def unpad_collected_hidden(hidden: Optional[torch.Tensor], token_count: int) -> Optional[torch.Tensor]:
    return None if hidden is None else hidden[:token_count]


class NoopHiddenCollector:
    """Null object used by models that do not expose speculative features."""

    def add(self, layer_index: int, hidden: torch.Tensor) -> None:
        return

    def prefill_outputs(self, final_hidden: torch.Tensor) -> List[torch.Tensor]:
        return [final_hidden]

    def finish(
        self,
        infer_state,
        final_hidden: torch.Tensor,
        forward_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        return None


class FinalHiddenCollector(NoopHiddenCollector):
    """Returns the final decoder hidden state without per-layer bookkeeping."""

    def finish(
        self,
        infer_state,
        final_hidden: torch.Tensor,
        forward_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        return final_hidden.contiguous()


class LayerHiddenCollector(NoopHiddenCollector):
    """Collects selected decoder-layer outputs for an intermediate-hidden draft."""

    def __init__(self, model, layer_ids: Iterable[int]) -> None:
        self.model = model
        self.layer_num = model.layers_num
        self.layer_ids = frozenset(int(layer_id) for layer_id in layer_ids)
        assert self.layer_ids, "layer hidden collector requires at least one layer id"
        self.layer_hiddens: List[torch.Tensor] = []

    def add(self, layer_index: int, hidden: torch.Tensor) -> None:
        if layer_index not in self.layer_ids:
            return
        # Most LightLLM layers reuse their input buffer. Preserve intermediate
        # layers while allowing the final layer output to remain zero-copy.
        self.layer_hiddens.append(hidden if layer_index == self.layer_num - 1 else hidden.clone())

    def _local_hidden(self) -> torch.Tensor:
        assert len(self.layer_hiddens) == len(
            self.layer_ids
        ), f"captured {len(self.layer_hiddens)} hidden layers, expected {len(self.layer_ids)}"
        if len(self.layer_hiddens) == 1:
            return self.layer_hiddens[0]
        return torch.cat(self.layer_hiddens, dim=-1)

    def prefill_outputs(self, final_hidden: torch.Tensor) -> List[torch.Tensor]:
        return [final_hidden, self._local_hidden()]

    def finish(
        self,
        infer_state,
        final_hidden: torch.Tensor,
        forward_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        local_hidden = self._local_hidden() if forward_outputs is None else forward_outputs[1]
        self.layer_hiddens.clear()
        hidden = self.model.pre_infer._tpsp_allgather(input=local_hidden, infer_state=infer_state)
        if infer_state.need_dp_prefill_balance:
            hidden = infer_state._all_to_all_unbalance_get(data=hidden)
        return hidden.contiguous()


class HiddenCollector:
    """Collect hidden states for one or more independently executed microbatches."""

    def __init__(
        self,
        model=None,
        spec_mode: Optional[str] = None,
        layer_ids: Optional[Iterable[int]] = None,
        microbatch_count: int = 1,
    ) -> None:
        assert microbatch_count > 0
        layer_ids = None if layer_ids is None else tuple(layer_ids)

        collector_type = NoopHiddenCollector
        collector_kwargs = {}
        if spec_mode is not None:
            assert model is not None
            if model.is_mtp_draft_model:
                if spec_mode not in ("dspark", "dflash"):
                    collector_type = FinalHiddenCollector
            elif spec_mode in ("eagle3", "dspark", "dflash"):
                assert layer_ids is not None
                collector_type = LayerHiddenCollector
                collector_kwargs = {"model": model, "layer_ids": layer_ids}
            else:
                collector_type = FinalHiddenCollector

        self.collectors = tuple(collector_type(**collector_kwargs) for _ in range(microbatch_count))

    def add(self, layer_index: int, hidden: torch.Tensor, microbatch_index: int = 0) -> None:
        self.collectors[microbatch_index].add(layer_index=layer_index, hidden=hidden)

    def finish(
        self,
        infer_state,
        final_hidden: torch.Tensor,
        forward_outputs: Optional[List[torch.Tensor]] = None,
        microbatch_index: int = 0,
    ) -> Optional[torch.Tensor]:
        return self.collectors[microbatch_index].finish(
            infer_state=infer_state,
            final_hidden=final_hidden,
            forward_outputs=forward_outputs,
        )

    def prefill_outputs(self, final_hidden: torch.Tensor, microbatch_index: int = 0) -> List[torch.Tensor]:
        return self.collectors[microbatch_index].prefill_outputs(final_hidden)
