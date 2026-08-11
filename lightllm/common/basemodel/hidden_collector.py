from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from transformers.configuration_utils import PretrainedConfig

from lightllm.utils.envs_utils import get_env_start_args


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

    def __init__(self, model, layer_ids: Optional[Iterable[int]] = None) -> None:
        self.model = model
        self.layer_num = model.layers_num
        self.layer_ids = self._resolve_layer_ids(layer_ids)
        self.layer_hiddens: List[torch.Tensor] = []

    def _resolve_layer_ids(self, layer_ids: Optional[Iterable[int]]) -> frozenset[int]:
        if layer_ids is None:
            draft_model_dirs = get_env_start_args().mtp_draft_model_dir
            assert draft_model_dirs
            draft_config, _ = PretrainedConfig.get_config_dict(draft_model_dirs[0])
            layer_ids = draft_config.get("target_layer_ids")
            if layer_ids is None:
                layer_ids = draft_config.get("dflash_config", {}).get("target_layer_ids")
            if layer_ids is None:
                layer_ids = [1, self.layer_num // 2 - 1, self.layer_num - 4]

        resolved_layer_ids = frozenset(int(layer_id) for layer_id in layer_ids)
        assert resolved_layer_ids and all(
            0 <= layer_id < self.layer_num for layer_id in resolved_layer_ids
        ), f"invalid target_layer_ids={resolved_layer_ids} for target layer_num={self.layer_num}"
        return resolved_layer_ids

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
        if spec_mode is not None:
            assert model is not None

        collector_kwargs = {}
        if spec_mode is None:
            collector_type = NoopHiddenCollector
        elif model.is_mtp_draft_model:
            collector_type = NoopHiddenCollector if spec_mode in ("dspark", "dflash") else FinalHiddenCollector
        elif spec_mode not in ("eagle3", "dspark", "dflash"):
            collector_type = FinalHiddenCollector
        else:
            collector_type = LayerHiddenCollector
            collector_kwargs = {"model": model, "layer_ids": layer_ids}

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
