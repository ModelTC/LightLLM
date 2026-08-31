from typing import ClassVar, Optional

from lightllm.common.basemodel.hidden_collector import (
    FinalHiddenCollector,
    HiddenCollector,
    LayerHiddenCollector,
    MtpHeadOutputCollector,
    NoopHiddenCollector,
)
from lightllm.utils.envs_utils import get_env_start_args


class MtpManager:
    """Manage MTP layout policy and model-local helper construction."""

    _instance: ClassVar[Optional["MtpManager"]] = None
    _CHAINED_DRAFT_MODES = ("vanilla_with_att", "vanilla_no_att")
    _RECURRENT_DRAFT_MODES = ("eagle_with_att", "eagle_no_att", "eagle3")
    _RECURRENT_ATTN_DRAFT_MODES = ("eagle_with_att", "eagle3")
    _BLOCK_DRAFT_MODES = ("dspark", "dflash")

    @classmethod
    def get_instance(cls) -> "MtpManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.args = get_env_start_args()

    def get_decode_batch_multiplier(self, is_draft_model: bool) -> int:
        """Return the physical decode rows used by one logical request."""

        spec_mode = self.args.mtp_mode
        if spec_mode is None:
            return 1

        verify_width = self.args.mtp_step + 1

        # The main model verifies one target token plus mtp_step draft tokens
        # for every logical request, regardless of how the draft is produced.
        if not is_draft_model:
            return verify_width

        # Chained MTP runs every draft module over the expanded verify layout,
        # but each physical row is still an independent one-token draft decode
        # from the attention backend's point of view.  CUDA Graph sizing must
        # account for the wider physical batch separately (see below).
        if spec_mode in self._CHAINED_DRAFT_MODES:
            return 1

        # Recurrent EAGLE draft models decode one row per logical request.
        if spec_mode in self._RECURRENT_DRAFT_MODES:
            return 1

        # Block draft models decode mtp_step rows per logical request.
        if spec_mode in self._BLOCK_DRAFT_MODES:
            return self.args.mtp_step

        return 1

    def get_decode_cuda_graph_batch_multiplier(self, is_draft_model: bool) -> int:
        """Return physical decode rows per logical request for graph sizing.

        Chained draft models keep normal one-token attention semantics, while
        their proposer forwards the complete target verification layout through
        every draft depth.  Consequently a logical batch of ``N`` requests has
        ``N * (mtp_step + 1)`` physical rows and needs graphs captured at that
        width even though :meth:`get_decode_batch_multiplier` returns one.
        """

        # Attention-backed recurrent EAGLE normally decodes one row per
        # request, but its first forward after every target verification runs
        # over the complete expanded verify layout to commit draft KV.  That
        # extend forward must be graph-covered as well.
        if is_draft_model and self.args.mtp_mode in (
            *self._CHAINED_DRAFT_MODES,
            *self._RECURRENT_ATTN_DRAFT_MODES,
        ):
            return self.args.mtp_step + 1
        return self.get_decode_batch_multiplier(is_draft_model=is_draft_model)

    def draft_model_needs_logical_batch_graphs(self, is_draft_model: bool) -> bool:
        """Whether a widened draft graph also needs one-row-per-request sizes."""

        return is_draft_model and self.args.mtp_mode in self._RECURRENT_ATTN_DRAFT_MODES

    def get_decode_cuda_graph_grow_step_size(self, is_draft_model: bool) -> int:
        """Return the batch-size stride used to capture decode CUDA Graphs."""

        # Draft model CUDA Graphs follow the drafter's physical forward layout,
        # which is wider than its attention semantics for chained MTP.
        if is_draft_model:
            return self.get_decode_cuda_graph_batch_multiplier(is_draft_model=True)
        # Main model CUDA Graphs use unit growth for dynamically compacted verify rows.
        else:
            if self.args.mtp_dynamic_verify:
                return 1
            return self.get_decode_batch_multiplier(is_draft_model=False)

    def get_decode_draft_step(self, is_draft_model: bool) -> int:
        """Return the number of extra decode rows processed per request."""

        return self.get_decode_batch_multiplier(is_draft_model) - 1

    def create_hidden_collector(
        self,
        model,
    ) -> HiddenCollector:
        """Create a model-local hidden-collector prototype for the configured MTP mode."""

        spec_mode = self.args.mtp_mode
        collector_kwargs = {}
        if spec_mode is None:
            collector_type = NoopHiddenCollector
        elif model.is_mtp_draft_model:
            if spec_mode == "dspark":
                collector_type = MtpHeadOutputCollector
            elif spec_mode in self._BLOCK_DRAFT_MODES:
                collector_type = NoopHiddenCollector
            else:
                collector_type = FinalHiddenCollector
        elif spec_mode in ("eagle3", *self._BLOCK_DRAFT_MODES):
            collector_type = LayerHiddenCollector
            collector_kwargs.update(model=model)
        else:
            collector_type = FinalHiddenCollector

        return collector_type(**collector_kwargs)
