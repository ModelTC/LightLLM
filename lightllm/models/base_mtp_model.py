from typing import List

from lightllm.common.basemodel.basemodel import TpPartBaseModel


class BaseMTPModel:
    """Shared wiring for MTP draft models: they reuse the main model's req/mem managers and rope
    caches, and pop the main_model / previous-draft-models kwargs before the base __init__ (#25).
    Mixed in BEFORE the concrete base model so these overrides win via MRO.

    Also carries the is_mtp_draft_model marker consumed by detection sites (#23)."""

    is_mtp_draft_model = True

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
        return

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager
        return
