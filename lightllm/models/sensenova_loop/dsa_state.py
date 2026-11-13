"""One replicated, unquantized Indexer K per token and execution layer."""

from lightllm.common.attention_state.full_att import MLAFullAttentionState
from lightllm.common.attention_state.utils import dtype_name


class DsaIndexerState(MLAFullAttentionState):
    """Reuse lossless latent-state CPU/DP/PD transport for the shared Indexer K.

    This is a separate resource from ordinary GQA K/V. The old cache repeated
    the same Indexer K in every K-head tail and reserved an unused V-head tail.
    Here standard GQA attention sees its original 256-dimensional K/V, while
    the Indexer scans contiguous 128-dimensional keys. Unequal TP2/TP4 roles
    exchange the replicated indexer field through the existing state manager.
    """

    def get_state_type(self):
        return f"dsa_indexer_{dtype_name(self.config.latent_dtype)}_d{self.config.latent_dim}"
