from ..conv_ssm_state import ConvSSMState


class KDAState(ConvSSMState):
    """KDA convolution history, recurrent state and accepted speculative cursor."""

    algorithm = "kda"

    def get_snapshot_storage_specs(self):
        assert (
            self.config.num_k_heads == self.config.num_v_heads and self.config.key_dim == self.config.value_dim
        ), "KDA pages require equal Q/K/V heads and dimensions"
        return super().get_snapshot_storage_specs()
