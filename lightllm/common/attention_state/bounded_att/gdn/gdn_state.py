from ..conv_ssm_state import ConvSSMState


class GDNState(ConvSSMState):
    """Gated DeltaNet convolution history, recurrent state and accepted speculative cursor."""

    algorithm = "gdn"
