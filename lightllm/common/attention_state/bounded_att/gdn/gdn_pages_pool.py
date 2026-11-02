from ..conv_ssm_pages_pool import ConvSSMPagesPool


class GDNPagesPool(ConvSSMPagesPool):
    """GDN pages with separate Q/K and V dimensions; allocators own the page slots."""
