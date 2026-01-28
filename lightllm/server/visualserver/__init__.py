from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend
from lightllm.common.basemodel.attention_vit.create_utils import get_vit_att_backend_class

VIT_ATTN_BACKEND: BaseVitAttBackend = None


def set_vit_att_backend(backend_name: str):
    global VIT_ATTN_BACKEND
    VIT_ATTN_BACKEND = get_vit_att_backend_class(backend_name)
    return


def get_vit_attn_backend():
    if VIT_ATTN_BACKEND is None:
        raise RuntimeError("VIT_ATTN_BACKEND is not initialized. Call init_vit_att_backend() first.")
    return VIT_ATTN_BACKEND._vit_att_fwd
