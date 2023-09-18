import torch
import torch.distributed as dist

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama_multimodal.infer_struct import LlamaMultiModalInferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer


"""
infer_state.kwargs['repad_embeds'] = [(embeds, offset), ...]
    embeds: torch.Tensor or None
    offset: int
"""
def repad_input_embeds(input_embeds, infer_state: LlamaMultiModalInferStateInfo):
    assert isinstance(infer_state, LlamaMultiModalInferStateInfo)

    if infer_state.kwargs and 'repad_embeds' in infer_state.kwargs:
        repad_embeds = infer_state.kwargs['repad_embeds']
        assert len(repad_embeds) == infer_state.batch_size, "length of repad_embeds != batch_size: {} vs {}!".format(len(repad_embeds), infer_state.batch_size)

        for i, (embeds, offset) in enumerate(repad_embeds):
            # no need to repad if not given repad embeds
            if embeds is None:
                continue
            assert isinstance(embeds, torch.Tensor), "given reapd embeds should be torch.Tensor but got {}!".format(type(embeds))

            start_idx = infer_state.b_start_loc[i]
            seq_len = infer_state.b_seq_len[i]
            pad_len, pad_dim = embeds.shape
            dim = input_embeds.shape[1]
            assert pad_dim == dim, "invalid pad_dim={}, input_embed_dim={}!".format(pad_dim, dim)
            assert offset + pad_len <= seq_len, "invalid seq_len={}, offset={}, pad_len={}!".format(seq_len, offset, pad_len)
            input_embeds[start_idx + offset: start_idx + offset + pad_len] = embeds
            print("repad input_embeds start_idx={} offset={} pad_len={}".format(start_idx, offset, pad_len))
    return input_embeds


class LlamaMultiModalPreLayerInfer(LlamaPreLayerInfer):

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        return

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: LlamaMultiModalInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embeds = super().context_forward(input_ids, infer_state, layer_weight)
        return repad_input_embeds(input_embeds, infer_state)


    def token_forward(self, input_ids, infer_state: LlamaMultiModalInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embeds = super().token_forward(input_ids, infer_state, layer_weight)
        return repad_input_embeds(input_embeds, infer_state)
