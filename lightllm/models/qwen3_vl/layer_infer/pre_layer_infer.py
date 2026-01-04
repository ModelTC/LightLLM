import torch
import torch.distributed as dist
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from ..layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight


class Qwen3VLMultimodalPreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        return

    def context_forward(
        self, input_ids, infer_state: Qwen3VLInferStateInfo, layer_weight: Qwen3VLPreAndPostLayerWeight
    ):
        img_start_token_ids = []
        img_token_lens = []
        img_start_locs_in_cache = []
        device = layer_weight.wte_weight_.weight.device
        dtype = layer_weight.wte_weight_.weight.dtype
        hidden_size = layer_weight.wte_weight_.weight.shape[1]

        for batch_id, p in enumerate(infer_state.multimodal_params):
            for img in p["images"] + p["audios"]:
                # skip the same image
                if img["token_id"] in img_start_token_ids:
                    continue
                img_start_token_ids.append(img["token_id"])
                img_token_lens.append(img["token_num"])
                img_start_locs_in_cache.append(img["start_index_in_embed_cache"])
        out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device=device)

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        cpu_embed_cache_tensor = g_infer_context.cpu_embed_cache_client.cpu_embed_cache_tensor
        infer_state.cpu_embed_cache_tensor = cpu_embed_cache_tensor

        assert cpu_embed_cache_tensor.shape[2] == hidden_size, (
            f"Dimension mismatch: text weight dimension is {hidden_size}, "
            f"but image embed dimension is {cpu_embed_cache_tensor.shape[2]}"
        )
        # each tp will fill the img embeds, should divide by world_size
        infer_state.img_start_token_ids = torch.tensor(
            img_start_token_ids, dtype=torch.long, device="cpu", pin_memory=True
        ).cuda(non_blocking=True)
        infer_state.img_token_lens = torch.tensor(img_token_lens, dtype=torch.long, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        infer_state.img_start_locs_in_cache = torch.tensor(
            img_start_locs_in_cache, dtype=torch.long, device="cpu", pin_memory=True
        ).cuda(non_blocking=True)
        infer_state.input_ids = input_ids

        multimodal_emb(
            out=out,
            prompt_ids=input_ids,
            text_weight_embs=layer_weight.wte_weight_.weight,
            embed_cache=cpu_embed_cache_tensor,
            img_token_lens=infer_state.img_token_lens,
            img_start_token_ids=infer_state.img_start_token_ids,
            img_start_locs_in_cache=infer_state.img_start_locs_in_cache,
            tp_text_start_token_id=layer_weight.wte_weight_.tp_vocab_start_id,
            tp_text_end_token_id=layer_weight.wte_weight_.tp_vocab_end_id,
            tp_world_size=self.tp_world_size_,
        )
        if self.tp_world_size_ > 1:
            all_reduce(out, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        return out
