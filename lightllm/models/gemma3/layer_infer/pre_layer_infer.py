import torch
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Gemma3PreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.embed_scale = torch.tensor(network_config["hidden_size"] ** 0.5, dtype=torch.float32)
        self.boi_token_index: int = 255_999
        self.eoi_token_index: int = 256_000
        return

    def context_forward(self, input_ids, infer_state, layer_weight):
        img_start_token_ids = []
        img_token_lens = []
        img_start_locs_in_cache = []
        device = layer_weight.wte_weight_.device
        dtype = layer_weight.wte_weight_.dtype
        hidden_size = layer_weight.wte_weight_.shape[1]
        weight_mask = torch.zeros((len(input_ids)), dtype=torch.float32, device=device)

        # TODO
        scale = self.embed_scale
        for idx, input_id in enumerate(input_ids):
            if input_id == self.boi_token_index:
                weight_mask[idx] = scale
                scale = 1.0
            elif input_id == self.eoi_token_index:
                scale = self.embed_scale
                weight_mask[idx] = scale
            else:
                weight_mask[idx] = scale

        for batch_id, p in enumerate(infer_state.multimodal_params):
            for img in p["images"]:
                # skip the same image
                if img["token_id"] in img_start_token_ids:
                    continue
                img_start_token_ids.append(img["token_id"])
                img_token_lens.append(img["token_num"])
                img_start_locs_in_cache.append(img["start_index_in_embed_cache"])
        out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device=device)

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        cpu_embed_cache_tensor = g_infer_context.cpu_embed_cache_client.cpu_embed_cache_tensor

        assert cpu_embed_cache_tensor.shape[2] == hidden_size, (
            f"Dimension mismatch: text weight dimension is {hidden_size}, "
            f"but image embed dimension is {cpu_embed_cache_tensor.shape[2]}"
        )
        # each tp will fill the img embeds, should divide by world_size
        img_start_token_ids = torch.tensor(img_start_token_ids, dtype=torch.long, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        img_token_lens = torch.tensor(img_token_lens, dtype=torch.long, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        img_start_locs_in_cache = torch.tensor(
            img_start_locs_in_cache, dtype=torch.long, device="cpu", pin_memory=True
        ).cuda(non_blocking=True)

        multimodal_emb(
            out=out,
            prompt_ids=input_ids,
            text_weight_embs=layer_weight.wte_weight_,
            embed_cache=cpu_embed_cache_tensor,
            img_token_lens=img_token_lens,
            img_start_token_ids=img_start_token_ids,
            img_start_locs_in_cache=img_start_locs_in_cache,
            tp_text_start_token_id=self.vob_start_id_,
            tp_text_end_token_id=self.vob_end_id_,
            tp_world_size=self.tp_world_size_,
        )
        input_dtype = out.dtype
        if self.tp_world_size_ > 1:
            all_reduce(out, group=infer_state.dist_group, op=torch.dist.ReduceOp.SUM, async_op=False)
        return (out.float() * weight_mask.unsqueeze(1).float()).to(input_dtype)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embedding = super().token_forward(input_ids, infer_state, layer_weight)
        input_dtype = input_embedding.dtype
        return (input_embedding.float() * self.embed_scale.to(input_embedding.device).float()).to(input_dtype)
