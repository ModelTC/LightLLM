import torch
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.embed_cache.utils import bytes2tensor, get_shm_name_embed, read_shm


class Gemma3PreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.embed_scale = torch.tensor(network_config["hidden_size"] ** 0.5, dtype=torch.float32)
        self.boi_token_index: int = 255_999
        self.eoi_token_index: int = 256_000
        return

    def context_forward(self, input_ids, infer_state, layer_weight):
        img_weight = []
        img_start_token_ids = []
        img_token_lens = []
        img_start_loc = 0
        img_start_locs = []
        device = layer_weight.wte_weight_.device
        dtype = layer_weight.wte_weight_.dtype
        hidden_size = layer_weight.wte_weight_.shape[1]
        weight_mask = torch.zeros((len(input_ids)), dtype=torch.float32, device=device)

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
                # pull the img_embeds by uid from shm
                data = read_shm(get_shm_name_embed(img["uuid"]))
                img_weight.append(bytes2tensor(data).cuda().reshape(img["token_num"], -1))
                img_start_token_ids.append(img["token_id"])
                img_token_lens.append(img["token_num"])
                img_start_locs.append(img_start_loc)
                img_start_loc += img["token_num"]
        out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device=device)
        if len(img_weight) > 0:
            img_weight = torch.cat(img_weight, dim=0).to(device=device, dtype=dtype)
        else:
            img_weight = torch.empty((0, hidden_size), device=device, dtype=dtype)
        assert img_weight.shape[1] == hidden_size, (
            f"Dimension mismatch: text weight dimension is {hidden_size}, "
            f"but image weight dimension is {img_weight.shape[1]}"
        )
        # each tp will fill the img embeds, should divide by world_size
        img_weight = img_weight / self.tp_world_size_
        img_start_token_ids = torch.Tensor(img_start_token_ids).to(device=device, dtype=torch.long)
        img_token_lens = torch.Tensor(img_token_lens).to(device=device, dtype=torch.long)
        img_start_locs = torch.Tensor(img_start_locs).to(device=device, dtype=torch.long)

        multimodal_emb(
            out,
            input_ids,
            layer_weight.wte_weight_,
            img_weight,
            img_token_lens,
            img_start_token_ids,
            img_start_locs,
            self.vob_start_id_,
            self.vob_end_id_,
        )
        input_dtype = out.dtype
        if self.tp_world_size_ > 1:
            all_reduce(out, group=infer_state.dist_group, op=torch.dist.ReduceOp.SUM, async_op=False)
        return (out.float() * weight_mask.unsqueeze(1).float()).to(input_dtype)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embedding = super().token_forward(input_ids, infer_state, layer_weight)
        input_dtype = input_embedding.dtype
        return (input_embedding.float() * self.embed_scale.to(input_embedding.device).float()).to(input_dtype)
