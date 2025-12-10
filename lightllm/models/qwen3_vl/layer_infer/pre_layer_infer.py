import torch
import torch.distributed as dist

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo

from lightllm.server.embed_cache.utils import (
    bytes2tensor,
    read_shm,
    get_shm_name_embed,
)
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce

from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Qwen3VLMultimodalPreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        return

    def context_forward(self, input_ids, infer_state: Qwen3VLInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        img_weight = []
        img_start_loc = 0

        infer_state.input_ids = input_ids
        infer_state.img_start_token_ids = []
        infer_state.img_token_lens = []
        infer_state.img_start_locs = []

        device = layer_weight.wte_weight_.device
        dtype = layer_weight.wte_weight_.dtype
        hidden_size = layer_weight.wte_weight_.shape[1]

        infer_state.mark_multimodal_objs_for_prefill(input_ids=input_ids)

        for batch_id, p in enumerate(infer_state.multimodal_params):
            for img in p["images"] + p["audios"]:
                # skip the same image
                if img["token_id"] in infer_state.img_start_token_ids or img["_prefill_"] is False:
                    continue

                all_img_embed_df = bytes2tensor(read_shm(get_shm_name_embed(img["uuid"])))
                per_image_deepstack = []

                deepstack_layer_num = all_img_embed_df.shape[0] // img["token_num"] - 1
                img_weight.append(all_img_embed_df[: img["token_num"]].cuda())

                for layer in range(deepstack_layer_num):
                    start = img["token_num"] * (layer + 1)
                    end = img["token_num"] * (layer + 2)
                    per_image_deepstack.append(all_img_embed_df[start:end])

                infer_state.deepstack_features.append(per_image_deepstack)
                infer_state.img_start_token_ids.append(img["token_id"])
                infer_state.img_token_lens.append(img["token_num"])
                infer_state.img_start_locs.append(img_start_loc)
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
        img_start_token_ids = torch.Tensor(infer_state.img_start_token_ids).to(device=device, dtype=torch.long)
        img_token_lens = torch.Tensor(infer_state.img_token_lens).to(device=device, dtype=torch.long)
        img_start_locs = torch.Tensor(infer_state.img_start_locs).to(device=device, dtype=torch.long)

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
        if self.tp_world_size_ > 1:
            all_reduce(out, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        return out
