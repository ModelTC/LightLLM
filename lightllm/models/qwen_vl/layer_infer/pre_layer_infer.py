import rpyc
import torch
import torch.distributed as dist

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo

from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.utils.envs_utils import get_env_start_args, get_cache_port
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.common.image_cache_manager import image_cache_manager
from lightllm.distributed.communication_op import all_reduce
from lightllm.server.embed_cache.utils import bytes2tensor, tensor2bytes, read_shm, create_shm, get_shm_name_embed

"""
infer_state.multimodal_params: batch list of MultimodalParams-dict like:
   {
       "images": [
           {
               "uuid": int,
               "token_id": int, image token start id,
               "token_num": int, image token num,
           },
       ]
       ...
   }
"""


class LlamaMultimodalPreLayerInfer(LlamaPreLayerInfer):
    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        return

    def _infer_image_embeds(self, infer_state, layer_weight):
        if layer_weight.visual_model is None:
            return
        image_weight = []
        for batch_id, p in enumerate(infer_state.multimodal_params):
            for uuid, image_data, token_num in p["image_data"]:
                if image_cache_manager.query_embed(uuid):
                    image_embed = image_cache_manager.get_embed(uuid)
                else:
                    image_data = image_data.to("cuda", non_blocking=True)
                    image_embed = layer_weight.visual_model.forward(image_data).view(token_num, -1)
                    image_cache_manager.set_embed(uuid, image_embed)
                image_weight.append(image_embed)
        if len(image_weight) > 0:
            image_weight = torch.cat(image_weight, dim=0)
            image_weight = image_weight / self.tp_world_size_
            assert image_weight.shape[1] == layer_weight.wte_weight_.shape[1], (
                f"Dimension mismatch: text weight dimension is {layer_weight.wte_weight_.shape[1]}, "
                f"but image weight dimension is {image_weight.shape[1]}"
            )
        else:
            hidden_size = layer_weight.wte_weight_.shape[1]
            image_weight = torch.empty((0, hidden_size), device="cpu", dtype=layer_weight.wte_weight_.dtype).to(
                "cuda", non_blocking=True
            )
        return image_weight

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        dtype = layer_weight.wte_weight_.dtype
        hidden_size = layer_weight.wte_weight_.shape[1]
        img_weight = self._infer_image_embeds(infer_state, layer_weight)

        # infer_state.mark_multimodal_objs_for_prefill(input_ids=input_ids)

        # self._infer_image_embeds(infer_state, layer_weight)
        # for batch_id, p in enumerate(infer_state.multimodal_params):
        #     for img in p["images"] + p["audios"]:
        #         # skip the same image
        #         if img["token_id"] in img_start_token_ids or img["_prefill_"] is False:
        #             continue
        #         # pull the img_embeds by uid from shm
        #         data = read_shm(get_shm_name_embed(img["uuid"]))
        #         img_weight.append(bytes2tensor(data).cuda().reshape(img["token_num"], -1))
        #         img_start_token_ids.append(img["token_id"])
        #         img_token_lens.append(img["token_num"])
        #         img_start_locs.append(img_start_loc)
        #         img_start_loc += img["token_num"]
        # out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device=device)
        # if len(img_weight) > 0:
        #     img_weight = torch.cat(img_weight, dim=0).to(device=device, dtype=dtype)
        # else:
        #     img_weight = torch.empty((0, hidden_size), device=device, dtype=dtype)
        # assert img_weight.shape[1] == hidden_size, (
        #     f"Dimension mismatch: text weight dimension is {hidden_size}, "
        #     f"but image weight dimension is {img_weight.shape[1]}"
        # )
        # each tp will fill the img embeds, should divide by world_size
        out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device="cpu").to("cuda", non_blocking=True)
        img_weight = img_weight / self.tp_world_size_
        if hasattr(infer_state, "image_start_token_ids"):
            img_start_token_ids = infer_state.image_start_token_ids.to("cuda", non_blocking=True)
            img_token_lens = infer_state.image_token_lens.to("cuda", non_blocking=True)
            img_start_locs = infer_state.image_start_locs.to("cuda", non_blocking=True)
        else:
            img_start_token_ids = torch.empty((0,), device="cpu", dtype=torch.long).to("cuda", non_blocking=True)
            img_token_lens = torch.empty((0,), device="cpu", dtype=torch.long).to("cuda", non_blocking=True)
            img_start_locs = torch.empty((0,), device="cpu", dtype=torch.long).to("cuda", non_blocking=True)

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
