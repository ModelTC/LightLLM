from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.server.embed_cache.utils import bytes2tensor, read_shm, get_shm_name_embed
import torch

"""
infer_state.multimodal_params: batch list of MultimodalParams-dict like:
   {
       "cache_port": int,
       "images": [
           {
               "uuid": int / torch.Tensor,
               "offset": int,
               "length": int,
           },
           ...
       ]
   }
"""
def prepare_multimodal_embeds(input_embs, infer_state: InferStateInfo):
    if infer_state.multimodal_params is None:
        return

    for batch_id, p in enumerate(infer_state.multimodal_params):
        for img in p["images"]:
            uid, offset, length = img["uuid"], img["offset"], img["length"]
            # pull the img_embeds by uid from cache server
            if isinstance(uid, int):
                embed_data = read_shm(get_shm_name_embed(uid))
                img_embeds = bytes2tensor(embed_data).reshape(img["length"], -1)
            elif isinstance(uid, torch.Tensor):
                img_embeds = uid
            else:
                raise ValueError("type of uuid = {} is not supported!".format(type(uid)))
            img_embeds = img_embeds.to(device=input_embs.device, dtype=input_embs.dtype)

            # check repad shapes and fill the mutimodal embeds into text embeds
            img_l, img_d = img_embeds.shape
            text_d = input_embs.shape[1]
            assert img_l == length and img_d == text_d, "Incorrect shapes: {} vs {}, {} vs {}".format(img_l, length, img_d, text_d)
            sidx = infer_state.b_start_loc[batch_id]
            input_embs[sidx + offset: sidx + offset + length] = img_embeds
            print("repad input_embeds start_idx={} offset={} length={} dim={}".format(sidx, offset, length, img_d))
