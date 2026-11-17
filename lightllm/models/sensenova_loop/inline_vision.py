"""Serialized ViT execution in the single-GPU LLM worker."""

import rpyc
import torch
from rpyc.utils.classic import obtain

from lightllm.common.basemodel.attention_vit.create_utils import init_vit_att_backend
from lightllm.models.qwen3_vl.qwen3_visual import Qwen3VisionTransformerPretrainedModel
from lightllm.server.multimodal_params import ImageItem
from lightllm.server.visualserver import set_vit_att_backend
from lightllm.utils.shm_port_args import get_shm_port_args


class InlineLoopVision:
    def __init__(self, weight_dir, vision_config, args):
        set_vit_att_backend(init_vit_att_backend(index=0))
        self.model = (
            Qwen3VisionTransformerPretrainedModel({"data_type": args.data_type}, **vision_config).eval().bfloat16()
        )
        self.model.load_model(weight_dir)
        self.model.cuda()
        self.cache = rpyc.connect("localhost", get_shm_port_args().cache_port, config={"allow_pickle": True})

    @torch.no_grad()
    def ensure_embeddings(self, multimodal_params):
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        # Reuse the normal cache's lifetime and ready bits; repeated chunks and
        # subsequent requests do not encode an image again while it is cached.
        images = {image["uuid"]: image for params in multimodal_params for image in params["images"]}
        if not images:
            return
        ready = obtain(self.cache.root.get_items_embed(list(images)))
        for image, is_ready in zip(images.values(), ready):
            if is_ready:
                continue
            item = ImageItem(type="base64", data="")
            for key, value in image.items():
                setattr(item, key, value)
            embeddings, _, ranges = self.model.encode([item])
            start, end = ranges[0]
            if end - start != item.token_num:
                raise ValueError("Inline ViT output length does not match the request's image tokens")
            g_infer_context.cpu_embed_cache_client.copy_vision_to_cache(
                embeddings[start:end], item.start_index_in_embed_cache
            )
            # Publish ready only after the D2H copy has completed. The following
            # prefill consumes this host cache on the same worker's CUDA stream.
            torch.cuda.current_stream().synchronize()
            self.cache.root.set_items_embed([item.uuid])
