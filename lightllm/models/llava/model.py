import json
import numpy as np
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import MultimodalParams


# Warp of the origal tokenizer
class LlavaTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_token = "<image>"
        # (image_size // patch_size) ** 2: (336 // 14) ** 2
        self.image_length = 576

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None):

        # split prompt by <image>, and merge parts by [pad_id] * 576
        ids_chunks = [self.tokenizer(x).input_ids for x in prompt.split(self.image_token)]
        input_ids = ids_chunks[0]
        image_id = 0

        for ids in ids_chunks[1:]:
            # skip the start token
            if len(ids) > 0 and ids[0] == self.tokenizer.bos_token_id:
                ids = ids[1:]

            token_id = multimodal_params.images[image_id].token_id
            token_num = multimodal_params.images[image_id].token_num
            assert token_num == self.image_length, "invalid token num: {} vs {}!".format(token_num, self.image_length)

            input_ids.extend(range(token_id, token_id + token_num))
            input_ids.extend(ids)
            image_id += 1

        if multimodal_params:
            image_cnt = len(multimodal_params.images)
            assert image_cnt == image_id, "invalid image tag num: {} vs {}!".format(image_cnt, image_id)
        return input_ids

    def __getattr__(self, name):
        if name != 'encode':
            return getattr(self.tokenizer, name)
        return self.encode


class LlavaTpPartModel(LlamaTpPartModel):

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
