import json
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


# Warp of the origal tokenizer
class LlavaTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_token = "<image>"
        # (image_size // patch_size) ** 2: (336 // 14) ** 2
        self.image_length = 576

    # only change the impl of the encode func:
    def encode(self, prompt):

        # split prompt by <image>, and merge parts by [pad_id] * 576
        ids_chunks = [self.tokenizer(x).input_ids for x in prompt.split(self.image_token)]
        input_ids = ids_chunks[0]
        offsets = []
        lengths = []

        for ids in ids_chunks[1:]:
            # skip the start token
            if len(ids) > 0 and ids[0] == self.tokenizer.bos_token_id:
                ids = ids[1:]

            offsets.append(len(input_ids))
            lengths.append(self.image_length)
            input_ids.extend([self.tokenizer.pad_token_id] * self.image_length)
            input_ids.extend(ids)

        return {"input_ids": input_ids, "offsets": offsets, "lengths": lengths}

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
