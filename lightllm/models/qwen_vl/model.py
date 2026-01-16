import unicodedata
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.server.core.objs import SamplingParams
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen.model import QWenTpPartModel
from .layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem


# Warp of the origal tokenizer
class QWenVLTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg):
        super().__init__(tokenizer)
        # <img>: 151857
        self.image_start_tag = tokenizer.image_start_tag
        self.image_start_id = tokenizer.img_start_id
        # </img>: 151858
        self.image_end_tag = tokenizer.image_end_tag
        self.image_end_id = tokenizer.img_end_id
        # <imgpad>: 151859
        self.image_length = model_cfg["visual"].get("n_queries", 256)

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def _list_find(self, input_list, target, start_idx):
        cur_list = input_list[start_idx:]
        if target in cur_list:
            return cur_list.index(target) + start_idx
        return -1

    def get_image_token_length(self, img: ImageItem):
        return self.image_length

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    # <img>xxx</img> -> Picture {image_idx}:<img>xxx</img>\n
    def _format_prompt(self, prompt):
        parts = prompt.split(self.image_start_tag)
        prompt = parts[0]
        for idx, part in enumerate(parts[1:]):
            prompt += f"Picture {idx + 1}:" + self.image_start_tag + part
        parts = prompt.split(self.image_end_tag)
        prompt = parts[0]
        for part in parts[1:]:
            prompt += self.image_end_tag + "\n" + part
        return prompt

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None):
        prompt = unicodedata.normalize("NFC", prompt)
        prompt = self._format_prompt(prompt)
        origin_ids = self.tokenizer.tokenizer.encode(prompt, allowed_special="all", disallowed_special=())

        input_ids = []
        image_id = 0
        end = 0
        while True:
            # <img>xxx</img> -> <img><imgpad>*256</img>
            start = self._list_find(origin_ids, self.image_start_id, end)
            if start == -1:
                break
            input_ids.extend(origin_ids[end:start])
            end = self._list_find(origin_ids, self.image_end_id, start)
            if end == -1:
                raise ValueError("Unclosed image token")

            token_id = multimodal_params.images[image_id].token_id
            token_num = multimodal_params.images[image_id].token_num
            assert token_num == self.image_length, "invalid token num: {} vs {}!".format(token_num, self.image_length)

            input_ids.append(self.image_start_id)
            input_ids.extend(range(token_id, token_id + token_num))
            input_ids.append(self.image_end_id)
            end += 1
            image_id += 1

        input_ids.extend(origin_ids[end:])
        if multimodal_params:
            image_cnt = len(multimodal_params.images)
            assert image_cnt == image_id, "invalid image tag num: {} vs {}!".format(image_cnt, image_id)
        return input_ids


@ModelRegistry("qwen", is_multimodal=True, condition=lambda cfg: "visual" in cfg)
class QWenVLTpPartModel(QWenTpPartModel):

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
