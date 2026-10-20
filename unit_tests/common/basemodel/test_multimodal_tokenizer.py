"""Virtual image/audio cache IDs must never reach the text tokenizer."""

import unittest
from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.server.detokenization.decode import decode_token
from lightllm.server.detokenization.decode_req import DecodeReq
from lightllm.server.embed_cache.utils import MULTIMODAL_TOKEN_ID_START


class _MultimodalTokenizer(BaseMultiModalTokenizer):
    def encode(self, prompt, multimodal_params=None, **kwargs):
        return self.tokenizer.encode(prompt, **kwargs)

    def init_imageitem_extral_params(self, img, multi_params, sampling_params):
        raise NotImplementedError

    def init_audioitem_extral_params(self, audio, multi_params, sampling_params):
        raise NotImplementedError

    def get_image_token_length(self, img):
        raise NotImplementedError

    def get_audio_token_length(self, audio):
        raise NotImplementedError


class TestMultimodalDecode(unittest.TestCase):
    def setUp(self):
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        self.text = PreTrainedTokenizerFast(tokenizer_object=backend)
        self.text.add_special_tokens({"additional_special_tokens": ["<image>"]})
        self.text.add_tokens(["added_text"])
        self.tokenizer = _MultimodalTokenizer(self.text)

    def test_virtual_ids_are_filtered_without_mutating_input(self):
        text_ids = self.text.encode("hello world", add_special_tokens=False)
        virtual_ids = [MULTIMODAL_TOKEN_ID_START, 2 ** 31, 2 ** 32, 2 ** 63 - 2]
        for virtual_id in virtual_ids:
            for tokens in ([virtual_id] + text_ids, np.array([virtual_id] + text_ids, dtype=np.int64)):
                with self.subTest(virtual_id=virtual_id, container=type(tokens).__name__):
                    original = list(tokens)
                    self.assertEqual(self.tokenizer.decode(tokens), "hello world")
                    self.assertEqual(list(tokens), original)

    def test_added_and_special_tokens_keep_their_original_behavior(self):
        token_ids = self.text.encode("hello", add_special_tokens=False)
        token_ids += self.text.convert_tokens_to_ids(["<image>", "added_text"])
        for skip_special_tokens in (False, True):
            with self.subTest(skip_special_tokens=skip_special_tokens):
                self.assertEqual(
                    self.tokenizer.decode([2 ** 40] + token_ids, skip_special_tokens=skip_special_tokens),
                    self.text.decode(token_ids, skip_special_tokens=skip_special_tokens),
                )

    def test_scalar_and_empty_inputs(self):
        token_id = self.text.encode("a", add_special_tokens=False)[0]
        for value in (token_id, np.int64(token_id), [token_id], []):
            with self.subTest(value=value):
                self.assertEqual(self.tokenizer.decode(value), self.text.decode(value))
        self.assertEqual(self.tokenizer.decode(2 ** 40), "")
        self.assertEqual(self.tokenizer.decode([2 ** 40, 2 ** 63 - 2]), "")

    def test_five_token_prompt_tail_and_partial_utf8_stream(self):
        prompt_ids = self.text.encode("prompt", add_special_tokens=False) + [2 ** 40 + i for i in range(5)]
        output_ids = self.text.encode("你 world", add_special_tokens=False)
        req = SimpleNamespace(
            request_id=1,
            group_req_id=1,
            input_len=len(prompt_ids),
            shm_prompt_ids=SimpleNamespace(arr=np.array(prompt_ids + output_ids, dtype=np.int64)),
            sample_params=SimpleNamespace(
                print_eos_token=False, skip_special_tokens=True, add_spaces_between_special_tokens=False
            ),
        )
        decode_req = DecodeReq(req)
        pieces = []
        for token_id in output_ids:
            decode_req.output_ids.append(token_id)
            pieces.append(decode_token(self.tokenizer, decode_req, token_id, []))
        self.assertEqual(pieces[:2], ["", ""])
        self.assertEqual("".join(pieces), "你 world")
        self.assertNotIn("\ufffd", "".join(pieces))


if __name__ == "__main__":
    unittest.main()
