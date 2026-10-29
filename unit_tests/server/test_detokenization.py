"""Incremental detokenization must ignore virtual image/audio cache IDs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lightllm.server.detokenization import decode_req as decode_req_module
from lightllm.server.detokenization.decode import decode_token
from lightllm.server.detokenization.decode_req import DecodeReq
from lightllm.server.embed_cache.utils import MULTIMODAL_TOKEN_ID_START


class TestDetokenization(unittest.TestCase):
    def setUp(self):
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        self.tokenizer.add_tokens(["added_text"])
        prefix_length = patch.object(decode_req_module, "LIGHTLLM_DECODE_PREFIX_LENGTH", 5)
        prefix_length.start()
        self.addCleanup(prefix_length.stop)

    def make_decode_req(self, prompt_ids, output_ids, **sampling_params):
        params = dict(print_eos_token=False, skip_special_tokens=True, add_spaces_between_special_tokens=False)
        params.update(sampling_params)
        req = SimpleNamespace(
            request_id=1,
            group_req_id=1,
            input_len=len(prompt_ids),
            shm_prompt_ids=SimpleNamespace(arr=np.array(prompt_ids + output_ids, dtype=np.int64)),
            sample_params=SimpleNamespace(**params),
        )
        return DecodeReq(req)

    def decode_stream(self, decode_req, output_ids):
        pieces = []
        for token_id in output_ids:
            decode_req.output_ids.append(token_id)
            pieces.append(decode_token(self.tokenizer, decode_req, token_id, []))
        return pieces

    def test_virtual_prompt_ids_at_boundary_and_large_values(self):
        output_ids = self.tokenizer.encode("hello world", add_special_tokens=False)
        for virtual_id in (MULTIMODAL_TOKEN_ID_START, 2 ** 31, 2 ** 32, 2 ** 63 - 2):
            with self.subTest(virtual_id=virtual_id):
                prompt_ids = self.tokenizer.encode("prompt", add_special_tokens=False) + [virtual_id]
                decode_req = self.make_decode_req(prompt_ids, output_ids)
                self.assertEqual("".join(self.decode_stream(decode_req, output_ids)), "hello world")

    def test_five_token_prompt_tail_and_partial_utf8_keep_original_positions(self):
        prompt_ids = self.tokenizer.encode("prompt", add_special_tokens=False) + [2 ** 40 + i for i in range(5)]
        output_ids = self.tokenizer.encode("你 world", add_special_tokens=False)
        decode_req = self.make_decode_req(prompt_ids, output_ids)
        original_tokens = decode_req.req.shm_prompt_ids.arr.copy()
        initial_offsets = (len(prompt_ids) - 5, len(prompt_ids))
        self.assertEqual((decode_req.prefix_offset, decode_req.read_offset), initial_offsets)

        pieces = []
        for index, token_id in enumerate(output_ids):
            shared_token_id, shared_index = decode_req.get_next_token_id_and_index()
            self.assertEqual(shared_token_id, token_id)
            self.assertEqual(shared_index, len(prompt_ids) + index)
            decode_req.output_ids.append(shared_token_id)
            pieces.append(decode_token(self.tokenizer, decode_req, token_id, []))
            if index < 2:
                self.assertEqual((decode_req.prefix_offset, decode_req.read_offset), initial_offsets)
            elif index == 2:
                self.assertEqual(
                    (decode_req.prefix_offset, decode_req.read_offset), (len(prompt_ids), len(prompt_ids) + 3)
                )

        self.assertEqual(pieces[:2], ["", ""])
        self.assertEqual("".join(pieces), "你 world")
        self.assertEqual(decode_req.prompt_ids, prompt_ids)
        self.assertEqual(decode_req.output_ids, output_ids)
        np.testing.assert_array_equal(decode_req.req.shm_prompt_ids.arr, original_tokens)
        self.assertEqual(decode_req.read_offset, len(original_tokens))
        self.assertEqual(decode_req.prefix_offset, len(original_tokens) - 1)

    def test_added_and_special_tokens_keep_tokenizer_options(self):
        output_ids = self.tokenizer.encode("hello", add_special_tokens=False)
        output_ids += self.tokenizer.convert_tokens_to_ids(["<image>", "added_text"])
        for skip_special_tokens in (False, True):
            for add_spaces in (False, True):
                with self.subTest(skip_special_tokens=skip_special_tokens, add_spaces=add_spaces):
                    decode_req = self.make_decode_req(
                        [2 ** 40],
                        output_ids,
                        skip_special_tokens=skip_special_tokens,
                        add_spaces_between_special_tokens=add_spaces,
                    )
                    expected = self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=skip_special_tokens,
                        spaces_between_special_tokens=add_spaces,
                    )
                    self.assertEqual("".join(self.decode_stream(decode_req, output_ids)), expected)

    def test_text_only_prompt(self):
        prompt_ids = self.tokenizer.encode("prompt", add_special_tokens=False)
        output_ids = self.tokenizer.encode("你 world", add_special_tokens=False)
        decode_req = self.make_decode_req(prompt_ids, output_ids)
        self.assertEqual("".join(self.decode_stream(decode_req, output_ids)), "你 world")


if __name__ == "__main__":
    unittest.main()
