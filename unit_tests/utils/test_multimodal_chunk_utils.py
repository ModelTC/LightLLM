import random
import unittest

from lightllm.utils.multimodal_chunk_utils import adjust_mm_chunk_end, clamp_mm_prefix_len, get_mm_input_spans


class MultimodalChunkTests(unittest.TestCase):
    def test_explicit_block_includes_tokenizer_padding(self):
        params = {
            "images": [{"block_start_idx": 2, "block_end_idx": 8, "start_idx": 4, "token_num": 8}],
        }
        self.assertEqual(get_mm_input_spans(list(range(10)), params), [(2, 8)])

    def test_embedding_position_locates_a_qwen_image(self):
        params = {"images": [{"start_idx": 3, "token_num": 4}]}
        self.assertEqual(get_mm_input_spans(list(range(9)), params), [(3, 7)])

    def test_adjacent_images_audio_and_reused_embeddings_remain_separate(self):
        ids = [1, 100, 101, 102, 200, 201, 100, 101, 102, 2]
        params = {
            "images": [{"token_id": 100, "token_num": 3}],
            "audios": [{"token_id": 200, "token_num": 2}],
        }
        self.assertEqual(get_mm_input_spans(ids, params), [(1, 4), (4, 6), (6, 9)])

    def test_rejects_partial_virtual_token_blocks(self):
        params = {"audios": [{"token_id": 100, "token_num": 3}]}
        for ids in ([1, 100, 101], [1, 101, 102], [100, 102, 101]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "incomplete token block"):
                get_mm_input_spans(ids, params)

    def test_missing_items_are_rejected_before_scheduling(self):
        params = {"audios": [{"token_id": 100, "token_num": 3}]}
        with self.assertRaisesRegex(ValueError, "missing token block"):
            get_mm_input_spans([1, 2, 3], params)

    def test_rejects_invalid_metadata_and_blocks_over_batch_budget(self):
        for spans in ([(2, 12)], [(2, 7), (6, 8)], [(5, 2)]):
            params = {"images": [{"block_start_idx": start, "block_end_idx": end} for start, end in spans]}
            with self.subTest(spans=spans), self.assertRaisesRegex(ValueError, "invalid or overlapping"):
                get_mm_input_spans(list(range(10)), params)
        params = {"images": [{"start_idx": 2, "token_num": 6}]}
        with self.assertRaisesRegex(ValueError, "--batch_max_tokens=4"):
            get_mm_input_spans(list(range(10)), params, max_tokens=4)

    def test_text_before_and_after_image_can_still_be_chunked(self):
        spans = [(1800, 2100)]
        chunks = []
        start = 0
        while start < 4000:
            end = adjust_mm_chunk_end(start, min(start + 2048, 4000), spans, 2048)
            chunks.append((start, end))
            start = end
        self.assertEqual(chunks, [(0, 1800), (1800, 3848), (3848, 4000)])

    def test_small_chunk_target_can_yield_to_a_complete_item(self):
        spans = [(100, 800)]
        self.assertEqual(adjust_mm_chunk_end(0, 256, spans, 1024), 100)
        self.assertEqual(adjust_mm_chunk_end(100, 356, spans, 1024), 800)
        with self.assertRaisesRegex(ValueError, "--batch_max_tokens=512"):
            adjust_mm_chunk_end(100, 356, spans, 512)

    def test_cached_prefix_must_not_start_a_chunk_inside_an_item(self):
        with self.assertRaisesRegex(ValueError, "Cached prefix ends inside"):
            adjust_mm_chunk_end(250, 1000, [(100, 800)], 1024)

    def test_checkpoint_boundary_can_be_crossed_without_splitting_an_item(self):
        spans = [(1900, 2200)]
        self.assertEqual(adjust_mm_chunk_end(0, 2048, spans, 4096), 1900)
        self.assertEqual(adjust_mm_chunk_end(1900, 2048, spans, 4096), 2200)

    def test_prefix_alignment_can_retreat_into_an_earlier_item(self):
        spans = [(450, 600), (700, 900)]
        self.assertEqual(clamp_mm_prefix_len(768, spans, 256), 256)
        self.assertEqual(clamp_mm_prefix_len(768, spans), 700)
        for boundary in (450, 600, 700, 900):
            self.assertEqual(clamp_mm_prefix_len(boundary, spans), boundary)

    def test_random_chunk_partitions_always_make_progress_within_budget(self):
        rng = random.Random(0)
        for _ in range(100):
            spans = []
            position = 0
            for _ in range(8):
                start = position + rng.randrange(0, 50)
                position = start + rng.randrange(1, 129)
                spans.append((start, position))
            total = position + 50
            target = rng.randrange(1, 129)
            position = 0
            while position < total:
                end = adjust_mm_chunk_end(position, min(position + target, total), spans, 128)
                self.assertLess(position, end)
                self.assertLessEqual(end - position, 128)
                self.assertFalse(any(start < end < stop for start, stop in spans))
                position = end


if __name__ == "__main__":
    unittest.main()
