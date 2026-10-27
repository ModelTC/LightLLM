import random
import unittest

from lightllm.utils.multimodal_chunk_utils import (
    adjust_mm_chunk_end,
    get_mm_input_spans,
)


class MultimodalChunkTests(unittest.TestCase):
    def test_embedding_position_locates_a_qwen_image(self):
        params = {"images": [{"start_idx": 3, "token_id": 100, "token_num": 4}], "audios": []}
        input_ids = [7, 8, 9, 100, 101, 102, 103]
        self.assertEqual(get_mm_input_spans(input_ids, params), [(3, 7)])

    def test_adjacent_images_audio_and_reused_embeddings_remain_separate(self):
        params = {
            "images": [
                {"start_idx": 1, "token_id": 100, "token_num": 3},
                {"start_idx": 6, "token_id": 100, "token_num": 3},
            ],
            "audios": [{"start_idx": 4, "token_id": 200, "token_num": 2}],
        }
        input_ids = [7, 100, 101, 102, 200, 201, 100, 101, 102]
        self.assertCountEqual(get_mm_input_spans(input_ids, params), [(1, 4), (4, 6), (6, 9)])

    def test_text_before_and_after_image_can_still_be_chunked(self):
        spans = [(1800, 2100)]
        chunks = []
        start = 0
        while start < 4000:
            end = adjust_mm_chunk_end(start, min(start + 2048, 4000), spans)
            chunks.append((start, end))
            start = end
        self.assertEqual(chunks, [(0, 1800), (1800, 3848), (3848, 4000)])

    def test_small_chunk_target_can_yield_to_a_complete_item(self):
        spans = [(100, 800)]
        self.assertEqual(adjust_mm_chunk_end(0, 256, spans), 100)
        self.assertEqual(adjust_mm_chunk_end(100, 356, spans), 800)

    def test_checkpoint_boundary_can_be_crossed_without_splitting_an_item(self):
        spans = [(1900, 2200)]
        self.assertEqual(adjust_mm_chunk_end(0, 2048, spans), 1900)
        self.assertEqual(adjust_mm_chunk_end(1900, 2048, spans), 2200)

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
                end = adjust_mm_chunk_end(position, min(position + target, total), spans)
                self.assertLess(position, end)
                self.assertLessEqual(end - position, 128)
                self.assertFalse(any(start < end < stop for start, stop in spans))
                position = end


if __name__ == "__main__":
    unittest.main()
