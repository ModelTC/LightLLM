from lightllm.models.qwen2_vl.model import QWen2VLTokenizer


class Qwen4ExpTokenizer(QWen2VLTokenizer):
    def count_prompt_tokens(self, text_tokens, image_tokens, image_count, audio_tokens, audio_count):
        return text_tokens + image_tokens + audio_tokens
