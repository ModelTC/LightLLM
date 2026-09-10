import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from lightllm.server.tokenizer import get_tokenizer


def test_transformers5_checkpoint_on_existing_image(tmp_path):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "<|user|>": 2, "<|assistant|>": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "extra_special_tokens": ["<|user|>", "<|assistant|>"],
                "unk_token": "[UNK]",
            }
        )
    )
    (tmp_path / "chat_template.jinja").write_text("<|user|>{{ messages[0]['content'] }}<|assistant|>")
    tokenizer = get_tokenizer(str(tmp_path))
    assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=True) == [2, 1, 3]
    assert tokenizer.decode([2, 1, 3], skip_special_tokens=True) == "hello"
