import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Tuple

from lightllm.utils.config_utils import get_eos_token_ids, get_vocab_size
from lightllm.utils.grammar_utils import create_tokenizer_info
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def create_output_grammar_compiler(args, tokenizer):
    if args.output_constraint_mode != "xgrammar":
        return None
    # PD master starts without loading a model, so it may not have resolved
    # --eos_id yet. Use the same model-config fallback as inference startup.
    eos_ids = args.eos_id if args.eos_id is not None else get_eos_token_ids(args.model_dir)
    return OutputGrammarCompiler(
        tokenizer, get_vocab_size(args.model_dir), eos_ids, timeout=args.grammar_compile_timeout
    )


class OutputGrammarCompiler:
    """Compile and serialize grammars before requests enter the inference pipeline."""

    def __init__(
        self, tokenizer, vocab_size: int, eos_ids: Sequence[int], timeout: float = 30.0, cache_size: int = 200
    ):
        import xgrammar as xgr

        self.timeout = timeout
        self.vocab_size = vocab_size
        self.tokenizer_info = create_tokenizer_info(tokenizer, vocab_size, list(eos_ids))
        self._compiler = xgr.GrammarCompiler(self.tokenizer_info, max_threads=2, cache_enabled=False)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="grammar-compile")
        # Only completed artifacts are cached; the HTTP event loop owns this cache.
        self._cache: OrderedDict[Tuple[str, str], bytes] = OrderedDict()
        self._cache_size = cache_size

    def _compile(self, kind: str, value: str) -> bytes:
        if kind == "json":
            grammar = self._compiler.compile_json_schema(value)
        elif kind == "regex":
            grammar = self._compiler.compile_regex(value)
        elif kind == "grammar":
            if value == "json":
                grammar = self._compiler.compile_json_schema('{"type":"object"}')
            else:
                grammar = self._compiler.compile_grammar(value)
        else:
            raise ValueError(f"Unsupported constraint kind: {kind}")
        return grammar.serialize_json().encode("utf-8")

    async def compile(self, kind: str, value: str) -> bytes:
        key = (kind, value)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        try:
            loop = asyncio.get_running_loop()
            # Each cache miss owns its job, so timeout/cancellation affects only this request.
            payload = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._compile, kind, value), timeout=self.timeout
            )
        except asyncio.TimeoutError as exc:
            raise ValueError(f"{kind} grammar compilation timed out after {self.timeout:g}s") from exc
        except Exception as exc:
            logger.exception("Failed to compile %s output grammar", kind)
            raise ValueError(f"Failed to compile {kind} grammar: {exc}") from exc

        self._cache[key] = payload
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return payload

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
