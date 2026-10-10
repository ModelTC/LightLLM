from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence

from lightllm.utils.grammar_utils import create_tokenizer_info
from lightllm.utils.log_utils import init_logger
from .state import ConstraintState

if TYPE_CHECKING:
    from xgrammar import CompiledGrammar
    from lightllm.server.core.objs import Req
    from lightllm.server.router.model_infer.infer_batch import InferSamplingParams

logger = init_logger(__name__)


class OutputGrammarCache:
    """Load HTTP-compiled artifacts; each request owns its independent matcher."""

    def __init__(self, tokenizer, vocab_size: int, eos_ids: Sequence[int], cache_size: int = 200):
        self.tokenizer_info = create_tokenizer_info(tokenizer, vocab_size, list(eos_ids))
        self.get_grammar = lru_cache(maxsize=cache_size)(self._deserialize)

    def _deserialize(self, payload: bytes) -> "CompiledGrammar":
        import xgrammar as xgr

        return xgr.CompiledGrammar.deserialize_json(payload.decode("utf-8"), self.tokenizer_info)

    def create_state(self, shm_req: "Req", sampling_params: "InferSamplingParams") -> Optional[ConstraintState]:
        try:
            payload = shm_req.get_compiled_grammar()
            if not payload:
                return None
            state = ConstraintState.from_grammar(self.get_grammar(payload), sampling_params)
            previous_output_len = sampling_params.shm_param.pd_previous_output_len
            # A new PD segment includes earlier output in its prompt. Replay only
            # that suffix to restore reasoning delimiters and grammar progress;
            # the current segment's first token is committed by normal post_handle.
            start = shm_req.input_len - previous_output_len
            for token_id in shm_req.shm_prompt_ids.arr[start : shm_req.input_len]:
                state.commit(int(token_id))
            return state
        except Exception as exc:
            # Internal transport/version errors use the existing request error
            # path. The inference worker never retries compilation.
            logger.exception("Failed to initialize output grammar")
            return ConstraintState(matcher=None, in_reasoning=False, error=str(exc))
