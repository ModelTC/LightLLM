from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from torch import Tensor

from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from xgrammar import CompiledGrammar, GrammarMatcher

    from lightllm.server.router.model_infer.infer_batch import InferSamplingParams


logger = init_logger(__name__)


@dataclass
class ConstraintState:
    """Request-owned grammar progress; only committed output advances it permanently."""

    # A valid artifact creates its matcher at admission; None denotes a load failure.
    matcher: Optional["GrammarMatcher"]
    in_reasoning: bool
    error: Optional[str] = None
    reasoning_end_token_ids: Tuple[int, ...] = ()
    reasoning_tail: Tuple[int, ...] = ()

    @classmethod
    def from_grammar(cls, grammar: "CompiledGrammar", params: "InferSamplingParams") -> "ConstraintState":
        import xgrammar as xgr

        return cls(
            matcher=xgr.GrammarMatcher(grammar),
            in_reasoning=bool(params.guided_reasoning_end),
            reasoning_end_token_ids=tuple(params.guided_reasoning_end),
        )

    def _get_grammar_start_after_reasoning(self, draft_tokens: Sequence[int]) -> int:
        """Return the zero-based index of the first grammar-constrained prediction.

        Return 0 if reasoning has already ended. If draft token N completes the
        end marker, return N + 1: the marker itself stays unrestricted. With no
        answer predictions, return len(draft_tokens) + 1, the end of the window.

        Scan with the committed marker tail to recognize markers split across
        iterations, without committing drafts or changing reasoning progress.
        """
        if not self.in_reasoning:
            return 0
        tail = self.reasoning_tail
        for draft_index, token_id in enumerate(draft_tokens):
            tail = (tail + (token_id,))[-len(self.reasoning_end_token_ids) :]
            if tail == self.reasoning_end_token_ids:
                return draft_index + 1
        return len(draft_tokens) + 1

    def commit(self, token_id: int) -> None:
        """Advance grammar or reasoning progress with one committed output token.

        Errors stay on this state. The request layer owns FinishStatus and
        reports them without letting one grammar break the overlap threads.
        """
        if self.error is not None:
            return
        try:
            if self.in_reasoning:
                self.reasoning_tail = (self.reasoning_tail + (token_id,))[-len(self.reasoning_end_token_ids) :]
                if self.reasoning_tail == self.reasoning_end_token_ids:
                    self.in_reasoning = False
                    self.reasoning_tail = ()
                # Neither thinking text nor its closing delimiter enters the matcher.
                return
            matcher = self.matcher
            if matcher.is_terminated() or not matcher.accept_token(token_id):
                raise RuntimeError(f"Grammar rejected sampled token {token_id}")
        except Exception as exc:
            self._record_error(exc)

    def is_terminated(self) -> bool:
        return self.matcher is not None and self.matcher.is_terminated()

    def fill_masks(self, bitmask: Tensor, draft_tokens: Sequence[int]) -> bool:
        """Fill this request's verify slots; return whether masking is needed.

        Slot N predicts after N draft tokens. Thinking and unreachable suffixes
        stay unrestricted. The matcher rolls back after exploring draft tokens;
        the request's reasoning progress remains unchanged.
        """
        if self.error is not None:
            return False

        mask_start_index = self._get_grammar_start_after_reasoning(draft_tokens)
        if mask_start_index == bitmask.shape[0]:
            return False
        # Always overwrite the used slots, including previously masked suffixes
        # that a shorter/invalid draft makes unreachable in this iteration.
        bitmask.fill_(-1)

        try:
            matcher = self.matcher
            num_draft_tokens_to_rollback = 0
            try:
                for mtp_index in range(mask_start_index, bitmask.shape[0]):
                    if matcher.is_terminated():
                        break
                    matcher.fill_next_token_bitmask(bitmask, mtp_index)
                    # The last target prediction has no draft token to advance.
                    if mtp_index == len(draft_tokens):
                        break
                    if not matcher.accept_token(draft_tokens[mtp_index]):
                        # This mask already excludes the invalid draft token;
                        # verification will discard all later predictions.
                        break
                    num_draft_tokens_to_rollback += 1
            finally:
                # Every draft advance is temporary, including across reasoning
                # end. Restore committed history even if mask generation fails.
                if num_draft_tokens_to_rollback:
                    matcher.rollback(num_draft_tokens_to_rollback)
        except Exception as exc:
            self._record_error(exc)
            # Ignore any partially written masks. In-flight sampling may finish;
            # post_handle ends this request with an error.
            return False
        return True

    def _record_error(self, exc: Exception) -> None:
        self.error = str(exc)
        logger.exception("Failed to process output grammar")
