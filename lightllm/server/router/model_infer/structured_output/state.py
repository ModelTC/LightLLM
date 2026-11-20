from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generator, Optional, Sequence, Tuple

from torch import Tensor

from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from xgrammar import CompiledGrammar, GrammarMatcher

    from lightllm.server.core.objs.sampling_params import ConstraintOutParams


logger = init_logger(__name__)


@dataclass
class ConstraintState:
    """Request-owned reasoning/grammar progress; only committed output advances it permanently."""

    # Budget-only requests do not need a grammar matcher; errors use the error field.
    matcher: Optional["GrammarMatcher"]
    in_reasoning: bool
    error: Optional[str] = None
    # 思考结束标记及其匹配后缀与 XGrammar matcher 一起由当前请求独立持有。
    reasoning_end_token_ids: Tuple[int, ...] = ()
    reasoning_tail: Tuple[int, ...] = field(default=(), init=False)
    thinking_token_budget: int = -1
    reasoning_tokens: int = field(default=0, init=False)
    # 记录通过统一接口成功推进的 grammar token 数，用于恢复临时状态。
    _grammar_steps: int = field(default=0, init=False)

    @classmethod
    def from_grammar(cls, grammar: Optional["CompiledGrammar"], params: "ConstraintOutParams") -> "ConstraintState":
        matcher = None
        if grammar is not None:
            import xgrammar as xgr

            matcher = xgr.GrammarMatcher(grammar)
        reasoning_end_token_ids = tuple(params.guided_reasoning_end.to_list())
        return cls(
            matcher=matcher,
            in_reasoning=bool(reasoning_end_token_ids),
            reasoning_end_token_ids=reasoning_end_token_ids,
            thinking_token_budget=params.thinking_token_budget,
        )

    def commit(self, token_id: int) -> None:
        """Advance grammar or reasoning progress with one committed output token.

        Errors stay on this state. The request layer owns FinishStatus and
        reports them without letting one grammar break the overlap threads.
        """
        if self.error is not None:
            return
        try:
            if not self._accept_token(token_id):
                raise RuntimeError(f"Grammar rejected sampled token {token_id}")
        except BaseException as exc:
            self._record_error(exc)

    def is_terminated(self) -> bool:
        # 错误和 grammar 正常结束都是终态；请求层优先读取 error，区分错误结束与正常停止。
        if self.error is not None:
            return True
        if self.in_reasoning:
            return False
        return self.matcher is not None and self.matcher.is_terminated()

    def fill_masks(self, bitmask: Tensor, draft_tokens: Sequence[int]) -> bool:
        """Fill this request's verify slots; return whether masking is needed.

        Slot N predicts after N draft tokens. Thinking and unreachable suffixes
        stay unrestricted. The matcher rolls back after exploring draft tokens;
        the request's reasoning progress remains unchanged.
        """
        # 每轮先清除旧掩码；思考位置和不可达后缀默认全允许，grammar 再覆盖对应行。
        bitmask.fill_(-1)
        if self.error is not None:
            return False

        try:
            enabled = False
            draft_token_count = len(draft_tokens)
            with self._temporary_state():
                # 每个 draft token 对应一行 target 预测，末尾额外预测一个后续 token。
                # draft_tokens 已排除当前已提交的 token；第 i 行预测的是 draft_tokens[i]。
                for mtp_index in range(draft_token_count + 1):
                    # 1. 统一接口按当前位置的 reasoning 或 grammar 状态填充预测行。
                    row_mask_enabled = self._fill_next_token_bitmask(bitmask, mtp_index)
                    enabled |= row_mask_enabled

                    # 2. grammar 已结束，后续预测不再可达。
                    if self.is_terminated():
                        break
                    # 3. 只在有对应 draft 时推进状态；最后一行填完 mask 后自然结束。
                    if mtp_index < draft_token_count:
                        next_draft_token = draft_tokens[mtp_index]
                        draft_accepted = self._accept_token(next_draft_token)
                        if not draft_accepted:
                            # 当前掩码已排除这个非法 draft，验证会在此拒绝并丢弃后续预测。
                            break
        except BaseException as exc:
            self._record_error(exc)
            # Ignore any partially written masks. In-flight sampling may finish;
            # post_handle ends this request with an error.
            return False
        return enabled

    def _accept_token(self, token_id: int) -> bool:
        """Advance the combined reasoning/grammar state with one token."""
        if self.is_terminated():
            return False
        match self.in_reasoning:
            case True:
                forced_token = self._forced_reasoning_token()
                if forced_token is not None:
                    if token_id != forced_token:
                        return False
                elif self.thinking_token_budget >= 0:
                    self.reasoning_tokens += 1
                self.reasoning_tail = (self.reasoning_tail + (token_id,))[-len(self.reasoning_end_token_ids) :]
                if self.reasoning_tail == self.reasoning_end_token_ids:
                    self.in_reasoning = False
                # 思考内容和结束标记只推进 reasoning 状态，不进入 grammar matcher。
                return True
            case False:
                if self.matcher is None:
                    return True
                accepted = self.matcher.accept_token(token_id)
                if accepted:
                    self._grammar_steps += 1
                return accepted

    def _fill_next_token_bitmask(self, bitmask: Tensor, index: int) -> bool:
        """Fill one prediction row; report whether this row needs grammar masking."""
        match self.in_reasoning:
            case True:
                forced_token = self._forced_reasoning_token()
                if forced_token is not None:
                    bitmask[index].zero_()
                    word, bit = divmod(forced_token, 32)
                    bitmask[index, word] = (1 << bit) if bit < 31 else -(1 << 31)
                    return True
                # 预算未耗尽或不限预算时，当前预测行保持全允许。
                bitmask[index].fill_(-1)
                return False
            case False:
                if self.matcher is None or self.is_terminated():
                    # 无正文 grammar 或 grammar 已结束时保持全允许。
                    bitmask[index].fill_(-1)
                    return False
                self.matcher.fill_next_token_bitmask(bitmask, index)
                return True

    def _forced_reasoning_token(self) -> Optional[int]:
        if self.thinking_token_budget < 0 or self.reasoning_tokens < self.thinking_token_budget:
            return None
        # 结束标记可能只生成了一部分；MTP 验证和 PD 续跑时接着补完。
        end_tokens = self.reasoning_end_token_ids
        matched_len = min(len(self.reasoning_tail), len(end_tokens) - 1)
        while matched_len > 0:
            if self.reasoning_tail[-matched_len:] == end_tokens[:matched_len]:
                break
            matched_len -= 1
        return end_tokens[matched_len]

    @contextmanager
    def _temporary_state(self) -> Generator[None, None, None]:
        """Restore reasoning and grammar progress after speculative traversal."""
        # 保存已提交的 reasoning 阶段和匹配后缀。tail 是不可变 tuple，直接保存引用即可；
        # 临时推进会生成新 tuple，不会修改快照，所以只需保留结束标记长度的后缀。
        saved_in_reasoning = self.in_reasoning
        saved_tail = self.reasoning_tail
        saved_reasoning_tokens = self.reasoning_tokens
        saved_grammar_steps = self._grammar_steps
        try:
            yield
        finally:
            try:
                # matcher 本身不复制：只回滚本轮成功接受的 grammar draft token，
                # 已提交 token 的进度保持不变；reasoning token 不计入该步数。
                grammar_steps = self._grammar_steps - saved_grammar_steps
                if grammar_steps:
                    self.matcher.rollback(grammar_steps)
            except BaseException as exc:
                # 回滚失败后 matcher 已无法可靠复用，立即标记错误终态。
                self._record_error(exc)
            finally:
                # 无论遍历是否跨过思考结束标记，都恢复进入前的阶段、tail 和计数。
                # matcher 回滚若失败，仍恢复这些 Python 状态，异常交给请求错误路径处理。
                self.in_reasoning = saved_in_reasoning
                self.reasoning_tail = saved_tail
                self.reasoning_tokens = saved_reasoning_tokens
                self._grammar_steps = saved_grammar_steps

    def _record_error(self, exc: BaseException) -> None:
        self.error = str(exc)
        logger.exception("Failed to process output grammar: %s", self.error)
