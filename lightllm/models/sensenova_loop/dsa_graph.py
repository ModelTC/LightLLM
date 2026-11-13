"""DSA decode graphs keyed by padded batch AND Indexer score-width bucket."""
import copy
from lightllm.common.basemodel.cuda_graph import CudaGraph


def score_bucket(length, capacity, minimum=2048):
    if length < 0 or length > capacity:
        raise ValueError(f"DSA KV length {length} exceeds capacity {capacity}")
    return min(capacity, max(minimum, 1 << (max(1, length) - 1).bit_length()))


class DsaCudaGraph(CudaGraph):
    """Keep FA3's existing global page-table stride; only bucket DSA work.

    Native replay copies tensor values but not Python lengths. Each score
    width therefore owns separate captured graphs, preventing stale-length
    truncation when a request crosses a bucket boundary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs_by_score_width = {}
        self.pending_score_width = score_bucket(2, self.graph_max_len_in_batch)
        self._startup_warmup = False
        self._building_buckets = False

    def can_run(self, batch_size, max_len_in_batch):
        ok = super().can_run(batch_size, max_len_in_batch)
        if ok:
            self.pending_score_width = score_bucket(
                2 if self._startup_warmup else max_len_in_batch, self.graph_max_len_in_batch
            )
        return ok

    def need_capture(self, batch_size):
        size = self.find_closest_graph_batch_size(batch_size)
        return size not in self.graphs_by_score_width.get(self.pending_score_width, {})

    def _select(self, infer_state, infer_state1=None):
        states = [x for x in (infer_state, infer_state1) if x is not None]
        width = (
            score_bucket(2, self.graph_max_len_in_batch)
            if self._startup_warmup
            else max(x.dsa_score_max_keys for x in states)
        )
        for state in states:
            state.dsa_score_max_keys = width
        self.graph = self.graphs_by_score_width.setdefault(width, {})
        return width

    def capture_decode(self, decode_func, infer_state, infer_state1=None):
        if not (self._startup_warmup or self._building_buckets):
            # Native capture executes warmup/capture/replay. Repeating that on
            # a live hybrid request can update recurrent state more than once.
            raise RuntimeError("DSA length buckets must be captured on startup dummy inputs")
        self._select(infer_state, infer_state1)
        return super().capture_decode(decode_func, infer_state, infer_state1)

    def replay(self, infer_state, infer_state1=None):
        self._select(infer_state, infer_state1)
        return super().replay(infer_state, infer_state1)

    def warmup(self, model):
        # Native warmup supplies graph_max_len as metadata but real dummy
        # sequences have length 2. Capture only the first bucket at startup.
        self._startup_warmup = True
        try:
            super().warmup(model)
        finally:
            self._startup_warmup = False
        self._prewarm_remaining_buckets(model, overlap=False)

    def warmup_overlap(self, model):
        self._startup_warmup = True
        try:
            super().warmup_overlap(model)
        finally:
            self._startup_warmup = False
        self._prewarm_remaining_buckets(model, overlap=True)

    def _prewarm_remaining_buckets(self, model, *, overlap):
        first = score_bucket(2, self.graph_max_len_in_batch)
        seeds = dict(self.graphs_by_score_width[first])
        widths = []
        width = first
        while width < self.graph_max_len_in_batch:
            width = min(width * 2, self.graph_max_len_in_batch)
            widths.append(width)
        self._building_buckets = True
        try:
            # Reuse the native HOLD_REQUEST_ID states and their input/FA3
            # tensors. No live request exists at startup. Sharing these tensors
            # avoids allocating a full-capacity page table for every bucket.
            for width in reversed(widths):
                for batch in sorted(seeds, reverse=True):
                    seed = seeds[batch]
                    state = copy.copy(seed[1])
                    state.dsa_score_max_keys = width
                    state1 = copy.copy(seed[2]) if overlap else None
                    if state1 is not None:
                        state1.dsa_score_max_keys = width
                    fn = model._overlap_tpsp_token_forward if overlap else model._token_forward
                    self.capture_decode(fn, state, state1)
        finally:
            self._building_buckets = False
            self.graph = self.graphs_by_score_width[first]
