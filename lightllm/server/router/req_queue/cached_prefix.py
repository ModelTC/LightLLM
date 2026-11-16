import os

from .peak_reservation import Reservation, estimate_peak


class CachedPrefixReservations:
    def __init__(self, args, dp_index):
        self.page_tokens = int(args.linear_att_hash_page_size)
        name = os.environ.get("LIGHTLLM_UNIQUE_SERVICE_NAME_ID")
        self.domain = (name + ":" + str(dp_index) + ":" + str(self.page_tokens)).encode() if name else None
        self.memo, self.by_tuple = {}, {}
        self.shared_prefix_tokens = 0

    def reset(self, requests):
        active = {int(request.request_id) for request in requests}
        self.memo = {identity: value for identity, value in self.memo.items() if identity in active}
        self.by_tuple.clear()

    def bind(self, reservation, request):
        self.by_tuple[id(reservation)] = self.prefix(request)
        return reservation

    def prefix(self, request):
        try:
            mm = getattr(request, "multimodal_params", None)
            if (
                self.domain is None
                or mm is None
                or mm.images
                or mm.audios
                or request.is_paused
                or request.is_aborted
                or request.sample_params.disable_prompt_cache
            ):
                return ()
            length = min(int(request.prompt_cache_len), int(request.input_len), int(request.shm_cur_kv_len))
            hashes = request.linear_att_token_hash_list.get_all() if length >= self.page_tokens else []
            count = min(length // self.page_tokens, len(hashes), 32)
            if count <= 0:
                return ()
            hashes = tuple(int(value) & ((1 << 64) - 1) for value in hashes[:count])
            identity = (int(request.request_id), count, hashes, int(request.input_len))
            saved = self.memo.get(int(request.request_id))
            if saved is not None and saved[0] == identity:
                return saved[1]
            if not hasattr(request, "shm_prompt_ids"):
                request.link_prompt_ids_shm_array()
            ids = request.shm_prompt_ids.arr
            if ids is None or ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < count * self.page_tokens:
                return ()
            pages = tuple(
                hashes[index].to_bytes(8, "little")
                + ids[index * self.page_tokens : (index + 1) * self.page_tokens].astype("<i8", copy=False).tobytes()
                for index in range(count)
            )
            self.memo[int(request.request_id)] = (identity, pages)
            return pages
        except (AttributeError, ValueError, IndexError, OSError, TypeError, AssertionError):
            return ()

    def calculate(self, reservations):
        entries = [
            Reservation(int(pair[0]), int(pair[1]), self.by_tuple.get(id(pair), ()), self.domain)
            for pair in reservations
        ]
        try:
            peak = estimate_peak(entries, self.page_tokens)
        except ValueError:
            peak = estimate_peak([Reservation(int(pair[0]), int(pair[1])) for pair in reservations], self.page_tokens)
        self.shared_prefix_tokens = peak.max_prefix_credit
        return peak.shared_tokens
