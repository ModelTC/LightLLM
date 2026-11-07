from .impl import ChunkedPrefillQueue
from ..cached_prefix import CachedPrefixReservations


def supports_cached_prefix_reservations(args):
    return (
        args.output_constraint_mode == "none"
        and not getattr(args, "first_token_constraint_mode", False)
        and args.dp == 1
        and args.nnodes == 1
        and args.run_mode == "normal"
        and not args.enable_rl
        and not args.disable_dynamic_prompt_cache
        and not args.enable_cpu_cache
        and args.linear_att_page_block_num == 1
        and type(args.linear_att_hash_page_size) is int
        and args.linear_att_hash_page_size > 0
    )


class CachedPrefixChunkedPrefillQueue(ChunkedPrefillQueue):
    def __init__(self, args, router, dp_index, dp_size_in_node):
        super().__init__(args, router, dp_index, dp_size_in_node)
        self.prefix_reservations = CachedPrefixReservations(args, dp_index)

    def _init_cache_list(self, current_batch, is_busy):
        requests = (
            []
            if current_batch is None
            else [
                request for request in current_batch.reqs if request.sample_params.suggested_dp_index == self.dp_index
            ]
        )
        self.prefix_reservations.reset(requests)
        super()._init_cache_list(current_batch, is_busy)

    def _get_req_reservation(self, request, is_busy):
        reservation = super()._get_req_reservation(request, is_busy)
        return self.prefix_reservations.bind(reservation, request)

    def _estimate_peak_tokens(self):
        return self.prefix_reservations.calculate(self.cache_len_list)
