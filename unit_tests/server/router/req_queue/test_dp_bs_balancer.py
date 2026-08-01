from types import SimpleNamespace

from lightllm.server.router.req_queue.dp_balancer.bs import DpBsBalancer


class FakeReq:
    def __init__(self, input_len, dp_index, shm_cur_kv_len=0):
        self.input_len = input_len
        self.shm_cur_kv_len = shm_cur_kv_len
        self.sample_params = SimpleNamespace(suggested_dp_index=dp_index)


class FakeQueue:
    def __init__(self):
        self.waiting_req_list = []

    def extend(self, reqs):
        self.waiting_req_list.extend(reqs)


class FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs

    def get_req_list_for_dp(self, dp_index):
        return [req for req in self.reqs if req.sample_params.suggested_dp_index == dp_index]


def test_prefill_balancer_uses_remaining_input_tokens():
    queues = [FakeQueue(), FakeQueue()]
    current_batch = FakeBatch(
        [
            FakeReq(160_000, 0, shm_cur_kv_len=20_000),
            FakeReq(1_000, 1),
            FakeReq(1_000, 1),
        ]
    )
    new_group = [FakeReq(10_000, -1)]

    balancer = DpBsBalancer(2, queues, balance_by_input_tokens=True)
    waiting_groups = [new_group]
    balancer.assign_reqs_to_dp(current_batch, waiting_groups)

    assert new_group[0].sample_params.suggested_dp_index == 1
    assert queues[1].waiting_req_list == new_group
    assert waiting_groups == []


def test_decode_balancer_keeps_request_count_behavior():
    queues = [FakeQueue(), FakeQueue()]
    current_batch = FakeBatch(
        [
            FakeReq(160_000, 0),
            FakeReq(1_000, 1),
            FakeReq(1_000, 1),
        ]
    )
    new_group = [FakeReq(10_000, -1)]

    balancer = DpBsBalancer(2, queues)
    balancer.assign_reqs_to_dp(current_batch, [new_group])

    assert new_group[0].sample_params.suggested_dp_index == 0
    assert queues[0].waiting_req_list == new_group
