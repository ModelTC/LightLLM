import pytest
import easydict
from lightllm.server.core.objs.req import Req, TokenHealingReq, ChunkedPrefillReq, SamplingParams
from lightllm.utils.envs_utils import set_env_start_args


@pytest.fixture(scope="module", autouse=True)
def setup_module_env():
    set_env_start_args(
        easydict.EasyDict(
            {
                "mtp_step": 0,
                "llm_prefill_att_backend": ["None"],
                "llm_decode_att_backend": ["None"],
                "cpu_cache_token_page_size": 256,
                "enable_cpu_cache": False,
            }
        )
    )


@pytest.fixture
def req():
    req_instance = Req()
    req_instance.init(1, [1, 2, 3], {"max_new_tokens": 1}, None, chunked_prefill_size=128)
    return req_instance


def test_req_init(req):
    assert req.request_id == 1
    assert req.input_len == 3


def test_create_prompt_ids_shm_array(req):
    assert hasattr(req, "shm_prompt_ids")


def test_get_used_tokens(req):
    req.shm_cur_kv_len = 5
    assert req.get_used_tokens() == 5


def test_token_healing_req_post_init():
    token_healing_req = TokenHealingReq()
    token_healing_req.init(1, [1, 2, 3, 4], {"max_new_tokens": 1}, None)
    assert token_healing_req.sample_params.max_new_tokens == 9


# def test_chunked_req_get_tuple_tokens():
#     chunked_req = ChunkedPrefillReq()
#     chunked_req.init(1, [1, 2, 3], {"max_new_tokens": 1}, None, chunked_prefill_size=256)
#     result = chunked_req.get_tuple_tokens(False, 10)
#     assert isinstance(result, tuple)


def test_finish_status(req):
    req.finish_status.set_status(req.finish_status.FINISHED_STOP)
    assert req.finish_status.is_finished()


if __name__ == "__main__":
    pytest.main()
