from lightllm.common.req_manager import ReqManager
from lightllm.utils.torch_memory_saver_utils import MemoryTag


def initialize_model_runtime(models):
    models = tuple(models)
    main = models[0]
    args = main.args
    with main.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
        requests = ReqManager(
            max_request_num=main.max_req_num,
            max_sequence_length=max(model.get_req_manager_max_seq_len() for model in models),
        )
    for model in models:
        model.bind_req_manager(requests)
        model.prepare_attention_states()
    manager = main.att_state_manager
    manager.finish_state_registration()
    for model in models:
        model.prepare_attention_backends()
        model.attention.reserve(model)
    with main.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
        manager.load_full_att_calibration(args)
        manager.initialize_storage(args, main.max_total_token_num, main.mem_fraction, device=main.device_runtime.device)
    for model in models:
        model.init_runtime()
