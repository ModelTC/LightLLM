import pytest

from lightllm.models.deepseek_v41.model import DeepseekV41TpPartModel, DeepseekV41VisionTpPartModel
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("model_class", [DeepseekV41TpPartModel, DeepseekV41VisionTpPartModel], ids=["text", "vision"])
@pytest.mark.parametrize("mtp_mode", [None, "dspark", "vanilla_with_att"])
def test_speculative_decoding_is_rejected_during_parameter_validation(model_class, mtp_mode):
    model = model_class.__new__(model_class)
    model.args = StartArgs(mtp_mode=mtp_mode)
    model.load_way = "HF"
    model.tp_world_size_ = 1
    model.enable_tpsp_mix_mode = False
    model.config = {"num_attention_heads": 64, "o_groups": 8, "index_n_heads": 32, "candidate_block_size": 8}

    if mtp_mode is None:
        model._verify_params()
    else:
        with pytest.raises(ValueError, match="DeepSeek-V4.1 does not support speculative decoding"):
            model._verify_params()
