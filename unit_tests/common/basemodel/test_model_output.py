import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector, ModelOutput


def test_decode_unpad_slices_spec_output_with_logits():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = ModelOutput(
        logits=torch.arange(24).view(6, 4),
        collector=ModelMtpOutputCollector(spec_hidden=torch.arange(18).view(6, 3)),
    )

    unpadded = model._create_unpad_decode_model_output(output, origin_batch_size=4)

    assert unpadded.logits.shape == (4, 4)
    assert unpadded.collector.spec_hidden.shape == (4, 3)
    # Unpadding returns a shallow output copy and leaves the graph-owned
    # tensors on the original ModelOutput intact.
    assert output.logits.shape == (6, 4)
    assert output.collector.spec_hidden.shape == (6, 3)


def test_prefill_unpad_uses_token_rows_for_spec_hidden():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = ModelOutput(
        logits=torch.arange(20).view(5, 4),
        collector=ModelMtpOutputCollector(spec_hidden=torch.arange(24).view(8, 3)),
        prompt_logics=torch.arange(32).view(8, 4),
    )

    unpadded = model._create_unpad_prefill_model_output(
        output,
        origin_handle_token_num=6,
        origin_batch_size=3,
    )

    assert unpadded.logits.shape == (3, 4)
    assert unpadded.collector.spec_hidden.shape == (6, 3)
    assert unpadded.prompt_logics.shape == (6, 4)
