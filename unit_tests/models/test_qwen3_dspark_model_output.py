import torch

from lightllm.common.basemodel import batch_objs
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.qwen3_dspark import model_output as dspark_model_output
from lightllm.models.qwen3_dspark.model import DSparkModelOutputMixin
from lightllm.models.qwen3_dspark.model_output import DSparkModelOutput


class _DSparkTestModel(DSparkModelOutputMixin, TpPartBaseModel):
    pass


def test_dspark_decode_unpad_preserves_output_type_and_slices_dspark_fields():
    model = _DSparkTestModel.__new__(_DSparkTestModel)
    output = DSparkModelOutput(
        logits=torch.arange(48).view(12, 4),
        spec_hidden=torch.arange(36).view(12, 3),
        confidence_logits=torch.arange(12).view(3, 4),
        draft_token_ids=torch.arange(12),
    )

    unpadded = model._create_unpad_decode_model_output(output, origin_batch_size=8)

    assert isinstance(unpadded, DSparkModelOutput)
    assert unpadded.logits.shape == (8, 4)
    assert unpadded.spec_hidden.shape == (8, 3)
    assert unpadded.confidence_logits.shape == (2, 4)
    assert unpadded.draft_token_ids.shape == (8,)
    assert output.logits.shape == (12, 4)
    assert output.confidence_logits.shape == (3, 4)
    assert output.draft_token_ids.shape == (12,)


def test_dspark_no_ref_conversion_dispatches_to_dspark_fields(monkeypatch):
    monkeypatch.setattr(batch_objs, "tensor_to_no_ref_tensor", torch.clone)
    monkeypatch.setattr(dspark_model_output, "tensor_to_no_ref_tensor", torch.clone)
    output = DSparkModelOutput(
        logits=torch.ones((2, 4)),
        spec_hidden=torch.ones((2, 3)),
        confidence_logits=torch.ones((1, 2)),
        draft_token_ids=torch.ones((2,), dtype=torch.int64),
    )
    original_ptrs = (
        output.logits.data_ptr(),
        output.spec_hidden.data_ptr(),
        output.confidence_logits.data_ptr(),
        output.draft_token_ids.data_ptr(),
    )

    output.to_no_ref_tensor()

    converted_ptrs = (
        output.logits.data_ptr(),
        output.spec_hidden.data_ptr(),
        output.confidence_logits.data_ptr(),
        output.draft_token_ids.data_ptr(),
    )
    assert all(converted != original for converted, original in zip(converted_ptrs, original_ptrs))
