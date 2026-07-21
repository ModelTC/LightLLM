import importlib.util

from lightllm.common.basemodel import PostLayerInferTpl, PreLayerInferTpl
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import (
    Gemma_2bTransformerLayerInfer,
)
from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import (
    Gemma_2bTransformerLayerWeight,
)
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.models.llama.layer_weights.transformer_layer_weight import (
    LlamaTransformerLayerWeight,
)
from lightllm.models.pi0.infer_struct import Pi0ActionInferStateInfo
from lightllm.models.pi0.layer_infer import (
    Pi0ActionPostLayerInfer,
    Pi0ActionPreLayerInfer,
    Pi0ActionTransformerLayerInfer,
)
from lightllm.models.pi0.layer_weights import (
    Pi0ActionTransformerLayerWeight,
    Pi0VLMTransformerLayerWeight,
)
from lightllm.models.pi0.model import Pi0VLMModel


def test_vlm_is_a_native_gemma_basemodel():
    assert issubclass(Pi0VLMModel, Gemma_2bTpPartModel)
    assert Pi0VLMModel.transformer_layer_infer_class is Gemma_2bTransformerLayerInfer
    assert Pi0VLMModel._init_req_manager is TpPartBaseModel._init_req_manager
    assert Pi0VLMModel._init_mem_manager is Gemma_2bTpPartModel._init_mem_manager
    assert Pi0VLMModel.prefill_causal is False


def test_pi0_weights_use_standard_meta_weight_containers():
    assert issubclass(Pi0VLMTransformerLayerWeight, LlamaTransformerLayerWeight)
    assert issubclass(Pi0VLMTransformerLayerWeight, Gemma_2bTransformerLayerWeight)
    assert issubclass(Pi0ActionTransformerLayerWeight, Gemma_2bTransformerLayerWeight)


def test_action_expert_uses_standard_layer_infer_components():
    assert issubclass(Pi0ActionInferStateInfo, InferStateInfo)
    assert issubclass(Pi0ActionPreLayerInfer, PreLayerInferTpl)
    assert issubclass(Pi0ActionPostLayerInfer, PostLayerInferTpl)
    assert issubclass(Pi0ActionTransformerLayerInfer, Gemma_2bTransformerLayerInfer)


def test_pi0_has_no_parallel_attention_or_kv_runtime_modules():
    for module_name in (
        "action_cuda_graph",
        "attention_plan",
        "kv_cache",
        "runtime_ops",
    ):
        assert importlib.util.find_spec(f"lightllm.models.pi0.{module_name}") is None


def test_infer_state_keeps_text_prefill_causal_by_default():
    state = InferStateInfo()
    assert state.prefill_causal is True
    assert state.use_ieee_fp32_attention is False
