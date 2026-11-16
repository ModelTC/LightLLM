import dataclasses

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import BaseDecodeAttState, BasePrefillAttState
from lightllm.common.basemodel.attention.dense.fa3 import Fa3DecodeAttState, Fa3PrefillAttState
from lightllm.common.basemodel.attention.dense.fa3_fp8 import Fp8Fa3DecodeAttState, Fp8Fa3PrefillAttState
from lightllm.common.basemodel.attention.dense.flashinfer import FlashInferDecodeAttState, FlashInferPrefillAttState
from lightllm.common.basemodel.attention.dense.triton import TritonDecodeAttState, TritonPrefillAttState
from lightllm.common.basemodel.attention.dense.triton_int8kv import Int8kvTritonPrefillAttState
from lightllm.common.basemodel.attention.linear.gdn import LinearAttDecodeAttState, LinearAttPrefillAttState
from lightllm.common.basemodel.attention.linear.kda import KDADecodeAttState, KDAPrefillAttState
from lightllm.common.basemodel.attention.linear.replay_gdn import ReplayGDNDecodeAttState, ReplayGDNPrefillAttState
from lightllm.common.basemodel.attention.linear.replay_kda import ReplayKDADecodeAttState, ReplayKDAPrefillAttState
from lightllm.common.basemodel.attention.mla.fa3 import MlaFa3DecodeAttState, MlaFa3PrefillAttState
from lightllm.common.basemodel.attention.mla.flashinfer import MlaFlashInferPrefillAttState
from lightllm.common.basemodel.attention.mla.triton import MlaTritonDecodeAttState
from lightllm.common.basemodel.attention.neo.fa3 import NeoFa3PrefillAttState
from lightllm.common.basemodel.attention.neo.triton import NeoTritonPrefillAttState
from lightllm.common.basemodel.attention.nsa import dsv4_flashmla_sparse_fp8 as dsv4
from lightllm.common.basemodel.attention.nsa.flashmla_sparse import (
    NsaFlashMlaSparseDecodeAttState,
    NsaFlashMlaSparsePrefillAttState,
)
from lightllm.common.basemodel.attention.nsa.flashmla_sparse_fp8 import NsaFlashMlaFp8SparsePrefillAttState
from lightllm.common.basemodel.attention.nsa.glm5_next import Glm5NextSparseDecodeState, Glm5NextSparsePrefillState
from lightllm.common.basemodel.attention.params import (
    DenseAttParams,
    Dsv4NsaAttParams,
    GdnAttParams,
    KdaAttParams,
    MlaAttParams,
    NsaDecodeAttParams,
    NsaPrefillAttParams,
)


def _state(state_class):
    # _check_params only reads class attributes, so skip the GPU-backed construction.
    return object.__new__(state_class)


SINK = DenseAttParams(sink_weight=torch.zeros(1))
WINDOW = DenseAttParams(sliding_window=(3, 0))
ALIBI = DenseAttParams(alibi=torch.zeros(1))


def test_enabled_features_lists_only_set_fields():
    assert DenseAttParams().enabled_features() == ()
    assert DenseAttParams(sliding_window=(3, 0), alibi=torch.zeros(1)).enabled_features() == ("sliding_window", "alibi")
    # The logical layer id selects FP8 calibration data; it is not an optional feature.
    assert DenseAttParams(logical_layer_id=3).enabled_features() == ()


def test_params_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        MlaAttParams(softmax_scale=1.0).softmax_scale = 2.0


@pytest.mark.parametrize(
    "state_class, accepted, rejected",
    [
        (TritonPrefillAttState, [WINDOW, ALIBI], [SINK]),
        (TritonDecodeAttState, [WINDOW, ALIBI], [SINK]),
        (Fa3PrefillAttState, [WINDOW, SINK], [ALIBI]),
        (Fa3DecodeAttState, [WINDOW, SINK], [ALIBI]),
        (FlashInferPrefillAttState, [], [WINDOW, SINK, ALIBI]),
        (FlashInferDecodeAttState, [], [WINDOW, SINK, ALIBI]),
        (Int8kvTritonPrefillAttState, [], [WINDOW, SINK, ALIBI]),
        # These inherit from a state with optional features without implementing them.
        (Fp8Fa3PrefillAttState, [], [WINDOW, SINK, ALIBI]),
        (Fp8Fa3DecodeAttState, [], [WINDOW, SINK, ALIBI]),
        (NeoFa3PrefillAttState, [], [WINDOW, SINK, ALIBI]),
        (NeoTritonPrefillAttState, [], [WINDOW, SINK, ALIBI]),
    ],
)
def test_dense_states_reject_features_they_do_not_implement(state_class, accepted, rejected):
    state = _state(state_class)

    assert state._check_params(None) == DenseAttParams()
    for params in accepted:
        assert state._check_params(params) is params
    for params in rejected:
        with pytest.raises(NotImplementedError, match=params.enabled_features()[0]):
            state._check_params(params)


MLA = MlaAttParams(softmax_scale=0.5)
NSA_PREFILL = NsaPrefillAttParams(topk_mem_indices=torch.zeros(1), softmax_scale=0.5, kv_lora_rank=4)
NSA_DECODE = NsaDecodeAttParams(topk_mem_indices=torch.zeros(1), softmax_scale=0.5, kv_lora_rank=4, qk_rope_head_dim=2)
DSV4 = Dsv4NsaAttParams(
    compress_ratio=0,
    head_dim_v=4,
    softmax_scale=0.5,
    attn_sink=torch.zeros(1),
    swa_indices=torch.zeros(1),
    swa_lengths=torch.zeros(1),
)
GDN = GdnAttParams(mixed_qkvzba=torch.zeros(1), layer_weight=None, layer_num=0)
KDA = KdaAttParams(
    mixed_qkv=torch.zeros(1), raw_gate=torch.zeros(1), raw_beta=torch.zeros(1), layer_weight=None, layer_num=0
)


@pytest.mark.parametrize(
    "state_class, accepted",
    [
        (MlaFa3PrefillAttState, MLA),
        (MlaFa3DecodeAttState, MLA),
        (MlaFlashInferPrefillAttState, MLA),
        (MlaTritonDecodeAttState, MLA),
        (NsaFlashMlaSparsePrefillAttState, NSA_PREFILL),
        (NsaFlashMlaSparseDecodeAttState, NSA_DECODE),
        (NsaFlashMlaFp8SparsePrefillAttState, NSA_PREFILL),
        (Glm5NextSparsePrefillState, NSA_PREFILL),
        (Glm5NextSparseDecodeState, NSA_DECODE),
        (LinearAttPrefillAttState, GDN),
        (LinearAttDecodeAttState, GDN),
        (KDAPrefillAttState, KDA),
        (KDADecodeAttState, KDA),
        (ReplayGDNPrefillAttState, GDN),
        (ReplayGDNDecodeAttState, GDN),
        (ReplayKDAPrefillAttState, KDA),
        (ReplayKDADecodeAttState, KDA),
        (dsv4._PrefillAttState, DSV4),
        (dsv4._DecodeAttState, DSV4),
    ],
)
def test_family_states_accept_only_their_own_params_type(state_class, accepted):
    state = _state(state_class)

    assert state._check_params(accepted) is accepted
    for other in (None, DenseAttParams(), MLA, NSA_PREFILL, NSA_DECODE, DSV4, GDN, KDA):
        if type(other) is type(accepted):
            continue
        with pytest.raises(TypeError, match=type(accepted).__name__):
            state._check_params(other)


def _concrete_state_classes(base):
    for subclass in base.__subclasses__():
        yield subclass
        yield from _concrete_state_classes(subclass)


def test_every_state_declares_a_known_params_type():
    known = {
        DenseAttParams,
        MlaAttParams,
        NsaPrefillAttParams,
        NsaDecodeAttParams,
        Dsv4NsaAttParams,
        GdnAttParams,
        KdaAttParams,
    }
    state_classes = set(_concrete_state_classes(BasePrefillAttState)) | set(_concrete_state_classes(BaseDecodeAttState))
    assert len(state_classes) >= 30

    for state_class in state_classes:
        assert state_class.params_type in known, state_class.__name__
        assert state_class.supported_features <= {"sliding_window", "sink_weight", "alibi"}, state_class.__name__
        if state_class.params_type is not DenseAttParams:
            assert state_class.supported_features == frozenset(), state_class.__name__
